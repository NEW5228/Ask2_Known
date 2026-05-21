import cv2
import numpy as np
from ask2know.features.basic_features import extract_features, extract_features_from_image
from ask2know.features.deep_adapter import DeepFeatureAdapter, DEFAULT_DEEP_FEATURE_NAME
from ask2know.concepts.basic_concepts import concepts_from_features, concept_similarity

CONCEPTS_BY_GROUP = {
    'color': {
        'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'pink',
        'brown', 'black', 'white', 'gray', 'color_family', 'dark', 'bright',
    },
    'shape': {
        'round', 'elongated', 'pear_like', 'rectangular_like', 'single_object',
        'cluster_like', 'repeated_parts', 'pet_outline', 'top_ear_like',
    },
    'texture': {'smooth_surface', 'texture_rich', 'edge_rich', 'cluster_like', 'repeated_parts', 'fur_like'},
    'surface': {'fuzzy_surface', 'rough_peel', 'speckled_surface', 'glossy_surface', 'fur_like'},
    'part': {
        'peel_like', 'flesh_like', 'cut_surface', 'seed_like', 'core_like',
        'segment_like', 'rind_like', 'face_like', 'eye_pair_like',
        'muzzle_like', 'nose_like', 'whisker_like', 'top_ear_like',
    },
    'quality': {'clear_foreground', 'background_interference'},
    'text': {'text_like', 'character_parts'},
    'sign': {'sign_like', 'arrow_like', 'prohibition_like'},
}


def _mean_vectors(vectors):
    if not vectors:
        return None
    return np.mean(np.stack(vectors), axis=0)


def _l2_normalize(vec):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm


def _hist_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    hist_len = max(0, len(a) - 20)  # HSV hist + 6 stats + 14 coarse color bins
    if hist_len > 0:
        ah = a[:hist_len]
        bh = b[:hist_len]
        inter = float(np.minimum(ah, bh).sum()) / (float(ah.sum()) + 1e-8)
        stat_dist = float(np.linalg.norm(a[hist_len:] - b[hist_len:])) if len(a) > hist_len else 0.0
        stat_sim = float(np.exp(-2.4 * stat_dist))
        return max(0.0, min(1.0, 0.68 * inter + 0.32 * stat_sim))
    return _vector_similarity(a, b, scale=2.0)


def _vector_similarity(a, b, scale=2.5):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        n = min(a.size, b.size)
        a = a.flatten()[:n]
        b = b.flatten()[:n]
    dist = float(np.linalg.norm(a - b)) / max(1.0, np.sqrt(float(a.size)))
    sim = float(np.exp(-scale * dist))
    return max(0.0, min(1.0, sim))


def _cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size != b.size:
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    cos = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, (cos + 1.0) * 0.5))


def _augmented_images(path, config):
    img = cv2.imread(str(path))
    if img is None:
        return []
    if not config or not config.get('enable', False):
        return []
    imgs = []
    h, w = img.shape[:2]
    if config.get('brightness', True):
        imgs.append(cv2.convertScaleAbs(img, alpha=1.0, beta=22))
        imgs.append(cv2.convertScaleAbs(img, alpha=1.0, beta=-22))
    if config.get('rotation', True):
        for angle in (-8, 8):
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            imgs.append(cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT))
    if config.get('crop', True):
        mx, my = int(w * 0.05), int(h * 0.05)
        if w > 2 * mx and h > 2 * my:
            crop = img[my:h-my, mx:w-mx]
            imgs.append(cv2.resize(crop, (w, h)))
    if config.get('blur', False):
        imgs.append(cv2.GaussianBlur(img, (3, 3), 0))
    return imgs[:6]


class PrototypeModel:
    def __init__(
        self,
        feature_names,
        augmentation_config=None,
        concept_config=None,
        system_feature_names=None,
        feature_groups=None,
        similarity_config=None,
        deep_feature_config=None,
        deep_cache_dir=None,
    ):
        self.feature_names = feature_names
        self.system_feature_names = list(system_feature_names or [])
        self.feature_groups = feature_groups or {}
        self.augmentation_config = augmentation_config or {'enable': False}
        self.concept_config = concept_config or {'enable': True, 'score_weight': 0.25}
        self.concepts_enabled = self.concept_config.get('enable', True)
        self.concept_score_weight = float(self.concept_config.get('score_weight', 0.25))
        self.similarity_config = similarity_config or {}
        self.knn_config = dict(self.similarity_config.get('knn', {}))
        self.knn_enabled = bool(self.knn_config.get('enable', False))
        self.knn_k = max(1, int(self.knn_config.get('k', 3)))
        self.knn_score_weight = max(0.0, min(0.8, float(self.knn_config.get('score_weight', 0.20))))
        self.text_config = dict(self.similarity_config.get('text_semantic', {}))
        self.text_enabled = bool(self.text_config.get('enable', False))
        self.text_score_weight = max(0.0, min(0.5, float(self.text_config.get('score_weight', 0.08))))
        self.text_prompt_templates = list(self.text_config.get('prompt_templates') or [
            'a photo of a {label}',
            'a close-up photo of a {label}',
        ])
        self.concept_gate_config = dict(self.similarity_config.get('concept_gate', {}))
        self.concept_gate_enabled = bool(self.concept_gate_config.get('enable', True))
        self.concept_gate_min_gap = max(0.0, float(self.concept_gate_config.get('min_top_gap', 0.035)))
        self.concept_weak_score_weight = max(0.0, min(
            self.concept_score_weight,
            float(self.concept_gate_config.get('weak_score_weight', 0.0)),
        ))
        self.deep_adapter = DeepFeatureAdapter(deep_feature_config, cache_dir=deep_cache_dir)
        self.deep_feature_name = self.deep_adapter.feature_name or DEFAULT_DEEP_FEATURE_NAME
        self.prototypes = {}
        self.text_prototypes = {}
        self.concept_prototypes = {}
        self.samples = {}
        self.sample_features = {}
        self.feature_counts = {}
        self.concept_counts = {}

    def _allowed_concepts(self):
        if not self.feature_groups:
            return None
        groups = set(self.feature_groups.keys())
        groups.update(name for name in self.system_feature_names if name in CONCEPTS_BY_GROUP)
        allowed = set()
        for group in groups:
            allowed.update(CONCEPTS_BY_GROUP.get(group, set()))
        return allowed

    def _concepts_from_features(self, feats):
        concepts = concepts_from_features(feats)
        allowed = self._allowed_concepts()
        if allowed is None:
            return concepts
        return {name: value for name, value in concepts.items() if name in allowed}

    def _attach_deep_features(self, feats, path=None, img=None, allow_augmented=False):
        if not self.deep_adapter.is_enabled():
            return feats
        if allow_augmented and not self.deep_adapter.include_augmented:
            return feats
        enriched = dict(feats)
        if path is not None:
            enriched.update(self.deep_adapter.extract_path(path))
        elif img is not None:
            enriched.update(self.deep_adapter.extract_image(img))
        return enriched

    def _extract_primary_features(self, path):
        return self._attach_deep_features(extract_features(path), path=path)

    def _feature_list_for_sample(self, path):
        feats_list = [self._extract_primary_features(path)]
        for img in _augmented_images(path, self.augmentation_config):
            try:
                feats = extract_features_from_image(img)
                feats_list.append(self._attach_deep_features(feats, img=img, allow_augmented=True))
            except Exception:
                pass
        return feats_list

    def fit(self, samples):
        grouped = {}
        concept_grouped = {}
        self.samples = {}
        self.sample_features = {}
        self.feature_counts = {}
        self.concept_counts = {}
        for sample in samples:
            label = sample['label']
            feats_list = self._feature_list_for_sample(sample['path'])
            self.samples.setdefault(label, [])
            self.samples[label].append(sample['path'])
            if feats_list:
                self.sample_features.setdefault(label, []).append({
                    'path': sample['path'],
                    'features': feats_list[0],
                })
            for feats in feats_list:
                all_feature_names = list(self.feature_names) + list(self.system_feature_names)
                grouped.setdefault(label, {name: [] for name in all_feature_names if name in feats})
                for name in all_feature_names:
                    if name in feats:
                        grouped[label].setdefault(name, []).append(feats[name])
                if self.concepts_enabled:
                    concept_grouped.setdefault(label, {})
                    for cname, value in self._concepts_from_features(feats).items():
                        concept_grouped[label].setdefault(cname, []).append(float(value))
        self.prototypes = {}
        self.text_prototypes = {}
        self.concept_prototypes = {}
        for label, fdict in grouped.items():
            self.prototypes[label] = {}
            self.feature_counts[label] = {}
            for name in list(self.feature_names) + list(self.system_feature_names):
                if name in fdict:
                    self.prototypes[label][name] = _mean_vectors(fdict[name])
                    self.feature_counts[label][name] = len(fdict[name])
        for label, cdict in concept_grouped.items():
            self.concept_prototypes[label] = {}
            self.concept_counts[label] = {}
            for name, values in cdict.items():
                self.concept_prototypes[label][name] = float(np.mean(values))
                self.concept_counts[label][name] = len(values)
        self._build_text_prototypes(self.prototypes.keys())
        return self

    def add_confirmed_sample(self, label, image_path):
        feats_list = self._feature_list_for_sample(image_path)
        self.samples.setdefault(label, []).append(image_path)
        if feats_list:
            self.sample_features.setdefault(label, []).append({
                'path': image_path,
                'features': feats_list[0],
            })
        self.prototypes.setdefault(label, {})
        self.feature_counts.setdefault(label, {})
        self.concept_prototypes.setdefault(label, {})
        self.concept_counts.setdefault(label, {})
        all_feature_names = list(self.feature_names) + list(self.system_feature_names)
        new_vectors = {name: [] for name in all_feature_names}
        new_concepts = {}
        for feats in feats_list:
            for name in all_feature_names:
                if name in feats:
                    new_vectors.setdefault(name, []).append(feats[name])
            if self.concepts_enabled:
                for cname, value in self._concepts_from_features(feats).items():
                    new_concepts.setdefault(cname, []).append(float(value))
        for name, vectors in new_vectors.items():
            if not vectors:
                continue
            new_mean = _mean_vectors(vectors)
            old = self.prototypes[label].get(name)
            old_count = int(self.feature_counts[label].get(name, 0))
            new_count = len(vectors)
            if old is None or old_count <= 0:
                self.prototypes[label][name] = new_mean
                self.feature_counts[label][name] = new_count
            else:
                total = old_count + new_count
                self.prototypes[label][name] = (old * old_count + new_mean * new_count) / total
                self.feature_counts[label][name] = total
        for name, values in new_concepts.items():
            if not values:
                continue
            new_mean = float(np.mean(values))
            old = self.concept_prototypes[label].get(name)
            old_count = int(self.concept_counts[label].get(name, 0))
            new_count = len(values)
            if old is None or old_count <= 0:
                self.concept_prototypes[label][name] = new_mean
                self.concept_counts[label][name] = new_count
            else:
                total = old_count + new_count
                self.concept_prototypes[label][name] = (float(old) * old_count + new_mean * new_count) / total
                self.concept_counts[label][name] = total
        self._build_text_prototypes([label])

    def _feature_similarity(self, name, a, b):
        if name == 'color':
            return _hist_similarity(a, b)
        if name == 'contour':
            return _vector_similarity(a, b, scale=5.5)
        if name == 'texture':
            return _vector_similarity(a, b, scale=4.5)
        if name == 'size':
            return _vector_similarity(a, b, scale=2.2)
        if name == 'fruit_color':
            return _vector_similarity(a, b, scale=4.0)
        if name == 'fruit_shape':
            return _vector_similarity(a, b, scale=5.0)
        if name == 'fruit_texture':
            return _vector_similarity(a, b, scale=4.8)
        if name == 'fruit_structure':
            return _vector_similarity(a, b, scale=5.2)
        if name == 'surface_mark':
            return _vector_similarity(a, b, scale=5.0)
        if name == 'fruit_part':
            return _vector_similarity(a, b, scale=5.1)
        if name == 'animal_shape':
            return _vector_similarity(a, b, scale=5.0)
        if name == 'fur_texture':
            return _vector_similarity(a, b, scale=4.9)
        if name == 'animal_face':
            return _vector_similarity(a, b, scale=5.3)
        if name == 'text_mark':
            return _vector_similarity(a, b, scale=5.0)
        if name == 'sign_symbol':
            return _vector_similarity(a, b, scale=5.4)
        if name == self.deep_feature_name or name == DEFAULT_DEEP_FEATURE_NAME:
            return _cosine_similarity(a, b)
        return _vector_similarity(a, b, scale=2.5)

    def _weighted_feature_score(self, feats, proto, weights):
        detail = {}
        score = 0.0
        total_w = 0.0
        for name, w in weights.items():
            if name not in proto or name not in feats:
                continue
            sim = self._feature_similarity(name, feats[name], proto[name])
            detail[name] = sim
            score += float(w) * sim
            total_w += float(w)
        return score / max(total_w, 1e-8), detail

    def _nearest_samples(self, label, feats, weights):
        if not self.knn_enabled:
            return None, []
        neighbors = []
        for item in self.sample_features.get(label, []):
            sample_feats = item.get('features') or {}
            score, detail = self._weighted_feature_score(feats, sample_feats, weights)
            if detail:
                neighbors.append({
                    'path': str(item.get('path')),
                    'score': float(score),
                    'detail': detail,
                    'group_detail': self._group_detail(detail),
                })
        neighbors.sort(key=lambda x: x['score'], reverse=True)
        top = neighbors[:self.knn_k]
        if not top:
            return None, []
        knn_score = float(np.mean([x['score'] for x in top]))
        return knn_score, top

    def _label_prompt_text(self, label):
        return str(label).replace('_', ' ').replace('-', ' ').strip()

    def _build_text_prototypes(self, labels):
        if not self.text_enabled or not self.deep_adapter.is_enabled():
            return
        for label in labels:
            readable = self._label_prompt_text(label)
            prompts = [template.format(label=readable) for template in self.text_prompt_templates]
            vectors = self.deep_adapter.extract_text_vectors(prompts)
            if vectors:
                self.text_prototypes[label] = _l2_normalize(_mean_vectors(vectors))

    def _text_semantic_score(self, label, feats):
        if not self.text_enabled:
            return None
        text_proto = self.text_prototypes.get(label)
        image_vec = feats.get(self.deep_feature_name)
        if image_vec is None:
            image_vec = feats.get(DEFAULT_DEEP_FEATURE_NAME)
        if text_proto is None or image_vec is None:
            return None
        return _cosine_similarity(image_vec, text_proto)

    def _group_detail(self, detail):
        out = {}
        for group, names in self.feature_groups.items():
            vals = [float(detail[name]) for name in names if name in detail]
            if vals:
                out[group] = float(np.mean(vals))
        return out

    def _concept_weight_for_rows(self, rows, row):
        if not self.concepts_enabled or row.get('concept_score') is None:
            return 0.0, 'missing'
        base_weight = max(0.0, min(0.8, self.concept_score_weight))
        if not self.concept_gate_enabled:
            return base_weight, 'ungated'
        scored = [item for item in rows if item.get('concept_score') is not None]
        if len(scored) < 2:
            return self.concept_weak_score_weight, 'insufficient_competition'
        ranked = sorted(scored, key=lambda item: float(item['concept_score']), reverse=True)
        concept_gap = float(ranked[0]['concept_score']) - float(ranked[1]['concept_score'])
        row['concept_top_gap'] = concept_gap
        if concept_gap >= self.concept_gate_min_gap:
            return base_weight, 'discriminative'
        return self.concept_weak_score_weight, 'weak_gap'

    def predict(self, image_path, weights):
        feats = self._extract_primary_features(image_path)
        sample_concepts = self._concepts_from_features(feats) if self.concepts_enabled else {}
        rows = []
        for label, proto in self.prototypes.items():
            prototype_score, detail = self._weighted_feature_score(feats, proto, weights)
            system_detail = {}
            for name in self.system_feature_names:
                if name in proto and name in feats:
                    system_detail[name] = self._feature_similarity(name, feats[name], proto[name])
            feature_score = prototype_score
            knn_score, nearest = self._nearest_samples(label, feats, weights)
            if knn_score is not None:
                kw = self.knn_score_weight
                feature_score = (1.0 - kw) * prototype_score + kw * knn_score
            final = feature_score
            text_score = self._text_semantic_score(label, feats)
            if text_score is not None:
                detail['text_semantic'] = text_score
                tw = self.text_score_weight
                final = (1.0 - tw) * final + tw * text_score
            concept_score = None
            concept_proto = self.concept_prototypes.get(label, {})
            if self.concepts_enabled and concept_proto:
                concept_score = concept_similarity(sample_concepts, concept_proto)
                detail['concept'] = concept_score
            rows.append({
                'label': label,
                'score': final,
                'feature_score': feature_score,
                'prototype_score': prototype_score,
                'knn_score': knn_score,
                'text_semantic_score': text_score,
                'concept_score': concept_score,
                'detail': detail,
                'group_detail': self._group_detail(detail),
                'system_detail': system_detail,
                'concepts': sample_concepts,
                'class_concepts': concept_proto,
                'nearest_samples': nearest,
            })
        for row in rows:
            concept_score = row.get('concept_score')
            cw, gate_reason = self._concept_weight_for_rows(rows, row)
            row['concept_score_weight_used'] = cw
            row['concept_gate_reason'] = gate_reason
            if concept_score is not None and cw > 0.0:
                row['score'] = (1.0 - cw) * float(row['score']) + cw * float(concept_score)
            if row.get('concept_top_gap') is None:
                row['concept_top_gap'] = None
        results = rows
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def export(self):
        return {
            'feature_prototypes': {
                label: {k: v.tolist() for k, v in feats.items()}
                for label, feats in self.prototypes.items()
            },
            'text_prototypes': {
                label: vec.tolist()
                for label, vec in self.text_prototypes.items()
            },
            'concept_prototypes': self.concept_prototypes,
            'concept_config': {
                'enable': self.concepts_enabled,
                'score_weight': self.concept_score_weight,
            },
            'similarity_config': {
                'knn': {
                    'enable': self.knn_enabled,
                    'k': self.knn_k,
                    'score_weight': self.knn_score_weight,
                },
                'text_semantic': {
                    'enable': self.text_enabled,
                    'score_weight': self.text_score_weight,
                    'prompt_templates': self.text_prompt_templates,
                },
                'concept_gate': {
                    'enable': self.concept_gate_enabled,
                    'min_top_gap': self.concept_gate_min_gap,
                    'weak_score_weight': self.concept_weak_score_weight,
                },
            },
            'deep_features': self.deep_adapter.metadata(),
            'feature_groups': self.feature_groups,
            'system_features': self.system_feature_names,
            'sample_index': {
                label: [str(path) for path in paths]
                for label, paths in self.samples.items()
            },
        }

