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


def _robust_mean_vectors(vectors, trim_fraction=0.12, min_samples=12):
    if not vectors:
        return None, {'count': 0, 'used_count': 0, 'trimmed_count': 0}
    rows = [np.asarray(vec, dtype=np.float32).reshape(-1) for vec in vectors]
    if len(rows) < int(min_samples):
        return np.mean(np.stack(rows), axis=0), {
            'count': len(rows),
            'used_count': len(rows),
            'trimmed_count': 0,
        }
    matrix = np.stack(rows)
    center = _l2_normalize(np.mean(matrix, axis=0))
    sims = np.asarray([_cosine_similarity(row, center) for row in matrix], dtype=np.float32)
    keep_count = int(round(len(rows) * (1.0 - float(trim_fraction))))
    keep_count = min(len(rows), max(1, keep_count))
    keep_indexes = np.argsort(sims)[-keep_count:]
    return np.mean(matrix[keep_indexes], axis=0), {
        'count': len(rows),
        'used_count': int(keep_count),
        'trimmed_count': int(len(rows) - keep_count),
        'min_kept_similarity': float(np.min(sims[keep_indexes])) if len(keep_indexes) else None,
        'min_similarity': float(np.min(sims)) if len(sims) else None,
    }


def _l2_normalize(vec):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm


def _normalized_matrix(vectors):
    rows = []
    for vec in vectors or []:
        arr = _l2_normalize(vec)
        if arr.size:
            rows.append(arr)
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(rows).astype(np.float32)


def _subprototype_centers(vectors, max_centers=3, min_samples_per_center=8, max_iter=30):
    x = _normalized_matrix(vectors)
    n = len(x)
    if n <= 0:
        return []
    k = min(int(max_centers), max(1, n // max(1, int(min_samples_per_center))))
    if k <= 1:
        return []
    centers = [x[0]]
    min_dist = np.sum((x - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(min_dist))
        centers.append(x[idx])
        dist = np.sum((x - x[idx]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
    centers = np.stack(centers).astype(np.float32)
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        sims = x @ centers.T
        new_labels = np.argmax(sims, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            members = x[labels == cluster_id]
            if len(members):
                centers[cluster_id] = _l2_normalize(np.mean(members, axis=0))
    return [centers[i].copy() for i in range(k)]


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
        self.subprototype_config = dict(self.similarity_config.get('sub_prototypes', {}))
        self.subprototype_enabled = bool(self.subprototype_config.get('enable', True))
        self.subprototype_max_centers = max(1, int(self.subprototype_config.get('max_centers', 3)))
        self.subprototype_min_samples = max(1, int(self.subprototype_config.get('min_samples_per_center', 8)))
        self.subprototype_score_weight = max(0.0, min(0.5, float(self.subprototype_config.get('score_weight', 0.06))))
        self.subprototype_mode = str(self.subprototype_config.get('mode', 'conservative')).strip().lower()
        self.subprototype_min_gain = max(0.0, float(self.subprototype_config.get('min_gain_over_prototype', 0.015)))
        self.subprototype_min_top_gap = max(0.0, float(self.subprototype_config.get('min_top_gap', 0.0)))
        self.subprototype_allow_rank_flip = bool(self.subprototype_config.get('allow_rank_flip', True))
        self.subprototype_max_base_margin_for_flip = max(0.0, float(self.subprototype_config.get('max_base_margin_for_flip', 0.010)))
        self.subprototype_prototype_veto_margin = max(0.0, float(self.subprototype_config.get('rank_flip_prototype_veto_margin', 0.003)))
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
        self.pairwise_config = dict(self.similarity_config.get('pairwise_rerank', {}))
        self.pairwise_enabled = bool(self.pairwise_config.get('enable', True))
        self.pairwise_local_k = max(1, int(self.pairwise_config.get('local_k', 5)))
        self.pairwise_score_weight = max(0.0, min(0.5, float(self.pairwise_config.get('score_weight', 0.25))))
        self.pairwise_max_margin = max(0.0, float(self.pairwise_config.get('max_score_margin', 0.018)))
        self.pairwise_min_pair_similarity = max(0.0, min(1.0, float(self.pairwise_config.get('min_pair_similarity', 0.90))))
        self.pairwise_min_local_gap = max(0.0, float(self.pairwise_config.get('min_local_gap', 0.008)))
        self.crop_rerank_config = dict(
            self.similarity_config.get('crop_rerank')
            or self.similarity_config.get('confusion_rerank')
            or {}
        )
        self.crop_rerank_enabled = bool(self.crop_rerank_config.get('enable', True))
        self.crop_rerank_candidate_count = max(2, int(self.crop_rerank_config.get('max_candidate_classes', 3)))
        self.crop_rerank_local_k = max(1, int(self.crop_rerank_config.get('local_k', 5)))
        self.crop_rerank_score_weight = max(0.0, min(0.5, float(self.crop_rerank_config.get('score_weight', 0.18))))
        self.crop_rerank_max_margin = max(0.0, float(self.crop_rerank_config.get('max_score_margin', 0.018)))
        self.crop_rerank_min_pair_similarity = max(0.0, min(
            1.0,
            float(self.crop_rerank_config.get('min_pair_similarity', 0.94)),
        ))
        self.crop_rerank_min_local_gap = max(0.0, float(self.crop_rerank_config.get('min_local_gap', 0.006)))
        self.crop_rerank_use_full_crop = bool(self.crop_rerank_config.get('use_full_crop', False))
        self.crop_rerank_trigger_mode = str(
            self.crop_rerank_config.get('trigger_mode', 'margin_and_pair_similarity')
        ).strip().lower()
        self.late_fusion_config = dict(self.similarity_config.get('late_fusion', {}))
        self.late_fusion_enabled = bool(self.late_fusion_config.get('enable', False))
        self.late_fusion_candidate_count = max(2, int(self.late_fusion_config.get('max_candidate_classes', 3)))
        self.late_fusion_weights = {
            str(name): max(0.0, float(weight))
            for name, weight in dict(self.late_fusion_config.get('weights') or {}).items()
        }
        if not self.late_fusion_weights:
            self.late_fusion_weights = {'score': 1.0}
        self.robust_config = dict(self.similarity_config.get('robust_prototype', {}))
        self.robust_enabled = bool(self.robust_config.get('enable', True))
        self.robust_deep_only = bool(self.robust_config.get('deep_only', True))
        self.robust_min_samples = max(2, int(self.robust_config.get('min_samples', 24)))
        self.robust_trim_fraction = max(0.0, min(0.4, float(self.robust_config.get('trim_fraction', 0.08))))
        self.robust_report_margin = max(0.0, float(self.robust_config.get('report_margin', 0.015)))
        self.robust_top_outliers = max(1, int(self.robust_config.get('top_outliers_per_class', 5)))
        self.deep_adapter = DeepFeatureAdapter(deep_feature_config, cache_dir=deep_cache_dir)
        self.deep_feature_name = self.deep_adapter.feature_name or DEFAULT_DEEP_FEATURE_NAME
        self.prototypes = {}
        self.sub_prototypes = {}
        self.text_prototypes = {}
        self.pairwise_similarities = {}
        self.concept_prototypes = {}
        self.samples = {}
        self.sample_features = {}
        self.feature_counts = {}
        self.concept_counts = {}
        self.prototype_stats = {}
        self.training_quality_report = {}

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

    def _extract_crop_embeddings_for_path(self, path):
        if not (self.crop_rerank_enabled and self.deep_adapter.is_enabled()):
            return []
        return self.deep_adapter.extract_multi_crop_path(path)

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
        self.prototype_stats = {}
        self.training_quality_report = {}
        for sample in samples:
            label = sample['label']
            feats_list = self._feature_list_for_sample(sample['path'])
            self.samples.setdefault(label, [])
            self.samples[label].append(sample['path'])
            if feats_list:
                self.sample_features.setdefault(label, []).append({
                    'path': sample['path'],
                    'features': feats_list[0],
                    'crop_embeddings': self._extract_crop_embeddings_for_path(sample['path']),
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
        self.sub_prototypes = {}
        self.text_prototypes = {}
        self.pairwise_similarities = {}
        self.concept_prototypes = {}
        for label, fdict in grouped.items():
            self.prototypes[label] = {}
            self.feature_counts[label] = {}
            for name in list(self.feature_names) + list(self.system_feature_names):
                if name in fdict:
                    use_robust = (
                        self.robust_enabled
                        and (not self.robust_deep_only or name in {self.deep_feature_name, DEFAULT_DEEP_FEATURE_NAME})
                    )
                    if use_robust:
                        mean, stats = _robust_mean_vectors(
                            fdict[name],
                            trim_fraction=self.robust_trim_fraction,
                            min_samples=self.robust_min_samples,
                        )
                        self.prototype_stats.setdefault(label, {})[name] = stats
                        self.prototypes[label][name] = mean
                    else:
                        self.prototypes[label][name] = _mean_vectors(fdict[name])
                    self.feature_counts[label][name] = len(fdict[name])
        for label, cdict in concept_grouped.items():
            self.concept_prototypes[label] = {}
            self.concept_counts[label] = {}
            for name, values in cdict.items():
                self.concept_prototypes[label][name] = float(np.mean(values))
                self.concept_counts[label][name] = len(values)
        self._build_sub_prototypes(self.prototypes.keys())
        self._build_text_prototypes(self.prototypes.keys())
        self._build_pairwise_similarities()
        self._build_training_quality_report()
        return self

    def add_confirmed_sample(self, label, image_path):
        feats_list = self._feature_list_for_sample(image_path)
        self.samples.setdefault(label, []).append(image_path)
        if feats_list:
            self.sample_features.setdefault(label, []).append({
                'path': image_path,
                'features': feats_list[0],
                'crop_embeddings': self._extract_crop_embeddings_for_path(image_path),
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
        self._build_sub_prototypes([label])
        self._build_text_prototypes([label])
        self._refresh_label_robust_deep_prototype(label)
        self._build_pairwise_similarities()
        self._build_training_quality_report()

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
        if name == 'car_shape':
            return _vector_similarity(a, b, scale=5.1)
        if name == 'car_part':
            return _vector_similarity(a, b, scale=5.2)
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

    def _pair_key(self, a, b):
        return tuple(sorted((str(a), str(b))))

    def _build_pairwise_similarities(self):
        self.pairwise_similarities = {}
        if not (self.pairwise_enabled or self.crop_rerank_enabled):
            return
        labels = sorted(self.prototypes.keys())
        for i, left in enumerate(labels):
            left_proto = self.prototypes.get(left, {})
            left_vec = left_proto.get(self.deep_feature_name)
            if left_vec is None:
                left_vec = left_proto.get(DEFAULT_DEEP_FEATURE_NAME)
            if left_vec is None:
                continue
            for right in labels[i + 1:]:
                right_proto = self.prototypes.get(right, {})
                right_vec = right_proto.get(self.deep_feature_name)
                if right_vec is None:
                    right_vec = right_proto.get(DEFAULT_DEEP_FEATURE_NAME)
                if right_vec is None:
                    continue
                self.pairwise_similarities[self._pair_key(left, right)] = _cosine_similarity(left_vec, right_vec)

    def _refresh_label_robust_deep_prototype(self, label):
        if not self.robust_enabled:
            return
        vectors = []
        for item in self.sample_features.get(label, []):
            feats = item.get('features') or {}
            vec = feats.get(self.deep_feature_name)
            if vec is None:
                vec = feats.get(DEFAULT_DEEP_FEATURE_NAME)
            if vec is not None:
                vectors.append(vec)
        if not vectors:
            return
        mean, stats = _robust_mean_vectors(
            vectors,
            trim_fraction=self.robust_trim_fraction,
            min_samples=self.robust_min_samples,
        )
        self.prototypes.setdefault(label, {})[self.deep_feature_name] = mean
        self.prototype_stats.setdefault(label, {})[self.deep_feature_name] = stats

    def _build_training_quality_report(self):
        self.training_quality_report = {}
        labels = sorted(self.prototypes.keys())
        for label in labels:
            rows = []
            own_proto = self.prototypes.get(label, {})
            own_vec = own_proto.get(self.deep_feature_name)
            if own_vec is None:
                own_vec = own_proto.get(DEFAULT_DEEP_FEATURE_NAME)
            if own_vec is None:
                continue
            for item in self.sample_features.get(label, []):
                feats = item.get('features') or {}
                vec = feats.get(self.deep_feature_name)
                if vec is None:
                    vec = feats.get(DEFAULT_DEEP_FEATURE_NAME)
                if vec is None:
                    continue
                own_score = _cosine_similarity(vec, own_vec)
                competitor_label = None
                competitor_score = None
                for other in labels:
                    if other == label:
                        continue
                    other_proto = self.prototypes.get(other, {})
                    other_vec = other_proto.get(self.deep_feature_name)
                    if other_vec is None:
                        other_vec = other_proto.get(DEFAULT_DEEP_FEATURE_NAME)
                    if other_vec is None:
                        continue
                    score = _cosine_similarity(vec, other_vec)
                    if competitor_score is None or score > competitor_score:
                        competitor_label = other
                        competitor_score = score
                margin = None if competitor_score is None else own_score - competitor_score
                rows.append({
                    'path': str(item.get('path')),
                    'own_score': float(own_score),
                    'nearest_competitor': competitor_label,
                    'nearest_competitor_score': None if competitor_score is None else float(competitor_score),
                    'margin': margin,
                })
            rows.sort(key=lambda row: row['own_score'])
            risk_rows = [
                row for row in rows
                if row['margin'] is not None and row['margin'] < self.robust_report_margin
            ]
            self.training_quality_report[label] = {
                'sample_count': len(rows),
                'prototype_stats': self.prototype_stats.get(label, {}).get(self.deep_feature_name, {}),
                'outliers': rows[:self.robust_top_outliers],
                'confusion_risk_samples': sorted(
                    risk_rows,
                    key=lambda row: row['margin'] if row['margin'] is not None else 0.0,
                )[:self.robust_top_outliers],
            }

    def _pairwise_local_score(self, label, feats, weights):
        neighbors = []
        for item in self.sample_features.get(label, []):
            sample_feats = item.get('features') or {}
            score, detail = self._weighted_feature_score(feats, sample_feats, weights)
            if detail:
                neighbors.append(float(score))
        if not neighbors:
            return None
        neighbors.sort(reverse=True)
        top = neighbors[:self.pairwise_local_k]
        return float(0.65 * top[0] + 0.35 * np.mean(top))

    def _apply_pairwise_rerank(self, rows, feats, weights):
        for row in rows:
            row['pairwise_score'] = None
            row['pairwise_score_weight_used'] = 0.0
            row['pairwise_gate_reason'] = 'not_candidate'
            row['pairwise_pair_similarity'] = None
            row['pairwise_local_gap'] = None
        if not self.pairwise_enabled or len(rows) < 2:
            return
        ranked = sorted(rows, key=lambda item: float(item['score']), reverse=True)
        top = ranked[0]
        second = ranked[1]
        score_margin = float(top['score']) - float(second['score'])
        pair_key = self._pair_key(top['label'], second['label'])
        pair_similarity = self.pairwise_similarities.get(pair_key)
        for row in (top, second):
            row['pairwise_pair_similarity'] = pair_similarity
        if score_margin > self.pairwise_max_margin:
            for row in (top, second):
                row['pairwise_gate_reason'] = 'score_margin_too_large'
            return
        if pair_similarity is not None and pair_similarity < self.pairwise_min_pair_similarity:
            for row in (top, second):
                row['pairwise_gate_reason'] = 'pair_not_similar'
            return
        top_local = self._pairwise_local_score(top['label'], feats, weights)
        second_local = self._pairwise_local_score(second['label'], feats, weights)
        if top_local is None or second_local is None:
            for row in (top, second):
                row['pairwise_gate_reason'] = 'missing_local_evidence'
            return
        top['pairwise_score'] = top_local
        second['pairwise_score'] = second_local
        local_gap = abs(top_local - second_local)
        for row in (top, second):
            row['pairwise_local_gap'] = local_gap
        if local_gap < self.pairwise_min_local_gap:
            for row in (top, second):
                row['pairwise_gate_reason'] = 'weak_local_gap'
            return
        for row in (top, second):
            row['pairwise_score_weight_used'] = self.pairwise_score_weight
            row['pairwise_gate_reason'] = 'local_evidence'
            row['score'] = (1.0 - self.pairwise_score_weight) * float(row['score']) + self.pairwise_score_weight * float(row['pairwise_score'])

    def _crop_vectors_for_scoring(self, crop_embeddings):
        vectors = []
        for item in crop_embeddings or []:
            if not self.crop_rerank_use_full_crop and item.get('crop_id') == 'full':
                continue
            vec = item.get('vector')
            if vec is not None:
                vectors.append(vec)
        return vectors

    def _crop_local_score(self, label, query_crops):
        query_vectors = self._crop_vectors_for_scoring(query_crops)
        if not query_vectors:
            return None
        candidate_scores = []
        for item in self.sample_features.get(label, []):
            sample_vectors = self._crop_vectors_for_scoring(item.get('crop_embeddings') or [])
            for query_vec in query_vectors:
                for sample_vec in sample_vectors:
                    candidate_scores.append(_cosine_similarity(query_vec, sample_vec))
        if not candidate_scores:
            return None
        candidate_scores.sort(reverse=True)
        top = candidate_scores[:self.crop_rerank_local_k]
        return float(0.70 * top[0] + 0.30 * np.mean(top))

    def _apply_crop_rerank(self, rows, image_path):
        for row in rows:
            row['crop_rerank_score'] = None
            row['crop_rerank_score_weight_used'] = 0.0
            row['crop_rerank_gate_reason'] = 'not_candidate'
            row['crop_rerank_pair_similarity'] = None
            row['crop_rerank_local_gap'] = None
            row['crop_rerank_crop_count'] = 0
        if not self.crop_rerank_enabled or len(rows) < 2:
            return
        ranked = sorted(rows, key=lambda item: float(item['score']), reverse=True)
        top = ranked[0]
        second = ranked[1]
        score_margin = float(top['score']) - float(second['score'])
        pair_similarity = self.pairwise_similarities.get(self._pair_key(top['label'], second['label']))
        candidates = ranked[:min(len(ranked), self.crop_rerank_candidate_count)]
        for row in candidates:
            row['crop_rerank_pair_similarity'] = pair_similarity

        ambiguous_by_margin = score_margin <= self.crop_rerank_max_margin
        ambiguous_by_pair = pair_similarity is not None and pair_similarity >= self.crop_rerank_min_pair_similarity
        trigger_mode = self.crop_rerank_trigger_mode
        if trigger_mode in {'margin_or_pair_similarity', 'or'}:
            should_trigger = ambiguous_by_margin or ambiguous_by_pair
        elif trigger_mode in {'margin_only', 'margin'}:
            should_trigger = ambiguous_by_margin
        elif trigger_mode in {'pair_similarity_only', 'pair_only', 'pair'}:
            should_trigger = ambiguous_by_pair
        else:
            should_trigger = ambiguous_by_margin and ambiguous_by_pair
        if not should_trigger:
            reason = 'not_ambiguous'
            if not ambiguous_by_margin:
                reason = 'score_margin_too_large'
            elif not ambiguous_by_pair:
                reason = 'pair_similarity_too_low'
            for row in candidates[:2]:
                row['crop_rerank_gate_reason'] = reason
            return

        query_crops = self._extract_crop_embeddings_for_path(image_path)
        query_vectors = self._crop_vectors_for_scoring(query_crops)
        if not query_vectors:
            for row in candidates:
                row['crop_rerank_gate_reason'] = 'missing_query_crops'
            return

        scored = []
        for row in candidates:
            crop_score = self._crop_local_score(row['label'], query_crops)
            row['crop_rerank_score'] = crop_score
            row['crop_rerank_crop_count'] = len(query_vectors)
            if crop_score is None:
                row['crop_rerank_gate_reason'] = 'missing_label_crop_evidence'
            else:
                scored.append(row)
        if len(scored) < 2:
            for row in scored:
                row['crop_rerank_gate_reason'] = 'insufficient_competition'
            return

        crop_ranked = sorted(scored, key=lambda item: float(item['crop_rerank_score']), reverse=True)
        local_gap = float(crop_ranked[0]['crop_rerank_score']) - float(crop_ranked[1]['crop_rerank_score'])
        for row in scored:
            row['crop_rerank_local_gap'] = local_gap
        if local_gap < self.crop_rerank_min_local_gap:
            for row in scored:
                row['crop_rerank_gate_reason'] = 'weak_crop_gap'
            return

        for row in scored:
            row['crop_rerank_score_weight_used'] = self.crop_rerank_score_weight
            row['crop_rerank_gate_reason'] = 'crop_local_evidence'
            row['score'] = (
                (1.0 - self.crop_rerank_score_weight) * float(row['score'])
                + self.crop_rerank_score_weight * float(row['crop_rerank_score'])
            )

    def _apply_late_fusion_rerank(self, rows):
        for row in rows:
            row['late_fusion_score'] = None
            row['late_fusion_gate_reason'] = 'disabled'
            row['late_fusion_evidence'] = {}
        if not self.late_fusion_enabled or len(rows) < 2:
            return

        ranked = sorted(rows, key=lambda item: float(item['score']), reverse=True)
        candidates = ranked[:min(len(ranked), self.late_fusion_candidate_count)]
        if len(candidates) < 2:
            return

        for row in candidates:
            weighted_sum = 0.0
            total_weight = 0.0
            evidence = {}
            for key, weight in self.late_fusion_weights.items():
                if weight <= 0.0:
                    continue
                value = row.get(key)
                if value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                weighted_sum += weight * numeric
                total_weight += weight
                evidence[key] = {
                    'weight': float(weight),
                    'score': float(numeric),
                }
            if total_weight <= 0.0:
                row['late_fusion_gate_reason'] = 'missing_weighted_sources'
                continue
            row['late_fusion_score'] = weighted_sum / total_weight
            row['late_fusion_gate_reason'] = 'applied'
            row['late_fusion_evidence'] = evidence

        scored = [row for row in candidates if row.get('late_fusion_score') is not None]
        if len(scored) < 2:
            return
        original_slots = sorted([float(row['score']) for row in candidates], reverse=True)
        fusion_ranked = sorted(scored, key=lambda item: float(item['late_fusion_score']), reverse=True)
        for idx, row in enumerate(fusion_ranked):
            row['score'] = original_slots[min(idx, len(original_slots) - 1)]

    def _label_prompt_text(self, label):
        return str(label).replace('_', ' ').replace('-', ' ').strip()

    def _build_sub_prototypes(self, labels):
        if not self.subprototype_enabled:
            return
        for label in labels:
            vectors = []
            for item in self.sample_features.get(label, []):
                feats = item.get('features') or {}
                vec = feats.get(self.deep_feature_name)
                if vec is None:
                    vec = feats.get(DEFAULT_DEEP_FEATURE_NAME)
                if vec is not None:
                    vectors.append(vec)
            centers = _subprototype_centers(
                vectors,
                max_centers=self.subprototype_max_centers,
                min_samples_per_center=self.subprototype_min_samples,
            )
            if centers:
                self.sub_prototypes[label] = centers
            else:
                self.sub_prototypes.pop(label, None)

    def _subprototype_score(self, label, feats):
        if not self.subprototype_enabled:
            return None
        centers = self.sub_prototypes.get(label) or []
        image_vec = feats.get(self.deep_feature_name)
        if image_vec is None:
            image_vec = feats.get(DEFAULT_DEEP_FEATURE_NAME)
        if image_vec is None or not centers:
            return None
        return max(_cosine_similarity(image_vec, center) for center in centers)

    def _subprototype_weight_for_row(self, rows, row, base_top, sub_top, sub_gap, base_margin):
        if not self.subprototype_enabled or row.get('subprototype_score') is None:
            return 0.0, 'missing'
        base_weight = self.subprototype_score_weight
        gain = float(row['subprototype_score']) - float(row['prototype_score'])
        row['subprototype_gain_over_prototype'] = gain
        row['subprototype_top_gap'] = sub_gap
        if self.subprototype_mode not in {'conservative', 'gated'}:
            return base_weight, 'ungated'
        if gain < self.subprototype_min_gain:
            return 0.0, 'low_gain'
        if row is base_top:
            return base_weight, 'base_top_support'
        if not self.subprototype_allow_rank_flip:
            return 0.0, 'rank_flip_disabled'
        if row is not sub_top:
            return 0.0, 'not_subprototype_top'
        if sub_gap < self.subprototype_min_top_gap:
            return 0.0, 'weak_subprototype_gap'
        if base_margin > self.subprototype_max_base_margin_for_flip:
            return 0.0, 'base_margin_too_large'
        prototype_advantage = float(base_top['prototype_score']) - float(row['prototype_score'])
        if prototype_advantage > self.subprototype_prototype_veto_margin:
            return 0.0, 'prototype_veto'
        return base_weight, 'rank_flip_allowed'

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

    def pair_discriminative_summary(self, label_a, label_b, top_n=5):
        proto_a = self.prototypes.get(label_a, {})
        proto_b = self.prototypes.get(label_b, {})
        group_rows = []
        for group, names in self.feature_groups.items():
            sims = []
            for name in names:
                if name in proto_a and name in proto_b:
                    sims.append(self._feature_similarity(name, proto_a[name], proto_b[name]))
            if sims:
                similarity = float(np.mean(sims))
                group_rows.append({
                    'group': group,
                    'prototype_similarity': similarity,
                    'discriminative_gap': max(0.0, 1.0 - similarity),
                })
        group_rows.sort(key=lambda item: item['discriminative_gap'], reverse=True)

        concept_rows = []
        concepts_a = self.concept_prototypes.get(label_a, {})
        concepts_b = self.concept_prototypes.get(label_b, {})
        for name in sorted(set(concepts_a) | set(concepts_b)):
            av = float(concepts_a.get(name, 0.0))
            bv = float(concepts_b.get(name, 0.0))
            gap = abs(av - bv)
            if gap <= 0.0:
                continue
            concept_rows.append({
                'concept': name,
                'a_score': av,
                'b_score': bv,
                'stronger_label': label_a if av >= bv else label_b,
                'gap': gap,
            })
        concept_rows.sort(key=lambda item: item['gap'], reverse=True)

        return {
            'labels': [label_a, label_b],
            'top_group_differences': group_rows[:top_n],
            'weak_group_differences': sorted(group_rows, key=lambda item: item['discriminative_gap'])[:top_n],
            'top_concept_differences': concept_rows[:top_n],
        }

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
            subprototype_score = self._subprototype_score(label, feats)
            if subprototype_score is not None:
                detail['subprototype'] = subprototype_score
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
                'base_score': final,
                'feature_score': feature_score,
                'prototype_score': prototype_score,
                'subprototype_score': subprototype_score,
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
        base_ranked = sorted(rows, key=lambda item: float(item['base_score']), reverse=True)
        base_top = base_ranked[0] if base_ranked else None
        base_margin = (
            float(base_ranked[0]['base_score']) - float(base_ranked[1]['base_score'])
            if len(base_ranked) > 1 else 0.0
        )
        sub_ranked = sorted(
            [row for row in rows if row.get('subprototype_score') is not None],
            key=lambda item: float(item['subprototype_score']),
            reverse=True,
        )
        sub_top = sub_ranked[0] if sub_ranked else None
        sub_gap = (
            float(sub_ranked[0]['subprototype_score']) - float(sub_ranked[1]['subprototype_score'])
            if len(sub_ranked) > 1 else 0.0
        )
        for row in rows:
            sw, sub_reason = self._subprototype_weight_for_row(rows, row, base_top, sub_top, sub_gap, base_margin)
            row['subprototype_score_weight_used'] = sw
            row['subprototype_gate_reason'] = sub_reason
            if row.get('subprototype_gain_over_prototype') is None:
                row['subprototype_gain_over_prototype'] = None
            if row.get('subprototype_top_gap') is None:
                row['subprototype_top_gap'] = sub_gap if sub_top is not None else None
            if row.get('subprototype_score') is not None and sw > 0.0:
                row['score'] = (1.0 - sw) * float(row['score']) + sw * float(row['subprototype_score'])
        for row in rows:
            concept_score = row.get('concept_score')
            cw, gate_reason = self._concept_weight_for_rows(rows, row)
            row['concept_score_weight_used'] = cw
            row['concept_gate_reason'] = gate_reason
            if concept_score is not None and cw > 0.0:
                row['score'] = (1.0 - cw) * float(row['score']) + cw * float(concept_score)
            if row.get('concept_top_gap') is None:
                row['concept_top_gap'] = None
        self._apply_pairwise_rerank(rows, feats, weights)
        self._apply_crop_rerank(rows, image_path)
        self._apply_late_fusion_rerank(rows)
        results = rows
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def export(self):
        return {
            'feature_prototypes': {
                label: {k: v.tolist() for k, v in feats.items()}
                for label, feats in self.prototypes.items()
            },
            'sub_prototypes': {
                label: [vec.tolist() for vec in vectors]
                for label, vectors in self.sub_prototypes.items()
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
                'sub_prototypes': {
                    'enable': self.subprototype_enabled,
                    'max_centers': self.subprototype_max_centers,
                    'min_samples_per_center': self.subprototype_min_samples,
                    'score_weight': self.subprototype_score_weight,
                    'mode': self.subprototype_mode,
                    'min_gain_over_prototype': self.subprototype_min_gain,
                    'min_top_gap': self.subprototype_min_top_gap,
                    'allow_rank_flip': self.subprototype_allow_rank_flip,
                    'max_base_margin_for_flip': self.subprototype_max_base_margin_for_flip,
                    'rank_flip_prototype_veto_margin': self.subprototype_prototype_veto_margin,
                },
                'text_semantic': {
                    'enable': self.text_enabled,
                    'score_weight': self.text_score_weight,
                    'prompt_templates': self.text_prompt_templates,
                },
                'pairwise_rerank': {
                    'enable': self.pairwise_enabled,
                    'local_k': self.pairwise_local_k,
                    'score_weight': self.pairwise_score_weight,
                    'max_score_margin': self.pairwise_max_margin,
                    'min_pair_similarity': self.pairwise_min_pair_similarity,
                    'min_local_gap': self.pairwise_min_local_gap,
                },
                'crop_rerank': {
                    'enable': self.crop_rerank_enabled,
                    'max_candidate_classes': self.crop_rerank_candidate_count,
                    'local_k': self.crop_rerank_local_k,
                    'score_weight': self.crop_rerank_score_weight,
                    'max_score_margin': self.crop_rerank_max_margin,
                    'min_pair_similarity': self.crop_rerank_min_pair_similarity,
                    'min_local_gap': self.crop_rerank_min_local_gap,
                    'use_full_crop': self.crop_rerank_use_full_crop,
                    'trigger_mode': self.crop_rerank_trigger_mode,
                },
                'late_fusion': {
                    'enable': self.late_fusion_enabled,
                    'max_candidate_classes': self.late_fusion_candidate_count,
                    'weights': self.late_fusion_weights,
                },
                'robust_prototype': {
                    'enable': self.robust_enabled,
                    'deep_only': self.robust_deep_only,
                    'min_samples': self.robust_min_samples,
                    'trim_fraction': self.robust_trim_fraction,
                    'report_margin': self.robust_report_margin,
                    'top_outliers_per_class': self.robust_top_outliers,
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
            'prototype_stats': self.prototype_stats,
            'training_quality_report': self.training_quality_report,
            'sample_index': {
                label: [str(path) for path in paths]
                for label, paths in self.samples.items()
            },
        }

