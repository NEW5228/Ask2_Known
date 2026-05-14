import cv2
import numpy as np
from ask2know.features.basic_features import extract_features, extract_features_from_image
from ask2know.concepts.basic_concepts import concepts_from_features, concept_similarity


def _mean_vectors(vectors):
    if not vectors:
        return None
    return np.mean(np.stack(vectors), axis=0)


def _hist_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    hist_len = max(0, len(a) - 14)  # HSV hist + 6 stats + 8 coarse color bins
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
    def __init__(self, feature_names, augmentation_config=None, concept_config=None, system_feature_names=None, feature_groups=None):
        self.feature_names = feature_names
        self.system_feature_names = list(system_feature_names or [])
        self.feature_groups = feature_groups or {}
        self.augmentation_config = augmentation_config or {'enable': False}
        self.concept_config = concept_config or {'enable': True, 'score_weight': 0.25}
        self.concepts_enabled = self.concept_config.get('enable', True)
        self.concept_score_weight = float(self.concept_config.get('score_weight', 0.25))
        self.prototypes = {}
        self.concept_prototypes = {}
        self.samples = {}
        self.feature_counts = {}
        self.concept_counts = {}

    def _feature_list_for_sample(self, path):
        feats_list = [extract_features(path)]
        for img in _augmented_images(path, self.augmentation_config):
            try:
                feats_list.append(extract_features_from_image(img))
            except Exception:
                pass
        return feats_list

    def fit(self, samples):
        grouped = {}
        concept_grouped = {}
        self.samples = {}
        self.feature_counts = {}
        self.concept_counts = {}
        for sample in samples:
            label = sample['label']
            feats_list = self._feature_list_for_sample(sample['path'])
            self.samples.setdefault(label, [])
            self.samples[label].append(sample['path'])
            for feats in feats_list:
                all_feature_names = list(self.feature_names) + list(self.system_feature_names)
                grouped.setdefault(label, {name: [] for name in all_feature_names if name in feats})
                for name in all_feature_names:
                    if name in feats:
                        grouped[label].setdefault(name, []).append(feats[name])
                if self.concepts_enabled:
                    concept_grouped.setdefault(label, {})
                    for cname, value in concepts_from_features(feats).items():
                        concept_grouped[label].setdefault(cname, []).append(float(value))
        self.prototypes = {}
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
        return self

    def add_confirmed_sample(self, label, image_path):
        feats_list = self._feature_list_for_sample(image_path)
        self.samples.setdefault(label, []).append(image_path)
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
                for cname, value in concepts_from_features(feats).items():
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
        return _vector_similarity(a, b, scale=2.5)

    def _group_detail(self, detail):
        out = {}
        for group, names in self.feature_groups.items():
            vals = [float(detail[name]) for name in names if name in detail]
            if vals:
                out[group] = float(np.mean(vals))
        return out

    def predict(self, image_path, weights):
        feats = extract_features(image_path)
        sample_concepts = concepts_from_features(feats) if self.concepts_enabled else {}
        results = []
        for label, proto in self.prototypes.items():
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
            system_detail = {}
            for name in self.system_feature_names:
                if name in proto and name in feats:
                    system_detail[name] = self._feature_similarity(name, feats[name], proto[name])
            feature_score = score / max(total_w, 1e-8)
            final = feature_score
            concept_score = None
            concept_proto = self.concept_prototypes.get(label, {})
            if self.concepts_enabled and concept_proto:
                concept_score = concept_similarity(sample_concepts, concept_proto)
                detail['concept'] = concept_score
                cw = max(0.0, min(0.8, self.concept_score_weight))
                final = (1.0 - cw) * feature_score + cw * concept_score
            results.append({
                'label': label,
                'score': final,
                'feature_score': feature_score,
                'concept_score': concept_score,
                'detail': detail,
                'group_detail': self._group_detail(detail),
                'system_detail': system_detail,
                'concepts': sample_concepts,
                'class_concepts': concept_proto,
            })
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def export(self):
        return {
            'feature_prototypes': {
                label: {k: v.tolist() for k, v in feats.items()}
                for label, feats in self.prototypes.items()
            },
            'concept_prototypes': self.concept_prototypes,
            'concept_config': {
                'enable': self.concepts_enabled,
                'score_weight': self.concept_score_weight,
            },
            'feature_groups': self.feature_groups,
            'system_features': self.system_feature_names,
        }
