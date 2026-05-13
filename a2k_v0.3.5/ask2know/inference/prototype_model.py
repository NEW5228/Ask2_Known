import cv2
import numpy as np
from ask2know.features.basic_features import extract_features, extract_features_from_image


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
    def __init__(self, feature_names, augmentation_config=None):
        self.feature_names = feature_names
        self.augmentation_config = augmentation_config or {'enable': False}
        self.prototypes = {}
        self.samples = {}

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
        self.samples = {}
        for sample in samples:
            label = sample['label']
            feats_list = self._feature_list_for_sample(sample['path'])
            self.samples.setdefault(label, [])
            self.samples[label].append(sample['path'])
            for feats in feats_list:
                grouped.setdefault(label, {name: [] for name in self.feature_names if name in feats})
                for name in self.feature_names:
                    if name in feats:
                        grouped[label].setdefault(name, []).append(feats[name])
        self.prototypes = {}
        for label, fdict in grouped.items():
            self.prototypes[label] = {}
            for name in self.feature_names:
                if name in fdict:
                    self.prototypes[label][name] = _mean_vectors(fdict[name])
        return self

    def add_confirmed_sample(self, label, image_path):
        feats = extract_features(image_path)
        self.samples.setdefault(label, []).append(image_path)
        n = len(self.samples[label])
        self.prototypes.setdefault(label, {})
        for name in self.feature_names:
            if name not in feats:
                continue
            old = self.prototypes[label].get(name)
            self.prototypes[label][name] = feats[name] if old is None else (old * (n - 1) + feats[name]) / n

    def _feature_similarity(self, name, a, b):
        if name == 'color':
            return _hist_similarity(a, b)
        if name == 'contour':
            return _vector_similarity(a, b, scale=5.5)
        if name == 'texture':
            return _vector_similarity(a, b, scale=4.5)
        if name == 'size':
            return _vector_similarity(a, b, scale=2.2)
        return _vector_similarity(a, b, scale=2.5)

    def predict(self, image_path, weights):
        feats = extract_features(image_path)
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
            final = score / max(total_w, 1e-8)
            results.append({'label': label, 'score': final, 'detail': detail})
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def export(self):
        return {label: {k: v.tolist() for k, v in feats.items()} for label, feats in self.prototypes.items()}
