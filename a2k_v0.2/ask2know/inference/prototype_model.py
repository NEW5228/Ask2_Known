import numpy as np
from ask2know.features.basic_features import extract_features

def _mean_vectors(vectors):
    if not vectors:
        return None
    return np.mean(np.stack(vectors), axis=0)

class PrototypeModel:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.prototypes = {}
        self.samples = {}

    def fit(self, samples):
        grouped = {}
        self.samples = {}
        for sample in samples:
            label = sample['label']
            feats = extract_features(sample['path'])
            grouped.setdefault(label, {name: [] for name in self.feature_names})
            self.samples.setdefault(label, [])
            self.samples[label].append(sample['path'])
            for name in self.feature_names:
                grouped[label][name].append(feats[name])
        self.prototypes = {}
        for label, fdict in grouped.items():
            self.prototypes[label] = {}
            for name in self.feature_names:
                self.prototypes[label][name] = _mean_vectors(fdict[name])
        return self

    def add_confirmed_sample(self, label, image_path):
        feats = extract_features(image_path)
        self.samples.setdefault(label, []).append(image_path)
        n = len(self.samples[label])
        self.prototypes.setdefault(label, {})
        for name in self.feature_names:
            old = self.prototypes[label].get(name)
            self.prototypes[label][name] = feats[name] if old is None else (old * (n - 1) + feats[name]) / n

    def _feature_similarity(self, a, b):
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        cos = float(np.dot(a, b) / denom)
        return (cos + 1.0) / 2.0

    def predict(self, image_path, weights):
        feats = extract_features(image_path)
        results = []
        for label, proto in self.prototypes.items():
            detail = {}
            score = 0.0
            total_w = 0.0
            for name, w in weights.items():
                if name not in proto:
                    continue
                sim = self._feature_similarity(feats[name], proto[name])
                detail[name] = sim
                score += float(w) * sim
                total_w += float(w)
            final = score / max(total_w, 1e-8)
            results.append({'label': label, 'score': final, 'detail': detail})
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def export(self):
        return {label: {k: v.tolist() for k, v in feats.items()} for label, feats in self.prototypes.items()}
