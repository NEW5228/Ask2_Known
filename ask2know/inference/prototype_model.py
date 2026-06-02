import cv2
import numpy as np
from ask2know.features.basic_features import extract_features, extract_features_from_image
from ask2know.features.deep_adapter import DeepFeatureAdapter, DEFAULT_DEEP_FEATURE_NAME
from ask2know.concepts.basic_concepts import concepts_from_features, concept_similarity
from ask2know.inference.hierarchy import label_hierarchy

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
        self.text_batch_size = max(1, int(self.text_config.get('batch_size', 64)))
        self.text_tiebreak_enabled = bool(self.text_config.get('tiebreak_enable', True))
        self.text_tiebreak_max_margin = max(0.0, float(self.text_config.get('tiebreak_max_score_margin', 0.020)))
        self.text_tiebreak_min_gap = max(0.0, float(self.text_config.get('tiebreak_min_text_gap', 0.008)))
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
        self.pair_confusion_config = dict(self.similarity_config.get('pair_confusion_rerank', {}))
        self.pair_confusion_enabled = bool(self.pair_confusion_config.get('enable', False))
        self.pair_confusion_max_margin = max(0.0, float(self.pair_confusion_config.get('max_score_margin', 0.018)))
        self.pair_confusion_min_pair_similarity = max(0.0, min(
            1.0,
            float(self.pair_confusion_config.get('min_pair_similarity', 0.96)),
        ))
        self.pair_confusion_min_local_gap = max(0.0, float(self.pair_confusion_config.get('min_local_gap', 0.018)))
        self.pair_confusion_score_weight = max(
            0.0,
            min(0.5, float(self.pair_confusion_config.get('score_weight', 0.10))),
        )
        self.pair_confusion_allow_rank_flip = bool(self.pair_confusion_config.get('allow_rank_flip', True))
        self.pair_confusion_support_sources = max(1, int(self.pair_confusion_config.get('min_support_sources_for_flip', 3)))
        self.pair_confusion_min_training_risk = max(0, int(self.pair_confusion_config.get('min_training_risk_count', 2)))
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
        self.crop_rerank_match_same_crop_id = bool(self.crop_rerank_config.get('match_same_crop_id', True))
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
        self.hierarchy_config = dict(self.similarity_config.get('hierarchy', {}))
        self.hierarchy_enabled = bool(self.hierarchy_config.get('enable', False))
        self.hierarchy_allow_aggressive = bool(self.hierarchy_config.get('allow_aggressive', False))
        hierarchy_max_weight = 0.12 if self.hierarchy_allow_aggressive else 0.03
        hierarchy_max_margin = (
            max(0.0, float(self.hierarchy_config.get('max_score_margin', 0.018)))
            if self.hierarchy_allow_aggressive
            else min(0.018, max(0.0, float(self.hierarchy_config.get('max_score_margin', 0.018))))
        )
        self.hierarchy_score_weight = max(
            0.0,
            min(hierarchy_max_weight, float(self.hierarchy_config.get('score_weight', 0.02))),
        )
        self.hierarchy_candidate_count = max(2, int(self.hierarchy_config.get('max_candidate_classes', 4)))
        self.hierarchy_min_group_size = max(2, int(self.hierarchy_config.get('min_group_size', 2)))
        self.hierarchy_max_score_margin = hierarchy_max_margin
        self.hierarchy_min_gap = max(0.006, float(self.hierarchy_config.get('min_gap', 0.006)))
        self.hierarchy_require_shared_group = bool(self.hierarchy_config.get('require_shared_group', True))
        self.hierarchy_min_support_sources = max(0, int(self.hierarchy_config.get('min_support_sources', 2)))
        self.hierarchy_level_weights = {
            str(name): max(0.0, float(weight))
            for name, weight in dict(self.hierarchy_config.get('level_weights') or {}).items()
        }
        self.robust_config = dict(self.similarity_config.get('robust_prototype', {}))
        self.robust_enabled = bool(self.robust_config.get('enable', True))
        self.robust_deep_only = bool(self.robust_config.get('deep_only', True))
        self.robust_min_samples = max(2, int(self.robust_config.get('min_samples', 24)))
        self.robust_trim_fraction = max(0.0, min(0.4, float(self.robust_config.get('trim_fraction', 0.08))))
        self.robust_report_margin = max(0.0, float(self.robust_config.get('report_margin', 0.015)))
        self.robust_top_outliers = max(1, int(self.robust_config.get('top_outliers_per_class', 5)))
        self.robust_confusion_filter_enabled = bool(self.robust_config.get('confusion_filter', False))
        self.robust_confusion_filter_margin = float(self.robust_config.get('confusion_filter_margin', 0.0))
        self.robust_confusion_filter_max_fraction = max(
            0.0,
            min(0.5, float(self.robust_config.get('confusion_filter_max_fraction', 0.20))),
        )
        self.robust_confusion_filter_min_samples = max(
            2,
            int(self.robust_config.get('confusion_filter_min_samples', 20)),
        )
        self.local_evidence_filter_enabled = bool(self.robust_config.get('local_evidence_filter', False))
        self.local_evidence_filter_margin = float(self.robust_config.get('local_evidence_filter_margin', 0.0))
        self.local_evidence_filter_max_fraction = max(
            0.0,
            min(0.5, float(self.robust_config.get('local_evidence_filter_max_fraction', 0.25))),
        )
        self.local_evidence_filter_min_samples = max(
            1,
            int(self.robust_config.get('local_evidence_filter_min_samples', 8)),
        )
        self.fast_config = dict(self.similarity_config.get('fast_search', {}))
        self.fast_search_enabled = bool(self.fast_config.get('enable', True))
        self.fast_candidate_count = max(2, int(self.fast_config.get('candidate_classes', 32)))
        self.fast_deep_only_local_scores = bool(self.fast_config.get('deep_only_local_scores', False))
        self.fast_skip_augmented_basic_features = bool(
            self.fast_config.get('skip_augmented_basic_features_when_deep_enabled', True)
        )
        self.deep_adapter = DeepFeatureAdapter(deep_feature_config, cache_dir=deep_cache_dir)
        self.deep_feature_name = self.deep_adapter.feature_name or DEFAULT_DEEP_FEATURE_NAME
        self.prototypes = {}
        self.sub_prototypes = {}
        self.text_prototypes = {}
        self.pairwise_similarities = {}
        self.concept_prototypes = {}
        self.label_hierarchies = {}
        self.hierarchy_group_labels = {}
        self.hierarchy_prototypes = {}
        self.samples = {}
        self.sample_features = {}
        self.feature_counts = {}
        self.concept_counts = {}
        self.prototype_stats = {}
        self.training_quality_report = {}
        self.training_confusion_pairs = {}
        self.deep_sample_index = {}
        self.crop_sample_index = {}

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
        if (
            self.fast_skip_augmented_basic_features
            and self.deep_adapter.is_enabled()
            and not self.deep_adapter.include_augmented
        ):
            return feats_list
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
                    'crop_embeddings': [],
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
        self.label_hierarchies = {}
        self.hierarchy_group_labels = {}
        self.hierarchy_prototypes = {}
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
        self._apply_confusion_filtered_deep_prototypes()
        self._mark_confusion_risk_samples()
        self._build_local_evidence_indexes()
        self._build_sub_prototypes(self.prototypes.keys())
        self._build_text_prototypes(self.prototypes.keys())
        self._build_pairwise_similarities()
        self._build_hierarchy_index()
        self._build_training_quality_report()
        return self

    def add_confirmed_sample(self, label, image_path):
        feats_list = self._feature_list_for_sample(image_path)
        self.samples.setdefault(label, []).append(image_path)
        if feats_list:
            self.sample_features.setdefault(label, []).append({
                'path': image_path,
                'features': feats_list[0],
                'crop_embeddings': [],
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
        self._refresh_label_robust_deep_prototype(label)
        self._mark_confusion_risk_samples()
        self._build_local_evidence_indexes()
        self._build_sub_prototypes(self.prototypes.keys())
        self._build_text_prototypes([label])
        self._build_pairwise_similarities()
        self._build_hierarchy_index()
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
        if self.fast_deep_only_local_scores:
            scores, items = self._deep_index_scores(label, feats)
        else:
            scores, items = None, []
        if scores is not None:
            neighbors = []
            for idx, score in self._top_scores(scores, self.knn_k):
                item = items[idx] if idx < len(items) else {}
                detail = {self.deep_feature_name: score}
                neighbors.append({
                    'path': str(item.get('path')),
                    'score': float(score),
                    'detail': detail,
                    'group_detail': self._group_detail(detail),
                })
            if not neighbors:
                return None, []
            knn_score = float(np.mean([x['score'] for x in neighbors]))
            return knn_score, neighbors
        neighbors = []
        for item in self._local_evidence_items(label):
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

    def _primary_deep_vector(self, feats):
        vec = (feats or {}).get(self.deep_feature_name)
        if vec is None:
            vec = (feats or {}).get(DEFAULT_DEEP_FEATURE_NAME)
        return vec

    def _label_deep_vector(self, label):
        proto = self.prototypes.get(label, {})
        vec = proto.get(self.deep_feature_name)
        if vec is None:
            vec = proto.get(DEFAULT_DEEP_FEATURE_NAME)
        return vec

    def _deep_proto_index(self, labels):
        index_labels = []
        vectors = []
        for label in labels:
            vec = self._label_deep_vector(label)
            if vec is None:
                continue
            index_labels.append(label)
            vectors.append(_l2_normalize(vec))
        if not vectors:
            return [], np.zeros((0, 0), dtype=np.float32)
        return index_labels, np.stack(vectors).astype(np.float32)

    def _deep_confusion_rows_for_label(self, label, labels=None, proto_vectors=None, proto_index=None):
        labels = list(labels or sorted(self.prototypes.keys()))
        if proto_index is not None:
            index_labels, proto_matrix = proto_index
            if label not in index_labels or proto_matrix.size == 0:
                return []
            own_index = index_labels.index(label)
            rows = []
            for item in self.sample_features.get(label, []):
                vec = self._primary_deep_vector(item.get('features') or {})
                if vec is None:
                    continue
                sample_vec = _l2_normalize(vec)
                sims = (proto_matrix @ sample_vec + 1.0) * 0.5
                own_score = float(sims[own_index])
                competitor_sims = sims.copy()
                competitor_sims[own_index] = -1.0
                competitor_index = int(np.argmax(competitor_sims))
                competitor_score = float(competitor_sims[competitor_index])
                rows.append({
                    'item': item,
                    'vector': vec,
                    'path': str(item.get('path')),
                    'own_score': own_score,
                    'nearest_competitor': index_labels[competitor_index],
                    'nearest_competitor_score': competitor_score,
                    'margin': float(own_score - competitor_score),
                })
            return rows

        if proto_vectors is None:
            proto_vectors = {
                candidate: self._label_deep_vector(candidate)
                for candidate in labels
            }
            proto_vectors = {
                candidate: vec for candidate, vec in proto_vectors.items()
                if vec is not None
            }
        own_vec = proto_vectors.get(label)
        if own_vec is None:
            return []
        rows = []
        for item in self.sample_features.get(label, []):
            vec = self._primary_deep_vector(item.get('features') or {})
            if vec is None:
                continue
            own_score = _cosine_similarity(vec, own_vec)
            competitor_label = None
            competitor_score = None
            for other, other_vec in proto_vectors.items():
                if other == label:
                    continue
                score = _cosine_similarity(vec, other_vec)
                if competitor_score is None or score > competitor_score:
                    competitor_label = other
                    competitor_score = score
            if competitor_score is None:
                continue
            rows.append({
                'item': item,
                'vector': vec,
                'path': str(item.get('path')),
                'own_score': float(own_score),
                'nearest_competitor': competitor_label,
                'nearest_competitor_score': float(competitor_score),
                'margin': float(own_score - competitor_score),
            })
        return rows

    def _mark_confusion_risk_samples(self):
        self.training_confusion_pairs = {}
        for rows in self.sample_features.values():
            for item in rows:
                item.pop('confusion_risk', None)
                item.pop('confusion_risk_reason', None)
                item.pop('confusion_margin', None)
                item.pop('nearest_competitor', None)
        if not (self.local_evidence_filter_enabled and self.local_evidence_filter_max_fraction > 0.0):
            return

        labels = sorted(self.prototypes.keys())
        proto_index = self._deep_proto_index(labels)
        if len(proto_index[0]) < 2:
            return

        for label in labels:
            rows = self._deep_confusion_rows_for_label(label, labels=labels, proto_index=proto_index)
            if not rows:
                continue
            sample_count = len(rows)
            max_mark = int(np.floor(sample_count * self.local_evidence_filter_max_fraction))
            if max_mark <= 0:
                continue
            risky = [
                row for row in sorted(rows, key=lambda item: item['margin'])
                if row['margin'] <= self.local_evidence_filter_margin
            ]
            mark_count = min(max_mark, len(risky))
            if mark_count <= 0:
                continue
            if sample_count - mark_count < self.local_evidence_filter_min_samples:
                mark_count = max(0, sample_count - self.local_evidence_filter_min_samples)
            for row in risky[:mark_count]:
                item = row['item']
                item['confusion_risk'] = True
                item['confusion_risk_reason'] = 'nearest_competitor_margin'
                item['confusion_margin'] = row['margin']
                item['nearest_competitor'] = row['nearest_competitor']
                pair_key = self._pair_key(label, row['nearest_competitor'])
                self.training_confusion_pairs[pair_key] = self.training_confusion_pairs.get(pair_key, 0) + 1

    def _local_evidence_items(self, label):
        items = list(self.sample_features.get(label, []))
        if not self.local_evidence_filter_enabled:
            return items
        kept = [item for item in items if not item.get('confusion_risk')]
        return kept if kept else items

    def _build_local_evidence_indexes(self):
        self.deep_sample_index = {}
        self.crop_sample_index = {}
        for label in self.sample_features.keys():
            deep_vectors = []
            deep_items = []
            crop_groups = {}
            for item in self._local_evidence_items(label):
                feats = item.get('features') or {}
                vec = self._primary_deep_vector(feats)
                if vec is not None:
                    deep_vectors.append(_l2_normalize(vec))
                    deep_items.append(item)
                for crop_id, crop_vec in self._crop_rows_for_scoring(item.get('crop_embeddings') or []):
                    crop_groups.setdefault(str(crop_id), {'vectors': [], 'items': []})
                    crop_groups[str(crop_id)]['vectors'].append(_l2_normalize(crop_vec))
                    crop_groups[str(crop_id)]['items'].append(item)
            if deep_vectors:
                self.deep_sample_index[label] = {
                    'matrix': np.stack(deep_vectors).astype(np.float32),
                    'items': deep_items,
                }
            indexed_crops = {}
            for crop_id, group in crop_groups.items():
                vectors = group.get('vectors') or []
                if vectors:
                    indexed_crops[crop_id] = {
                        'matrix': np.stack(vectors).astype(np.float32),
                        'items': group.get('items') or [],
                    }
            if indexed_crops:
                self.crop_sample_index[label] = indexed_crops

    def _ensure_crop_index_for_label(self, label):
        if label in self.crop_sample_index:
            return
        if not (self.crop_rerank_enabled and self.deep_adapter.is_enabled()):
            return
        crop_groups = {}
        for item in self._local_evidence_items(label):
            if not item.get('crop_embeddings'):
                path = item.get('path')
                if path:
                    item['crop_embeddings'] = self._extract_crop_embeddings_for_path(path)
            for crop_id, crop_vec in self._crop_rows_for_scoring(item.get('crop_embeddings') or []):
                crop_groups.setdefault(str(crop_id), {'vectors': [], 'items': []})
                crop_groups[str(crop_id)]['vectors'].append(_l2_normalize(crop_vec))
                crop_groups[str(crop_id)]['items'].append(item)
        indexed_crops = {}
        for crop_id, group in crop_groups.items():
            vectors = group.get('vectors') or []
            if vectors:
                indexed_crops[crop_id] = {
                    'matrix': np.stack(vectors).astype(np.float32),
                    'items': group.get('items') or [],
                }
        if indexed_crops:
            self.crop_sample_index[label] = indexed_crops

    def _deep_index_scores(self, label, feats_or_vec):
        index = self.deep_sample_index.get(label) or {}
        matrix = index.get('matrix')
        if matrix is None or matrix.size == 0:
            return None, []
        if isinstance(feats_or_vec, dict):
            vec = self._primary_deep_vector(feats_or_vec)
        else:
            vec = feats_or_vec
        if vec is None:
            return None, []
        query = _l2_normalize(vec)
        if query.size == 0 or matrix.shape[1] != query.size:
            return None, []
        scores = (matrix @ query + 1.0) * 0.5
        return scores.astype(np.float32), list(index.get('items') or [])

    def _top_scores(self, scores, top_k):
        if scores is None or len(scores) <= 0:
            return []
        top_k = max(1, min(int(top_k), len(scores)))
        if top_k >= len(scores):
            indexes = np.argsort(scores)[::-1]
        else:
            indexes = np.argpartition(scores, -top_k)[-top_k:]
            indexes = indexes[np.argsort(scores[indexes])[::-1]]
        return [(int(idx), float(scores[idx])) for idx in indexes]

    def _apply_confusion_filtered_deep_prototypes(self):
        if not (
            self.robust_enabled
            and self.robust_confusion_filter_enabled
            and self.robust_confusion_filter_max_fraction > 0.0
        ):
            return
        labels = sorted(self.prototypes.keys())
        proto_index = self._deep_proto_index(labels)
        if len(proto_index[0]) < 2:
            return

        for label in labels:
            rows = self._deep_confusion_rows_for_label(label, labels=labels, proto_index=proto_index)
            sample_count = len(rows)
            if sample_count < self.robust_confusion_filter_min_samples:
                continue
            max_remove = int(np.floor(sample_count * self.robust_confusion_filter_max_fraction))
            if max_remove <= 0:
                continue
            risky = [
                row for row in sorted(rows, key=lambda item: item['margin'])
                if row['margin'] <= self.robust_confusion_filter_margin
            ]
            remove_count = min(max_remove, len(risky))
            if remove_count <= 0 or sample_count - remove_count < self.robust_confusion_filter_min_samples:
                continue
            remove_paths = {row['path'] for row in risky[:remove_count]}
            kept_vectors = [row['vector'] for row in rows if row['path'] not in remove_paths]
            mean, stats = _robust_mean_vectors(
                kept_vectors,
                trim_fraction=self.robust_trim_fraction,
                min_samples=self.robust_min_samples,
            )
            if mean is None:
                continue
            self.prototypes.setdefault(label, {})[self.deep_feature_name] = mean
            self.feature_counts.setdefault(label, {})[self.deep_feature_name] = len(kept_vectors)
            stats = dict(stats)
            stats.update({
                'confusion_filtered_count': int(remove_count),
                'confusion_filter_margin': float(self.robust_confusion_filter_margin),
                'confusion_filter_max_fraction': float(self.robust_confusion_filter_max_fraction),
                'confusion_filter_worst_margin': float(risky[0]['margin']) if risky else None,
                'confusion_filter_removed': [
                    {
                        'path': row['path'],
                        'own_score': row['own_score'],
                        'nearest_competitor': row['nearest_competitor'],
                        'nearest_competitor_score': row['nearest_competitor_score'],
                        'margin': row['margin'],
                    }
                    for row in risky[:min(remove_count, self.robust_top_outliers)]
                ],
            })
            self.prototype_stats.setdefault(label, {})[self.deep_feature_name] = stats

    def _refresh_label_robust_deep_prototype(self, label):
        if not self.robust_enabled:
            return
        vectors = []
        for item in self.sample_features.get(label, []):
            vec = self._primary_deep_vector(item.get('features') or {})
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
        proto_index = self._deep_proto_index(labels)
        for label in labels:
            raw_rows = self._deep_confusion_rows_for_label(label, labels=labels, proto_index=proto_index)
            rows = []
            for row in raw_rows:
                item = row.get('item') or {}
                rows.append({
                    'path': row['path'],
                    'own_score': row['own_score'],
                    'nearest_competitor': row['nearest_competitor'],
                    'nearest_competitor_score': row['nearest_competitor_score'],
                    'margin': row['margin'],
                    'local_evidence_filtered': bool(item.get('confusion_risk')),
                })
            rows.sort(key=lambda row: row['own_score'])
            risk_rows = [
                row for row in rows
                if row['margin'] is not None and row['margin'] < self.robust_report_margin
            ]
            self.training_quality_report[label] = {
                'sample_count': len(rows),
                'local_evidence_filtered_count': sum(1 for row in rows if row.get('local_evidence_filtered')),
                'prototype_stats': self.prototype_stats.get(label, {}).get(self.deep_feature_name, {}),
                'outliers': rows[:self.robust_top_outliers],
                'confusion_risk_samples': sorted(
                    risk_rows,
                    key=lambda row: row['margin'] if row['margin'] is not None else 0.0,
                )[:self.robust_top_outliers],
            }

    def _build_hierarchy_index(self):
        self.label_hierarchies = {}
        self.hierarchy_group_labels = {}
        self.hierarchy_prototypes = {}
        if not self.hierarchy_enabled:
            return

        for label in sorted(self.prototypes.keys()):
            path = label_hierarchy(label, self.hierarchy_config)
            self.label_hierarchies[label] = path
            for item in path:
                group_key = self._hierarchy_group_key(item)
                self.hierarchy_group_labels.setdefault(group_key, []).append(label)

        for group_key, labels in self.hierarchy_group_labels.items():
            if len(labels) < self.hierarchy_min_group_size:
                continue
            grouped = {}
            for label in labels:
                proto = self.prototypes.get(label, {})
                for name, value in proto.items():
                    if value is not None:
                        grouped.setdefault(name, []).append(value)
            self.hierarchy_prototypes[group_key] = {
                name: _mean_vectors(values)
                for name, values in grouped.items()
                if values
            }

    def _hierarchy_group_key(self, item):
        return (str(item.get('level')), str(item.get('key')))

    def _hierarchy_level_weight(self, level):
        if self.hierarchy_level_weights:
            return self.hierarchy_level_weights.get(str(level), 0.0)
        return 1.0

    def _hierarchy_group_keys_for_label(self, label):
        return {
            self._hierarchy_group_key(item)
            for item in self.label_hierarchies.get(label, [])
            if self._hierarchy_group_key(item) in self.hierarchy_prototypes
        }

    def _share_hierarchy_group(self, left_label, right_label):
        left = self._hierarchy_group_keys_for_label(left_label)
        right = self._hierarchy_group_keys_for_label(right_label)
        return bool(left and right and left.intersection(right))

    def _supporting_source_count(self, row, opponent, min_gap=0.001):
        count = 0
        for key in (
            'base_score',
            'prototype_score',
            'subprototype_score',
            'knn_score',
            'text_semantic_score',
            'pairwise_score',
            'crop_rerank_score',
            'late_fusion_score',
        ):
            value = row.get(key)
            other = opponent.get(key)
            if value is None or other is None:
                continue
            try:
                if float(value) - float(other) > min_gap:
                    count += 1
            except (TypeError, ValueError):
                continue
        return count

    def _hierarchy_score(self, label, feats, weights):
        evidence = []
        weighted_sum = 0.0
        total_weight = 0.0
        for item in self.label_hierarchies.get(label, []):
            group_key = self._hierarchy_group_key(item)
            proto = self.hierarchy_prototypes.get(group_key)
            if not proto:
                continue
            score, detail = self._weighted_feature_score(feats, proto, weights)
            if not detail:
                continue
            level_weight = self._hierarchy_level_weight(item.get('level'))
            if level_weight <= 0.0:
                continue
            weighted_sum += level_weight * score
            total_weight += level_weight
            evidence.append({
                'level': item.get('level'),
                'key': item.get('key'),
                'display': item.get('display'),
                'score': float(score),
                'weight': float(level_weight),
                'member_count': len(self.hierarchy_group_labels.get(group_key, [])),
            })
        if total_weight <= 0.0:
            return None, evidence
        return float(weighted_sum / total_weight), evidence

    def _apply_hierarchy_rerank(self, rows, feats, weights):
        for row in rows:
            row['hierarchy_score'] = None
            row['hierarchy_score_weight_used'] = 0.0
            row['hierarchy_gate_reason'] = 'disabled' if not self.hierarchy_enabled else 'not_candidate'
            row['hierarchy_evidence'] = []
        if not self.hierarchy_enabled or len(rows) < 2:
            return

        ranked = sorted(rows, key=lambda item: float(item['score']), reverse=True)
        score_margin = float(ranked[0]['score']) - float(ranked[1]['score'])
        candidate_count = self.hierarchy_candidate_count if self.hierarchy_allow_aggressive else 2
        candidates = ranked[:min(len(ranked), candidate_count)]
        if score_margin > self.hierarchy_max_score_margin:
            for row in candidates:
                row['hierarchy_gate_reason'] = 'score_margin_too_large'
            return
        if (
            self.hierarchy_require_shared_group
            and not self._share_hierarchy_group(ranked[0]['label'], ranked[1]['label'])
        ):
            for row in candidates:
                row['hierarchy_gate_reason'] = 'no_shared_hierarchy_group'
            return

        scored = []
        for row in candidates:
            hierarchy_score, evidence = self._hierarchy_score(row['label'], feats, weights)
            row['hierarchy_score'] = hierarchy_score
            row['hierarchy_evidence'] = evidence
            if hierarchy_score is None:
                row['hierarchy_gate_reason'] = 'missing_hierarchy_evidence'
            else:
                scored.append(row)
        if len(scored) < 2:
            for row in scored:
                row['hierarchy_gate_reason'] = 'insufficient_competition'
            return

        hierarchy_ranked = sorted(scored, key=lambda item: float(item['hierarchy_score']), reverse=True)
        hierarchy_gap = float(hierarchy_ranked[0]['hierarchy_score']) - float(hierarchy_ranked[1]['hierarchy_score'])
        if hierarchy_gap < self.hierarchy_min_gap:
            for row in scored:
                row['hierarchy_gate_reason'] = 'weak_hierarchy_gap'
            return

        hierarchy_winner = hierarchy_ranked[0]
        score_top = ranked[0]
        if hierarchy_winner is not score_top:
            support_count = self._supporting_source_count(hierarchy_winner, score_top)
            if support_count < self.hierarchy_min_support_sources:
                for row in scored:
                    row['hierarchy_gate_reason'] = 'hierarchy_lacks_source_support'
                return

        for row in scored:
            row['hierarchy_score_weight_used'] = self.hierarchy_score_weight
            row['hierarchy_gate_reason'] = 'applied'
            row['score'] = (
                (1.0 - self.hierarchy_score_weight) * float(row['score'])
                + self.hierarchy_score_weight * float(row['hierarchy_score'])
            )

    def _pairwise_local_score(self, label, feats, weights):
        scores, _ = self._deep_index_scores(label, feats) if self.fast_deep_only_local_scores else (None, [])
        if scores is not None:
            top = [score for _, score in self._top_scores(scores, self.pairwise_local_k)]
            if not top:
                return None
            return float(0.65 * top[0] + 0.35 * np.mean(top))
        neighbors = []
        for item in self._local_evidence_items(label):
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
        return [vec for _, vec in self._crop_rows_for_scoring(crop_embeddings)]

    def _crop_rows_for_scoring(self, crop_embeddings):
        rows = []
        for item in crop_embeddings or []:
            crop_id = str(item.get('crop_id') or '')
            if not self.crop_rerank_use_full_crop and crop_id == 'full':
                continue
            vec = item.get('vector')
            if vec is not None:
                rows.append((crop_id, vec))
        return rows

    def _crop_local_score(self, label, query_crops):
        query_rows = self._crop_rows_for_scoring(query_crops)
        if not query_rows:
            return None
        self._ensure_crop_index_for_label(label)
        crop_index = self.crop_sample_index.get(label) or {}
        if crop_index:
            candidate_scores = []
            for query_id, query_vec in query_rows:
                query = _l2_normalize(query_vec)
                if query.size == 0:
                    continue
                groups = []
                if self.crop_rerank_match_same_crop_id:
                    group = crop_index.get(str(query_id))
                    if group:
                        groups.append(group)
                else:
                    groups.extend(crop_index.values())
                for group in groups:
                    matrix = group.get('matrix')
                    if matrix is None or matrix.size == 0 or matrix.shape[1] != query.size:
                        continue
                    scores = (matrix @ query + 1.0) * 0.5
                    candidate_scores.extend(score for _, score in self._top_scores(scores, self.crop_rerank_local_k))
            if not candidate_scores:
                return None
            candidate_scores.sort(reverse=True)
            top = candidate_scores[:self.crop_rerank_local_k]
            return float(0.70 * top[0] + 0.30 * np.mean(top))
        candidate_scores = []
        for item in self._local_evidence_items(label):
            sample_rows = self._crop_rows_for_scoring(item.get('crop_embeddings') or [])
            for query_id, query_vec in query_rows:
                for sample_id, sample_vec in sample_rows:
                    if self.crop_rerank_match_same_crop_id and query_id != sample_id:
                        continue
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

    def _deep_local_score(self, label, feats, local_k=5):
        image_vec = self._primary_deep_vector(feats)
        if image_vec is None:
            return None
        indexed_scores, _ = self._deep_index_scores(label, image_vec)
        if indexed_scores is not None:
            top = [score for _, score in self._top_scores(indexed_scores, max(1, int(local_k)))]
            if not top:
                return None
            return float(0.70 * top[0] + 0.30 * np.mean(top))
        scores = []
        for item in self._local_evidence_items(label):
            sample_vec = self._primary_deep_vector(item.get('features') or {})
            if sample_vec is not None:
                scores.append(_cosine_similarity(image_vec, sample_vec))
        if not scores:
            return None
        scores.sort(reverse=True)
        top = scores[:max(1, int(local_k))]
        return float(0.70 * top[0] + 0.30 * np.mean(top))

    def _pair_confusion_support_count(self, row, opponent, min_gap=0.001):
        count = 0
        for key in (
            'pair_confusion_deep_score',
            'crop_rerank_score',
            'pairwise_score',
            'knn_score',
            'subprototype_score',
            'late_fusion_score',
        ):
            value = row.get(key)
            other = opponent.get(key)
            if value is None or other is None:
                continue
            try:
                if float(value) - float(other) > min_gap:
                    count += 1
            except (TypeError, ValueError):
                continue
        return count

    def _pair_confusion_score_for_row(self, row):
        weighted_sum = 0.0
        total_weight = 0.0
        evidence = {}
        for key, weight in (
            ('pair_confusion_deep_score', 1.0),
            ('crop_rerank_score', 0.8),
            ('pairwise_score', 0.6),
            ('knn_score', 0.4),
            ('subprototype_score', 0.3),
            ('late_fusion_score', 0.3),
        ):
            value = row.get(key)
            if value is None:
                continue
            weighted_sum += float(weight) * float(value)
            total_weight += float(weight)
            evidence[key] = float(value)
        if total_weight <= 0.0:
            return None, evidence
        return float(weighted_sum / total_weight), evidence

    def _apply_pair_confusion_rerank(self, rows, feats):
        for row in rows:
            row['pair_confusion_score'] = None
            row['pair_confusion_deep_score'] = None
            row['pair_confusion_score_weight_used'] = 0.0
            row['pair_confusion_gate_reason'] = 'disabled' if not self.pair_confusion_enabled else 'not_candidate'
            row['pair_confusion_pair_similarity'] = None
            row['pair_confusion_local_gap'] = None
            row['pair_confusion_training_risk_count'] = 0
            row['pair_confusion_evidence'] = {}
        if not self.pair_confusion_enabled or len(rows) < 2:
            return

        ranked = sorted(rows, key=lambda item: float(item['score']), reverse=True)
        top = ranked[0]
        second = ranked[1]
        score_margin = float(top['score']) - float(second['score'])
        pair_key = self._pair_key(top['label'], second['label'])
        pair_similarity = self.pairwise_similarities.get(pair_key)
        risk_count = int(self.training_confusion_pairs.get(pair_key, 0))
        for row in (top, second):
            row['pair_confusion_pair_similarity'] = pair_similarity
            row['pair_confusion_training_risk_count'] = risk_count
        if score_margin > self.pair_confusion_max_margin:
            for row in (top, second):
                row['pair_confusion_gate_reason'] = 'score_margin_too_large'
            return
        if pair_similarity is not None and pair_similarity < self.pair_confusion_min_pair_similarity:
            for row in (top, second):
                row['pair_confusion_gate_reason'] = 'pair_not_similar'
            return
        if risk_count < self.pair_confusion_min_training_risk:
            for row in (top, second):
                row['pair_confusion_gate_reason'] = 'no_training_confusion_pair'
            return

        scored = []
        for row in (top, second):
            row['pair_confusion_deep_score'] = self._deep_local_score(
                row['label'],
                feats,
                local_k=max(self.knn_k, self.pairwise_local_k),
            )
            score, evidence = self._pair_confusion_score_for_row(row)
            row['pair_confusion_score'] = score
            row['pair_confusion_evidence'] = evidence
            if score is not None:
                scored.append(row)
        if len(scored) < 2:
            for row in scored:
                row['pair_confusion_gate_reason'] = 'missing_local_evidence'
            return

        pair_ranked = sorted(scored, key=lambda item: float(item['pair_confusion_score']), reverse=True)
        local_gap = float(pair_ranked[0]['pair_confusion_score']) - float(pair_ranked[1]['pair_confusion_score'])
        for row in scored:
            row['pair_confusion_local_gap'] = local_gap
        if local_gap < self.pair_confusion_min_local_gap:
            for row in scored:
                row['pair_confusion_gate_reason'] = 'weak_local_gap'
            return

        winner = pair_ranked[0]
        loser = pair_ranked[1]
        if winner is top:
            for row in scored:
                row['pair_confusion_gate_reason'] = 'top_already_supported'
            return

        support_count = self._pair_confusion_support_count(winner, loser)
        if not self.pair_confusion_allow_rank_flip or support_count < self.pair_confusion_support_sources:
            for row in scored:
                row['pair_confusion_gate_reason'] = 'insufficient_flip_support'
            return

        for row in scored:
            row['pair_confusion_score_weight_used'] = self.pair_confusion_score_weight
            row['pair_confusion_gate_reason'] = 'pair_local_evidence'
            row['score'] = (
                (1.0 - self.pair_confusion_score_weight) * float(row['score'])
                + self.pair_confusion_score_weight * float(row['pair_confusion_score'])
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

    def _apply_text_tiebreak_rerank(self, rows):
        for row in rows:
            row['text_tiebreak_gate_reason'] = 'disabled' if not self.text_tiebreak_enabled else 'not_candidate'
            row['text_tiebreak_gap'] = None
        if not (self.text_enabled and self.text_tiebreak_enabled) or len(rows) < 2:
            return
        ranked = sorted(rows, key=lambda item: float(item['score']), reverse=True)
        top = ranked[0]
        second = ranked[1]
        score_margin = float(top['score']) - float(second['score'])
        for row in (top, second):
            row['text_tiebreak_gate_reason'] = 'score_margin_too_large'
        if score_margin > self.text_tiebreak_max_margin:
            return
        top_text = top.get('text_semantic_score')
        second_text = second.get('text_semantic_score')
        if top_text is None or second_text is None:
            for row in (top, second):
                row['text_tiebreak_gate_reason'] = 'missing_text_score'
            return
        text_gap = float(second_text) - float(top_text)
        for row in (top, second):
            row['text_tiebreak_gap'] = text_gap
        if text_gap <= self.text_tiebreak_min_gap:
            for row in (top, second):
                row['text_tiebreak_gate_reason'] = 'weak_text_gap'
            return
        second['score'] = float(top['score']) + 1e-6
        top['text_tiebreak_gate_reason'] = 'yielded_to_text'
        second['text_tiebreak_gate_reason'] = 'text_tiebreak'

    def _label_prompt_text(self, label):
        return str(label).replace('_', ' ').replace('-', ' ').strip()

    def _build_sub_prototypes(self, labels):
        if not self.subprototype_enabled:
            return
        for label in labels:
            vectors = []
            for item in self._local_evidence_items(label):
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
        prompt_rows = []
        for label in list(labels):
            readable = self._label_prompt_text(label)
            for template in self.text_prompt_templates:
                prompt_rows.append((label, template.format(label=readable)))
        grouped = {}
        for start in range(0, len(prompt_rows), self.text_batch_size):
            batch = prompt_rows[start:start + self.text_batch_size]
            vectors = self.deep_adapter.extract_text_vectors([text for _, text in batch])
            for (label, _), vec in zip(batch, vectors):
                grouped.setdefault(label, []).append(vec)
        for label, vectors in grouped.items():
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
                'knn_score': None,
                'text_semantic_score': text_score,
                'concept_score': concept_score,
                'detail': detail,
                'group_detail': self._group_detail(detail),
                'system_detail': system_detail,
                'concepts': sample_concepts,
                'class_concepts': concept_proto,
                'nearest_samples': [],
            })
        if self.knn_enabled and rows:
            if self.fast_search_enabled:
                candidate_count = min(len(rows), self.fast_candidate_count)
            else:
                candidate_count = len(rows)
            candidates = sorted(rows, key=lambda item: float(item['base_score']), reverse=True)[:candidate_count]
            for row in candidates:
                knn_score, nearest = self._nearest_samples(row['label'], feats, weights)
                row['knn_score'] = knn_score
                row['nearest_samples'] = nearest
                if knn_score is None:
                    continue
                feature_score = (
                    (1.0 - self.knn_score_weight) * float(row['prototype_score'])
                    + self.knn_score_weight * float(knn_score)
                )
                final = feature_score
                text_score = row.get('text_semantic_score')
                if text_score is not None:
                    final = (1.0 - self.text_score_weight) * final + self.text_score_weight * float(text_score)
                row['feature_score'] = feature_score
                row['score'] = final
                row['base_score'] = final
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
        self._apply_pair_confusion_rerank(rows, feats)
        self._apply_text_tiebreak_rerank(rows)
        self._apply_hierarchy_rerank(rows, feats, weights)
        results = rows
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    @staticmethod
    def _export_feature_dict(features):
        return {
            name: np.asarray(value, dtype=np.float32).tolist()
            for name, value in (features or {}).items()
            if value is not None
        }

    @staticmethod
    def _import_feature_dict(features):
        return {
            name: np.asarray(value, dtype=np.float32)
            for name, value in (features or {}).items()
            if value is not None
        }

    @staticmethod
    def _export_crop_embeddings(crop_embeddings):
        rows = []
        for item in crop_embeddings or []:
            rows.append({
                'crop_id': item.get('crop_id'),
                'box': list(item.get('box') or []),
                'vector': None if item.get('vector') is None else np.asarray(item.get('vector'), dtype=np.float32).tolist(),
            })
        return rows

    @staticmethod
    def _import_crop_embeddings(crop_embeddings):
        rows = []
        for item in crop_embeddings or []:
            rows.append({
                'crop_id': item.get('crop_id'),
                'box': list(item.get('box') or []),
                'vector': None if item.get('vector') is None else np.asarray(item.get('vector'), dtype=np.float32),
            })
        return rows

    @classmethod
    def from_export(cls, data, deep_feature_config=None, deep_cache_dir=None):
        feature_prototypes = data.get('feature_prototypes') or {}
        system_features = list(data.get('system_features') or [])
        feature_names = list(data.get('feature_names') or [])
        if not feature_names:
            for feats in feature_prototypes.values():
                for name in feats:
                    if name not in feature_names and name not in system_features:
                        feature_names.append(name)

        model = cls(
            feature_names,
            augmentation_config=data.get('augmentation_config') or {'enable': False},
            concept_config=data.get('concept_config') or {'enable': True, 'score_weight': 0.25},
            system_feature_names=system_features,
            feature_groups=data.get('feature_groups') or {},
            similarity_config=data.get('similarity_config') or {},
            deep_feature_config=deep_feature_config or data.get('deep_feature_config') or data.get('deep_features') or {},
            deep_cache_dir=deep_cache_dir,
        )
        model.prototypes = {
            label: cls._import_feature_dict(feats)
            for label, feats in feature_prototypes.items()
        }
        model.sub_prototypes = {
            label: [np.asarray(vec, dtype=np.float32) for vec in vectors]
            for label, vectors in (data.get('sub_prototypes') or {}).items()
        }
        model.text_prototypes = {
            label: np.asarray(vec, dtype=np.float32)
            for label, vec in (data.get('text_prototypes') or {}).items()
        }
        model.concept_prototypes = {
            label: {name: float(value) for name, value in concepts.items()}
            for label, concepts in (data.get('concept_prototypes') or {}).items()
        }
        model.feature_counts = {
            label: {name: int(value) for name, value in counts.items()}
            for label, counts in (data.get('feature_counts') or {}).items()
        }
        model.concept_counts = {
            label: {name: int(value) for name, value in counts.items()}
            for label, counts in (data.get('concept_counts') or {}).items()
        }
        model.prototype_stats = data.get('prototype_stats') or {}
        model.training_quality_report = data.get('training_quality_report') or {}
        model.training_confusion_pairs = {
            tuple(str(key).split('|||', 1)): int(value)
            for key, value in (data.get('training_confusion_pairs') or {}).items()
            if len(str(key).split('|||', 1)) == 2
        }
        model.samples = {
            label: list(paths)
            for label, paths in (data.get('sample_index') or {}).items()
        }
        model.sample_features = {}
        for label, rows in (data.get('sample_features') or {}).items():
            model.sample_features[label] = []
            for item in rows or []:
                model.sample_features[label].append({
                    'path': item.get('path'),
                    'features': cls._import_feature_dict(item.get('features') or {}),
                    'crop_embeddings': cls._import_crop_embeddings(item.get('crop_embeddings') or []),
                    'confusion_risk': bool(item.get('confusion_risk', False)),
                    'confusion_risk_reason': item.get('confusion_risk_reason'),
                    'confusion_margin': item.get('confusion_margin'),
                    'nearest_competitor': item.get('nearest_competitor'),
                })
        if not model.samples and model.sample_features:
            model.samples = {
                label: [item.get('path') for item in rows if item.get('path')]
                for label, rows in model.sample_features.items()
            }

        model.pairwise_similarities = {}
        for key, value in (data.get('pairwise_similarities') or {}).items():
            parts = str(key).split('|||', 1)
            if len(parts) == 2:
                model.pairwise_similarities[tuple(parts)] = float(value)
        if not model.pairwise_similarities:
            model._build_pairwise_similarities()
        if not model.training_confusion_pairs and model.sample_features:
            model._mark_confusion_risk_samples()
        model._build_local_evidence_indexes()
        model._build_hierarchy_index()
        return model

    def export(self, include_sample_features=True):
        return {
            'schema_version': 'prototype_model_v2',
            'feature_names': list(self.feature_names),
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
            'augmentation_config': self.augmentation_config,
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
                    'batch_size': self.text_batch_size,
                    'tiebreak_enable': self.text_tiebreak_enabled,
                    'tiebreak_max_score_margin': self.text_tiebreak_max_margin,
                    'tiebreak_min_text_gap': self.text_tiebreak_min_gap,
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
                    'match_same_crop_id': self.crop_rerank_match_same_crop_id,
                    'trigger_mode': self.crop_rerank_trigger_mode,
                },
                'pair_confusion_rerank': {
                    'enable': self.pair_confusion_enabled,
                    'max_score_margin': self.pair_confusion_max_margin,
                    'min_pair_similarity': self.pair_confusion_min_pair_similarity,
                    'min_local_gap': self.pair_confusion_min_local_gap,
                    'score_weight': self.pair_confusion_score_weight,
                    'allow_rank_flip': self.pair_confusion_allow_rank_flip,
                    'min_support_sources_for_flip': self.pair_confusion_support_sources,
                    'min_training_risk_count': self.pair_confusion_min_training_risk,
                },
                'late_fusion': {
                    'enable': self.late_fusion_enabled,
                    'max_candidate_classes': self.late_fusion_candidate_count,
                    'weights': self.late_fusion_weights,
                },
                'hierarchy': {
                    'enable': self.hierarchy_enabled,
                    'parser': self.hierarchy_config.get('parser', 'auto'),
                    'delimiter': self.hierarchy_config.get('delimiter', '_'),
                    'max_depth': self.hierarchy_config.get('max_depth', 3),
                    'score_weight': self.hierarchy_score_weight,
                    'max_candidate_classes': self.hierarchy_candidate_count,
                    'min_group_size': self.hierarchy_min_group_size,
                    'max_score_margin': self.hierarchy_max_score_margin,
                    'min_gap': self.hierarchy_min_gap,
                    'level_weights': self.hierarchy_level_weights,
                },
                'robust_prototype': {
                    'enable': self.robust_enabled,
                    'deep_only': self.robust_deep_only,
                    'min_samples': self.robust_min_samples,
                    'trim_fraction': self.robust_trim_fraction,
                    'report_margin': self.robust_report_margin,
                    'top_outliers_per_class': self.robust_top_outliers,
                    'confusion_filter': self.robust_confusion_filter_enabled,
                    'confusion_filter_margin': self.robust_confusion_filter_margin,
                    'confusion_filter_max_fraction': self.robust_confusion_filter_max_fraction,
                    'confusion_filter_min_samples': self.robust_confusion_filter_min_samples,
                    'local_evidence_filter': self.local_evidence_filter_enabled,
                    'local_evidence_filter_margin': self.local_evidence_filter_margin,
                    'local_evidence_filter_max_fraction': self.local_evidence_filter_max_fraction,
                    'local_evidence_filter_min_samples': self.local_evidence_filter_min_samples,
                },
                'fast_search': {
                    'enable': self.fast_search_enabled,
                    'candidate_classes': self.fast_candidate_count,
                    'deep_only_local_scores': self.fast_deep_only_local_scores,
                    'skip_augmented_basic_features_when_deep_enabled': self.fast_skip_augmented_basic_features,
                },
                'concept_gate': {
                    'enable': self.concept_gate_enabled,
                    'min_top_gap': self.concept_gate_min_gap,
                    'weak_score_weight': self.concept_weak_score_weight,
                },
            },
            'deep_features': self.deep_adapter.metadata(),
            'deep_feature_config': dict(self.deep_adapter.config),
            'feature_groups': self.feature_groups,
            'system_features': self.system_feature_names,
            'feature_counts': self.feature_counts,
            'concept_counts': self.concept_counts,
            'prototype_stats': self.prototype_stats,
            'training_quality_report': self.training_quality_report,
            'training_confusion_pairs': {
                '|||'.join(key): int(value)
                for key, value in self.training_confusion_pairs.items()
            },
            'pairwise_similarities': {
                '|||'.join(key): float(value)
                for key, value in self.pairwise_similarities.items()
            },
            'sample_index': {
                label: [str(path) for path in paths]
                for label, paths in self.samples.items()
            },
            'sample_features': {
                label: [
                    {
                        'path': str(item.get('path')),
                        'features': self._export_feature_dict(item.get('features') or {}),
                        'crop_embeddings': self._export_crop_embeddings(item.get('crop_embeddings') or []),
                        'confusion_risk': bool(item.get('confusion_risk', False)),
                        'confusion_risk_reason': item.get('confusion_risk_reason'),
                        'confusion_margin': item.get('confusion_margin'),
                        'nearest_competitor': item.get('nearest_competitor'),
                    }
                    for item in rows
                ]
                for label, rows in self.sample_features.items()
            } if include_sample_features else {},
        }

