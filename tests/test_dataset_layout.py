from ask2know.data.dataset_loader import DatasetLoader
from ask2know.features.deep_adapter import DeepFeatureAdapter
from ask2know.sample_pool.manager import SamplePoolManager
from scripts.bootstrap_clusters import _build_review_rows, _extract_embeddings, _parse_skip_answer


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'fake image bytes')


def test_loader_separates_unknown_and_eval_samples(tmp_path):
    dataset = tmp_path / 'datasets'
    _touch(dataset / 'unknown' / 'mixed_a.jpg')
    _touch(dataset / 'unlabeled' / 'class_a' / 'eval_a.jpg')
    _touch(dataset / 'unlabeled' / 'class_b' / 'eval_b.jpg')

    loader = DatasetLoader(dataset)

    unknown = loader.load_unknown_samples()
    eval_samples = loader.load_eval_samples()

    assert len(unknown) == 1
    assert unknown[0]['label'] is None
    assert {item['label'] for item in eval_samples} == {'class_a', 'class_b'}


def test_copy_confirmed_keeps_unknown_source(tmp_path):
    dataset = tmp_path / 'datasets'
    output = tmp_path / 'outputs'
    src = dataset / 'unknown' / 'mixed_a.jpg'
    _touch(src)

    pool = SamplePoolManager(output_dir=output, dataset_dir=dataset)
    saved = pool.copy_confirmed(src, 'class_a')

    assert src.exists()
    assert (dataset / 'train' / 'class_a').exists()
    assert saved.endswith('.jpg')


def test_copy_confirmed_preserves_chinese_class_directory(tmp_path):
    dataset = tmp_path / 'datasets'
    output = tmp_path / 'outputs'
    src = dataset / 'unknown' / 'face.jpg'
    _touch(src)

    pool = SamplePoolManager(output_dir=output, dataset_dir=dataset)
    pool.copy_confirmed(src, '张三')

    assert (dataset / 'train' / '张三').exists()


def test_normalize_unknown_does_not_touch_eval_tree(tmp_path):
    dataset = tmp_path / 'datasets'
    output = tmp_path / 'outputs'
    _touch(dataset / 'unknown' / 'raw name.jpg')
    _touch(dataset / 'unlabeled' / 'class_a' / 'eval raw.jpg')

    pool = SamplePoolManager(output_dir=output, dataset_dir=dataset)
    changed = pool.normalize_unknown()

    assert len(changed) == 1
    assert (dataset / 'unknown' / 'unknown_001.jpg').exists()
    assert (dataset / 'unlabeled' / 'class_a' / 'eval raw.jpg').exists()


def test_deep_feature_cache_key_includes_model_config(tmp_path):
    img = tmp_path / 'sample.jpg'
    _touch(img)

    base = {
        'enable': True,
        'provider': 'open_clip',
        'feature_name': 'image_embedding',
    }
    adapter_a = DeepFeatureAdapter({**base, 'model_name': 'ViT-B-32', 'pretrained': 'a'})
    adapter_b = DeepFeatureAdapter({**base, 'model_name': 'ViT-L-14', 'pretrained': 'a'})
    adapter_c = DeepFeatureAdapter({**base, 'model_name': 'ViT-B-32', 'pretrained': 'b'})

    assert adapter_a._cache_key(img) != adapter_b._cache_key(img)
    assert adapter_a._cache_key(img) != adapter_c._cache_key(img)


def test_bootstrap_embedding_extraction_uses_path_cache(tmp_path):
    img = tmp_path / 'unknown' / 'sample.jpg'
    _touch(img)

    class FakeAdapter:
        feature_name = 'image_embedding'

        def __init__(self):
            self.path_calls = 0
            self.vector_calls = 0

        def extract_path(self, path):
            self.path_calls += 1
            return {self.feature_name: [1.0, 0.0, 0.0]}

        def extract_image_vector(self, img):
            self.vector_calls += 1
            return [0.0, 1.0, 0.0]

    adapter = FakeAdapter()
    paths, vectors = _extract_embeddings([{'path': str(img), 'label': None}], adapter)

    assert paths == [img]
    assert vectors.shape == (1, 3)
    assert adapter.path_calls == 1
    assert adapter.vector_calls == 0


def test_sample_pool_index_keeps_storage_and_display_names(tmp_path):
    dataset = tmp_path / 'datasets'
    output = tmp_path / 'outputs'
    src = dataset / 'unknown' / 'sample.jpg'
    _touch(src)

    pool = SamplePoolManager(output_dir=output, dataset_dir=dataset)
    saved = pool.copy_confirmed(src, 'red apple')
    index = pool._load_index()

    assert (dataset / 'train' / 'red_apple').exists()
    assert saved.endswith('red_apple_001.jpg')
    assert index['classes']['red_apple']['storage_name'] == 'red_apple'
    assert index['classes']['red_apple']['label'] == 'red apple'
    assert index['classes']['red_apple']['display_name'] == 'red apple'


def test_bootstrap_skip_answer_accepts_indexes_and_ranges():
    shown = [
        {'path': 'a.jpg'},
        {'path': 'b.jpg'},
        {'path': 'c.jpg'},
        {'path': 'd.jpg'},
    ]

    skipped, skip_all = _parse_skip_answer('1, 3-4', shown)

    assert skipped == {'a.jpg', 'c.jpg', 'd.jpg'}
    assert skip_all is False


def test_bootstrap_review_rows_skip_single_image(tmp_path):
    keep = tmp_path / 'unknown' / 'keep.jpg'
    skip = tmp_path / 'unknown' / 'skip.jpg'
    summaries = [{
        'cluster_id': 0,
        'mean_similarity': 0.8,
        'similarity_std': 0.1,
        'items': [
            {'path': str(keep), 'similarity': 0.92, 'review_candidate': False},
            {'path': str(skip), 'similarity': 0.61, 'review_candidate': True},
        ],
    }]

    rows = _build_review_rows(summaries, {0: 'red apple'}, skipped_paths={skip})

    assert [row['action'] for row in rows] == ['copy', 'skip']
    assert rows[0]['label'] == 'red_apple'
    assert rows[1]['review_candidate'] is True


def test_prototype_text_semantic_score_can_break_ties():
    import numpy as np
    from ask2know.inference.prototype_model import PrototypeModel

    model = PrototypeModel(
        ['image_embedding'],
        concept_config={'enable': False},
        similarity_config={'text_semantic': {'enable': True, 'score_weight': 0.20}},
        deep_feature_config={'enable': False},
    )
    same_proto = np.asarray([1.0, 0.0], dtype=np.float32)
    model.prototypes = {
        'cat': {'image_embedding': same_proto},
        'dog': {'image_embedding': same_proto},
    }
    model.text_prototypes = {
        'cat': np.asarray([1.0, 0.0], dtype=np.float32),
        'dog': np.asarray([0.0, 1.0], dtype=np.float32),
    }
    model._extract_primary_features = lambda path: {'image_embedding': same_proto}

    results = model.predict('sample.jpg', {'image_embedding': 1.0})

    assert results[0]['label'] == 'cat'
    assert results[0]['text_semantic_score'] > results[1]['text_semantic_score']


def test_prototype_subprototype_score_can_break_mean_ties():
    import numpy as np
    from ask2know.inference.prototype_model import PrototypeModel

    model = PrototypeModel(
        ['image_embedding'],
        concept_config={'enable': False},
        similarity_config={'sub_prototypes': {'enable': True, 'score_weight': 0.30}},
        deep_feature_config={'enable': False},
    )
    same_proto = np.asarray([0.7, 0.7], dtype=np.float32)
    model.prototypes = {
        'striped': {'image_embedding': same_proto},
        'plain': {'image_embedding': same_proto},
    }
    model.sub_prototypes = {
        'striped': [np.asarray([1.0, 0.0], dtype=np.float32)],
        'plain': [np.asarray([0.0, 1.0], dtype=np.float32)],
    }
    model._extract_primary_features = lambda path: {'image_embedding': np.asarray([1.0, 0.0], dtype=np.float32)}

    results = model.predict('sample.jpg', {'image_embedding': 1.0})

    assert results[0]['label'] == 'striped'
    assert results[0]['subprototype_score'] > results[1]['subprototype_score']


def test_subprototype_gate_blocks_flip_when_prototype_vetoes():
    import numpy as np
    from ask2know.inference.prototype_model import PrototypeModel

    model = PrototypeModel(
        ['image_embedding'],
        concept_config={'enable': False},
        similarity_config={
            'sub_prototypes': {
                'enable': True,
                'score_weight': 0.50,
                'min_gain_over_prototype': 0.001,
                'max_base_margin_for_flip': 0.010,
                'rank_flip_prototype_veto_margin': 0.003,
            }
        },
        deep_feature_config={'enable': False},
    )
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    model.prototypes = {
        'prototype_supported': {'image_embedding': np.asarray([0.99, 0.141067], dtype=np.float32)},
        'subprototype_supported': {'image_embedding': np.asarray([0.982, 0.188881], dtype=np.float32)},
    }
    model.sub_prototypes = {
        'subprototype_supported': [query],
    }
    model._extract_primary_features = lambda path: {'image_embedding': query}

    results = model.predict('sample.jpg', {'image_embedding': 1.0})
    by_label = {row['label']: row for row in results}

    assert results[0]['label'] == 'prototype_supported'
    assert by_label['subprototype_supported']['subprototype_gate_reason'] == 'prototype_veto'
    assert by_label['subprototype_supported']['subprototype_score_weight_used'] == 0.0


def test_subprototype_gate_allows_flip_when_base_margin_is_tiny():
    import numpy as np
    from ask2know.inference.prototype_model import PrototypeModel

    model = PrototypeModel(
        ['image_embedding'],
        concept_config={'enable': False},
        similarity_config={
            'sub_prototypes': {
                'enable': True,
                'score_weight': 0.50,
                'min_gain_over_prototype': 0.001,
                'max_base_margin_for_flip': 0.010,
                'rank_flip_prototype_veto_margin': 0.003,
            }
        },
        deep_feature_config={'enable': False},
    )
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    model.prototypes = {
        'barely_base_top': {'image_embedding': np.asarray([0.986, 0.166745], dtype=np.float32)},
        'local_match': {'image_embedding': np.asarray([0.982, 0.188881], dtype=np.float32)},
    }
    model.sub_prototypes = {
        'local_match': [query],
    }
    model._extract_primary_features = lambda path: {'image_embedding': query}

    results = model.predict('sample.jpg', {'image_embedding': 1.0})
    by_label = {row['label']: row for row in results}

    assert results[0]['label'] == 'local_match'
    assert by_label['local_match']['subprototype_gate_reason'] == 'rank_flip_allowed'
    assert by_label['local_match']['subprototype_score_weight_used'] == 0.50


def test_pairwise_rerank_uses_local_evidence_for_close_pair():
    import numpy as np
    from ask2know.inference.prototype_model import PrototypeModel

    model = PrototypeModel(
        ['image_embedding'],
        concept_config={'enable': False},
        similarity_config={
            'knn': {'enable': False},
            'sub_prototypes': {'enable': False},
            'pairwise_rerank': {
                'enable': True,
                'local_k': 1,
                'score_weight': 0.50,
                'max_score_margin': 0.020,
                'min_pair_similarity': 0.80,
                'min_local_gap': 0.005,
            },
        },
        deep_feature_config={'enable': False},
    )
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    model.prototypes = {
        'base_top': {'image_embedding': np.asarray([0.991, 0.133862], dtype=np.float32)},
        'local_match': {'image_embedding': np.asarray([0.985, 0.172554], dtype=np.float32)},
    }
    model.sample_features = {
        'base_top': [{'path': 'a.jpg', 'features': {'image_embedding': np.asarray([0.96, 0.28], dtype=np.float32)}}],
        'local_match': [{'path': 'b.jpg', 'features': {'image_embedding': query}}],
    }
    model._build_pairwise_similarities()
    model._extract_primary_features = lambda path: {'image_embedding': query}

    results = model.predict('sample.jpg', {'image_embedding': 1.0})
    by_label = {row['label']: row for row in results}

    assert results[0]['label'] == 'local_match'
    assert by_label['local_match']['pairwise_gate_reason'] == 'local_evidence'
    assert by_label['local_match']['pairwise_score_weight_used'] == 0.50


def test_pairwise_rerank_skips_large_score_margin():
    import numpy as np
    from ask2know.inference.prototype_model import PrototypeModel

    model = PrototypeModel(
        ['image_embedding'],
        concept_config={'enable': False},
        similarity_config={
            'knn': {'enable': False},
            'sub_prototypes': {'enable': False},
            'pairwise_rerank': {
                'enable': True,
                'local_k': 1,
                'score_weight': 0.50,
                'max_score_margin': 0.005,
                'min_pair_similarity': 0.80,
                'min_local_gap': 0.005,
            },
        },
        deep_feature_config={'enable': False},
    )
    query = np.asarray([1.0, 0.0], dtype=np.float32)
    model.prototypes = {
        'clear_top': {'image_embedding': query},
        'local_match': {'image_embedding': np.asarray([0.94, 0.341174], dtype=np.float32)},
    }
    model.sample_features = {
        'clear_top': [{'path': 'a.jpg', 'features': {'image_embedding': np.asarray([0.95, 0.31225], dtype=np.float32)}}],
        'local_match': [{'path': 'b.jpg', 'features': {'image_embedding': query}}],
    }
    model._build_pairwise_similarities()
    model._extract_primary_features = lambda path: {'image_embedding': query}

    results = model.predict('sample.jpg', {'image_embedding': 1.0})
    by_label = {row['label']: row for row in results}

    assert results[0]['label'] == 'clear_top'
    assert by_label['clear_top']['pairwise_gate_reason'] == 'score_margin_too_large'
    assert by_label['local_match']['pairwise_score_weight_used'] == 0.0


def test_robust_prototype_downweights_embedding_outliers():
    import numpy as np
    from ask2know.inference.prototype_model import PrototypeModel

    model = PrototypeModel(
        ['image_embedding'],
        concept_config={'enable': False},
        similarity_config={
            'knn': {'enable': False},
            'sub_prototypes': {'enable': False},
            'pairwise_rerank': {'enable': False},
            'robust_prototype': {
                'enable': True,
                'min_samples': 4,
                'trim_fraction': 0.25,
            },
        },
        deep_feature_config={'enable': False},
    )
    vectors = {
        'a': [1.0, 0.0],
        'b': [0.99, 0.01],
        'c': [0.98, 0.02],
        'd': [-1.0, 0.0],
    }
    model._feature_list_for_sample = lambda path: [{'image_embedding': np.asarray(vectors[str(path)], dtype=np.float32)}]

    model.fit([
        {'label': 'class_a', 'path': 'a'},
        {'label': 'class_a', 'path': 'b'},
        {'label': 'class_a', 'path': 'c'},
        {'label': 'class_a', 'path': 'd'},
    ])

    proto = model.prototypes['class_a']['image_embedding']

    assert proto[0] > 0.9
    assert model.prototype_stats['class_a']['image_embedding']['trimmed_count'] == 1
    assert model.training_quality_report['class_a']['outliers'][0]['path'] == 'd'


def test_prototype_concept_gate_ignores_weak_concept_gap():
    from ask2know.inference.prototype_model import PrototypeModel

    model = PrototypeModel(
        ['image_embedding'],
        concept_config={'enable': True, 'score_weight': 0.50},
        similarity_config={'concept_gate': {'enable': True, 'min_top_gap': 0.10, 'weak_score_weight': 0.0}},
        deep_feature_config={'enable': False},
    )
    model.prototypes = {
        'a': {'image_embedding': [1.0, 0.0]},
        'b': {'image_embedding': [0.9, 0.1]},
    }
    model.concept_prototypes = {
        'a': {'round': 0.50},
        'b': {'round': 0.54},
    }
    model._extract_primary_features = lambda path: {'image_embedding': [1.0, 0.0]}
    model._concepts_from_features = lambda feats: {'round': 0.52}

    results = model.predict('sample.jpg', {'image_embedding': 1.0})

    assert results[0]['label'] == 'a'
    assert results[0]['concept_score_weight_used'] == 0.0
    assert results[0]['concept_gate_reason'] == 'weak_gap'


def test_diagnose_prediction_reports_low_margin_and_sources():
    from ask2know.inference.diagnostics import diagnose_prediction

    results = [
        {'label': 'wrong', 'score': 0.901, 'prototype_score': 0.95, 'knn_score': 0.80},
        {'label': 'right', 'score': 0.895, 'prototype_score': 0.90, 'knn_score': 0.86},
    ]

    diagnosis = diagnose_prediction(results, true_label='right', low_margin_threshold=0.01, weak_signal_threshold=0.005)

    assert diagnosis['needs_review'] is True
    assert 'misclassified' in diagnosis['reason_codes']
    assert 'prototype_led' in diagnosis['reason_codes']
    assert 'true_label_supported_by_knn_score' in diagnosis['reason_codes']
