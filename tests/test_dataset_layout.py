from ask2know.data.dataset_loader import DatasetLoader
from ask2know.features.deep_adapter import DeepFeatureAdapter
from ask2know.sample_pool.manager import SamplePoolManager
from scripts.bootstrap_clusters import _extract_embeddings


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
