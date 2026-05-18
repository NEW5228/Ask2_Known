from ask2know.data.dataset_loader import DatasetLoader
from ask2know.sample_pool.manager import SamplePoolManager


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
