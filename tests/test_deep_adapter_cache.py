import numpy as np

from ask2know.features.deep_adapter import DeepFeatureAdapter


class FakeTextAdapter(DeepFeatureAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def _open_clip_text_embeddings(self, texts):
        self.calls += 1
        vectors = []
        for idx, _ in enumerate(texts or []):
            vec = np.zeros(4, dtype=np.float32)
            vec[idx % 4] = 1.0
            vectors.append(vec)
        return vectors


def test_text_vectors_use_file_cache(tmp_path):
    adapter = FakeTextAdapter(
        {
            'enable': True,
            'provider': 'open_clip',
            'feature_name': 'image_embedding',
            'cache': True,
            'model_name': 'fake',
            'pretrained': 'fake',
        },
        cache_dir=tmp_path,
    )

    first = adapter.extract_text_vectors(['speed limit 30'])
    second = adapter.extract_text_vectors(['speed limit 30'])

    assert adapter.calls == 1
    assert np.allclose(first[0], second[0])
    assert list(tmp_path.glob('*.text.json'))
