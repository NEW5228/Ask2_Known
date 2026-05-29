import numpy as np

from ask2know.features.deep_adapter import _crop_image_specs
from ask2know.inference.prototype_model import PrototypeModel


def test_semantic_crop_specs_include_object_and_head_boxes():
    img = np.full((100, 120, 3), 245, dtype=np.uint8)
    img[25:85, 35:95] = np.asarray([30, 60, 180], dtype=np.uint8)

    specs = _crop_image_specs(img, crop_names=['object', 'head'])
    by_id = {item['crop_id']: item for item in specs}

    assert {'object', 'head'} <= set(by_id)
    object_box = by_id['object']['box']
    head_box = by_id['head']['box']
    assert object_box[0] <= 35
    assert object_box[2] >= 95
    assert head_box[1] <= object_box[1]
    assert head_box[3] <= object_box[3]


def test_crop_rerank_uses_local_crop_evidence_for_low_margin_pair(monkeypatch):
    model = PrototypeModel(
        feature_names=['image_embedding'],
        similarity_config={
            'pairwise_rerank': {'enable': False},
            'crop_rerank': {
                'enable': True,
                'max_candidate_classes': 2,
                'score_weight': 0.50,
                'max_score_margin': 0.05,
                'min_local_gap': 0.001,
                'trigger_mode': 'margin_only',
            },
        },
        deep_feature_config={
            'enable': True,
            'provider': 'open_clip',
            'feature_name': 'image_embedding',
            'multi_crop': {'enable': True},
        },
    )
    model.sample_features = {
        'apple': [
            {
                'path': 'apple.jpg',
                'features': {},
                'crop_embeddings': [
                    {'crop_id': 'center', 'vector': np.asarray([1.0, 0.0], dtype=np.float32)}
                ],
            }
        ],
        'banana': [
            {
                'path': 'banana.jpg',
                'features': {},
                'crop_embeddings': [
                    {'crop_id': 'center', 'vector': np.asarray([0.0, 1.0], dtype=np.float32)}
                ],
            }
        ],
    }
    monkeypatch.setattr(
        model,
        '_extract_crop_embeddings_for_path',
        lambda path: [{'crop_id': 'center', 'vector': np.asarray([1.0, 0.0], dtype=np.float32)}],
    )
    rows = [
        {'label': 'banana', 'score': 0.900},
        {'label': 'apple', 'score': 0.895},
    ]

    model._apply_crop_rerank(rows, 'query.jpg')
    rows.sort(key=lambda row: row['score'], reverse=True)

    assert rows[0]['label'] == 'apple'
    assert rows[0]['crop_rerank_gate_reason'] == 'crop_local_evidence'
    assert rows[0]['crop_rerank_score_weight_used'] == 0.50


def test_late_fusion_reranks_top_candidates_from_configured_sources():
    model = PrototypeModel(
        feature_names=['image_embedding'],
        similarity_config={
            'late_fusion': {
                'enable': True,
                'max_candidate_classes': 3,
                'weights': {
                    'base_score': 1.0,
                    'knn_score': 1.0,
                },
            },
        },
    )
    rows = [
        {'label': 'banana', 'score': 0.91, 'base_score': 0.91, 'knn_score': 0.80},
        {'label': 'apple', 'score': 0.90, 'base_score': 0.90, 'knn_score': 0.95},
        {'label': 'pear', 'score': 0.70, 'base_score': 0.70, 'knn_score': 0.72},
        {'label': 'orange', 'score': 0.69, 'base_score': 0.69, 'knn_score': 0.99},
    ]

    model._apply_late_fusion_rerank(rows)
    rows.sort(key=lambda row: row['score'], reverse=True)

    assert rows[0]['label'] == 'apple'
    assert rows[0]['late_fusion_gate_reason'] == 'applied'
    assert rows[-1]['label'] == 'orange'
    assert rows[-1]['late_fusion_score'] is None
    assert rows[-1]['score'] == 0.69
