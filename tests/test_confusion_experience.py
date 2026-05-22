from ask2know.experience.confusion import build_confusion_experience_report
from ask2know.experience.pairwise import PairwiseExperienceManager


class DummyModel:
    def pair_discriminative_summary(self, label_a, label_b, top_n=5):
        return {
            'labels': [label_a, label_b],
            'top_group_differences': [
                {'group': 'shape', 'prototype_similarity': 0.70, 'discriminative_gap': 0.30}
            ],
            'weak_group_differences': [],
            'top_concept_differences': [
                {'concept': 'top_ear_like', 'a_score': 0.8, 'b_score': 0.2, 'stronger_label': label_a, 'gap': 0.6}
            ],
        }


def test_confusion_experience_report_summarizes_error_sources():
    rows = [
        {
            'path': 'sample.jpg',
            'true_label': 'class_a',
            'predicted_label': 'class_b',
            'correct': False,
            'top_predictions': [
                {'label': 'class_b', 'score': 0.91},
                {'label': 'class_a', 'score': 0.89},
            ],
            'diagnosis': {
                'score_margin': 0.02,
                'reason_codes': ['misclassified', 'wrong_label_supported_by_text_semantic_score'],
                'true_label_compare': {
                    'source_deltas_top_minus_true': {
                        'text_semantic_score': 0.04,
                        'knn_score': -0.03,
                    }
                },
            },
        }
    ]

    report = build_confusion_experience_report(rows, model=DummyModel())
    pair = report['pairs'][0]

    assert report['summary']['error_count'] == 1
    assert pair['true_label'] == 'class_a'
    assert pair['predicted_label'] == 'class_b'
    assert pair['true_in_top_k_count'] == 1
    assert pair['wrong_supported_sources']['text_semantic_score'] == 1
    assert pair['true_supported_sources']['knn_score'] == 1
    assert pair['model_contrast_summary']['top_group_differences'][0]['group'] == 'shape'
    assert pair['recommendations']


def test_pairwise_user_note_becomes_question_hint(tmp_path):
    manager = PairwiseExperienceManager(metadata_dir=tmp_path, version='0.4.6.2a')

    manager.record_corrections(
        'class_b',
        'class_a',
        [('shape', 'shape differs')],
        free_note='class_a has longer ears than class_b',
    )

    hint = manager.pair_specific_question_hint('class_a', 'class_b')
    prompt = manager.suggest_pair_prompt('class_a', 'class_b')

    assert 'longer ears' in hint
    assert 'longer ears' in prompt


def test_online_confusion_experience_learns_from_earlier_error():
    from ask2know.experience.confusion import OnlineConfusionExperience

    memory = OnlineConfusionExperience(
        weak_signal_threshold=0.001,
        max_margin=0.05,
        adjustment_weight=2.0,
        max_adjustment=0.05,
        min_observations=1,
    )
    first_error = {
        'path': 'first.jpg',
        'true_label': 'class_a',
        'predicted_label': 'class_b',
        'correct': False,
        'top_predictions': [
            {'label': 'class_b', 'score': 0.51, 'knn_score': 0.30, 'prototype_score': 0.60},
            {'label': 'class_a', 'score': 0.49, 'knn_score': 0.34, 'prototype_score': 0.58},
        ],
        'diagnosis': {'score_margin': 0.02},
    }
    memory.observe(first_error)

    results, adjustment = memory.apply([
        {'label': 'class_b', 'score': 0.51, 'knn_score': 0.30, 'prototype_score': 0.60},
        {'label': 'class_a', 'score': 0.49, 'knn_score': 0.34, 'prototype_score': 0.58},
    ])

    assert adjustment['applied'] is True
    assert adjustment['changed_top1'] is True
    assert results[0]['label'] == 'class_a'
    assert memory.export()['stats']['observed_errors'] == 1
