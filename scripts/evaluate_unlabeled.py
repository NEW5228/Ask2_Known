import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ask2know.data.dataset_loader import DatasetLoader
from ask2know.features.feature_config import (
    initial_feature_weights,
    parse_feature_config,
    resolve_deep_feature_config,
)
from ask2know.inference.diagnostics import diagnose_prediction
from ask2know.inference.prototype_model import PrototypeModel
from ask2know.learning.weights import AdaptiveWeights
from ask2know.utils.io_utils import ensure_dir, load_yaml, save_json

VERSION = '0.4.4.1'


def class_names(objects):
    return [o['name'] for o in objects]


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Ask2Know on datasets/unlabeled/<class_name>/ images.'
    )
    parser.add_argument('--config', required=True, help='Project task_config.yaml')
    parser.add_argument('--top-k', type=int, default=3, help='Number of predictions to keep per sample.')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_dir = cfg['paths']['dataset_dir']
    output_dir = Path(cfg['paths']['output_dir'])
    ensure_dir(output_dir)
    ensure_dir(output_dir / 'logs')

    loader = DatasetLoader(dataset_dir)
    objects = loader.load_objects()
    labels = class_names(objects)
    train_samples = loader.load_train_samples()
    eval_samples = loader.load_eval_samples()

    if not labels:
        print('No classes found. Create train folders or objects.json first.')
        return 1
    if not train_samples:
        print(f'No train samples found in {Path(dataset_dir) / "train"}')
        return 1
    if not eval_samples:
        print(f'No evaluation samples found in {Path(dataset_dir) / "unlabeled" / "class_name"}')
        return 1

    deep_feature_config = resolve_deep_feature_config(cfg)
    feature_spec = parse_feature_config(cfg, classes=labels or cfg.get('classes', []))
    diagnostic_cfg = cfg.get('diagnostics', {})
    low_margin_threshold = float(diagnostic_cfg.get('low_margin_threshold', 0.015))
    weak_signal_threshold = float(diagnostic_cfg.get('weak_signal_threshold', 0.005))
    weights = AdaptiveWeights(
        initial_feature_weights(cfg, feature_spec),
        cfg['learning'].get('update_step', 0.07),
        cfg['learning'].get('min_weight', 0.05),
        cfg['learning'].get('max_weight', 0.70),
    ).export()

    model = PrototypeModel(
        feature_spec['scoring_features'],
        augmentation_config=cfg.get('augmentation', {}),
        concept_config=cfg.get('concepts', {'enable': True, 'score_weight': 0.25}),
        system_feature_names=feature_spec['system_features'],
        feature_groups=feature_spec['group_features'],
        similarity_config=cfg.get('similarity', {}),
        deep_feature_config=deep_feature_config,
        deep_cache_dir=output_dir / '.cache' / 'deep_features',
    ).fit(train_samples)

    rows = []
    confusion = defaultdict(lambda: defaultdict(int))
    per_class = defaultdict(lambda: {'total': 0, 'correct': 0})

    for sample in eval_samples:
        results = model.predict(sample['path'], weights)
        predicted = results[0]['label'] if results else None
        true_label = sample['label']
        correct = predicted == true_label
        diagnosis = diagnose_prediction(
            results[:max(2, args.top_k)],
            true_label=true_label,
            low_margin_threshold=low_margin_threshold,
            weak_signal_threshold=weak_signal_threshold,
        )
        per_class[true_label]['total'] += 1
        per_class[true_label]['correct'] += 1 if correct else 0
        confusion[true_label][predicted or 'none'] += 1
        rows.append({
            'path': sample['path'],
            'true_label': true_label,
            'predicted_label': predicted,
            'correct': correct,
            'top_predictions': [
                {
                    'label': item['label'],
                    'score': round(float(item['score']), 6),
                    'base_score': round(float(item.get('base_score', item['score'])), 6),
                    'prototype_score': round(float(item['prototype_score']), 6),
                    'subprototype_score': None if item.get('subprototype_score') is None else round(float(item['subprototype_score']), 6),
                    'subprototype_score_weight_used': round(float(item.get('subprototype_score_weight_used', 0.0)), 6),
                    'subprototype_gate_reason': item.get('subprototype_gate_reason'),
                    'subprototype_gain_over_prototype': None if item.get('subprototype_gain_over_prototype') is None else round(float(item['subprototype_gain_over_prototype']), 6),
                    'subprototype_top_gap': None if item.get('subprototype_top_gap') is None else round(float(item['subprototype_top_gap']), 6),
                    'knn_score': None if item.get('knn_score') is None else round(float(item['knn_score']), 6),
                    'text_semantic_score': None if item.get('text_semantic_score') is None else round(float(item['text_semantic_score']), 6),
                    'concept_score': None if item.get('concept_score') is None else round(float(item['concept_score']), 6),
                    'concept_score_weight_used': round(float(item.get('concept_score_weight_used', 0.0)), 6),
                    'concept_gate_reason': item.get('concept_gate_reason'),
                }
                for item in results[:max(1, args.top_k)]
            ],
            'diagnosis': diagnosis,
        })

    total = len(rows)
    correct_count = sum(1 for row in rows if row['correct'])
    reason_counts = defaultdict(int)
    review_count = 0
    for row in rows:
        diagnosis = row.get('diagnosis') or {}
        if diagnosis.get('needs_review'):
            review_count += 1
        for reason in diagnosis.get('reason_codes') or []:
            reason_counts[reason] += 1
    report = {
        'schema_version': VERSION,
        'dataset_dir': str(dataset_dir),
        'eval_dir': str(Path(dataset_dir) / 'unlabeled'),
        'train_sample_count': len(train_samples),
        'eval_sample_count': total,
        'accuracy': correct_count / max(1, total),
        'correct_count': correct_count,
        'diagnostics': {
            'low_margin_threshold': low_margin_threshold,
            'weak_signal_threshold': weak_signal_threshold,
            'needs_review_count': review_count,
            'reason_counts': dict(sorted(reason_counts.items())),
        },
        'per_class': {
            label: {
                'total': item['total'],
                'correct': item['correct'],
                'accuracy': item['correct'] / max(1, item['total']),
            }
            for label, item in sorted(per_class.items())
        },
        'confusion': {
            true_label: dict(preds)
            for true_label, preds in sorted(confusion.items())
        },
        'errors': [row for row in rows if not row['correct']],
        'samples': rows,
    }

    report_path = output_dir / 'evaluation_report.json'
    save_json(report_path, report)

    print(f'Evaluated {total} samples from {Path(dataset_dir) / "unlabeled"}')
    print(f'Accuracy: {correct_count}/{total} = {report["accuracy"]:.3f}')
    for label, item in report['per_class'].items():
        print(f'{label}: {item["correct"]}/{item["total"]} = {item["accuracy"]:.3f}')
    print('Report:', report_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
