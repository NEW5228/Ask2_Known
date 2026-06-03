from collections import defaultdict
from pathlib import Path

from ask2know.inference.diagnostics import diagnose_prediction
from ask2know.utils.io_utils import save_json


def _rounded_prediction(item):
    return {
        'label': item.get('label'),
        'score': round(float(item.get('score', 0.0)), 6),
        'base_score': round(float(item.get('base_score', item.get('score', 0.0))), 6),
        'prototype_score': _round_optional(item.get('prototype_score')),
        'subprototype_score': _round_optional(item.get('subprototype_score')),
        'knn_score': _round_optional(item.get('knn_score')),
        'text_semantic_score': _round_optional(item.get('text_semantic_score')),
        'pairwise_score': _round_optional(item.get('pairwise_score')),
        'crop_rerank_score': _round_optional(item.get('crop_rerank_score')),
        'late_fusion_score': _round_optional(item.get('late_fusion_score')),
        'concept_score': _round_optional(item.get('concept_score')),
    }


def _round_optional(value):
    if value is None:
        return None
    return round(float(value), 6)


def evaluate_labeled_samples(
    model,
    weights,
    samples,
    *,
    dataset_dir=None,
    eval_dir=None,
    top_k=3,
    low_margin_threshold=0.015,
    weak_signal_threshold=0.005,
    schema_version='validation_report_v1',
):
    rows = []
    confusion = defaultdict(lambda: defaultdict(int))
    per_class = defaultdict(lambda: {'total': 0, 'correct': 0})
    reason_counts = defaultdict(int)
    review_count = 0

    for idx, sample in enumerate(samples or []):
        results = model.predict(sample['path'], weights)
        predicted = results[0]['label'] if results else None
        true_label = sample.get('label')
        correct = predicted == true_label
        diagnosis = diagnose_prediction(
            results[:max(2, int(top_k))],
            true_label=true_label,
            low_margin_threshold=low_margin_threshold,
            weak_signal_threshold=weak_signal_threshold,
        )
        if diagnosis.get('needs_review'):
            review_count += 1
        for reason in diagnosis.get('reason_codes') or []:
            reason_counts[reason] += 1

        per_class[true_label]['total'] += 1
        per_class[true_label]['correct'] += 1 if correct else 0
        confusion[true_label][predicted or 'none'] += 1
        rows.append({
            'index': idx,
            'path': sample['path'],
            'true_label': true_label,
            'predicted_label': predicted,
            'correct': correct,
            'top_predictions': [_rounded_prediction(item) for item in results[:max(1, int(top_k))]],
            'diagnosis': diagnosis,
        })

    total = len(rows)
    correct_count = sum(1 for row in rows if row['correct'])
    return {
        'schema_version': schema_version,
        'dataset_dir': None if dataset_dir is None else str(dataset_dir),
        'eval_dir': None if eval_dir is None else str(eval_dir),
        'eval_sample_count': total,
        'accuracy': correct_count / max(1, total),
        'correct_count': correct_count,
        'diagnostics': {
            'low_margin_threshold': float(low_margin_threshold),
            'weak_signal_threshold': float(weak_signal_threshold),
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


def evaluate_unknown_audit_logs(cfg, logs, output_dir):
    pass_threshold = float(cfg.get('validation', {}).get('pass_accuracy_threshold', 0.85))
    rows = []
    per_class = defaultdict(lambda: {'total': 0, 'correct': 0})
    confusion = defaultdict(lambda: defaultdict(int))

    for idx, item in enumerate(logs or []):
        pool = item.get('pool') or {}
        true_label = pool.get('label')
        if pool.get('decision') != 'confirmed' or not true_label:
            continue
        predictions = item.get('after') or item.get('before') or []
        if not predictions:
            continue
        top = predictions[0]
        predicted = top.get('label')
        correct = predicted == true_label
        per_class[true_label]['total'] += 1
        per_class[true_label]['correct'] += 1 if correct else 0
        confusion[true_label][predicted or 'none'] += 1
        rows.append({
            'index': idx,
            'path': item.get('sample'),
            'true_label': true_label,
            'predicted_label': predicted,
            'correct': correct,
            'top_predictions': [_rounded_prediction(row) for row in predictions[:5]],
        })

    total = len(rows)
    correct_count = sum(1 for row in rows if row['correct'])
    accuracy = correct_count / max(1, total)
    report = {
        'schema_version': 'unknown_audit_validation_v1',
        'source': 'confirmed_unknown_logs',
        'eval_sample_count': total,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'validation_standard': {
            'pass_accuracy_threshold': pass_threshold,
            'passed': total > 0 and accuracy >= pass_threshold,
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
    output_dir = Path(output_dir)
    save_json(output_dir / 'unknown_validation_report.json', report)
    save_json(output_dir / 'validation_status.json', {
        'schema_version': 'validation_status_v1',
        'source': 'unknown_audit',
        'passed': report['validation_standard']['passed'],
    })
    return report
