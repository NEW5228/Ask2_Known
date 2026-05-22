import argparse
import random
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
from ask2know.experience.confusion import PairVisualRuleMemory, OnlineConfusionExperience, build_confusion_experience_report
from ask2know.inference.diagnostics import diagnose_prediction
from ask2know.inference.prototype_model import PrototypeModel
from ask2know.learning.weights import AdaptiveWeights
from ask2know.utils.io_utils import ensure_dir, load_yaml, save_json

VERSION = '0.4.6.2b'


def class_names(objects):
    return [o['name'] for o in objects]


def rounded_prediction(item):
    return {
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
        'pairwise_score': None if item.get('pairwise_score') is None else round(float(item['pairwise_score']), 6),
        'pairwise_score_weight_used': round(float(item.get('pairwise_score_weight_used', 0.0)), 6),
        'pairwise_gate_reason': item.get('pairwise_gate_reason'),
        'pairwise_pair_similarity': None if item.get('pairwise_pair_similarity') is None else round(float(item['pairwise_pair_similarity']), 6),
        'pairwise_local_gap': None if item.get('pairwise_local_gap') is None else round(float(item['pairwise_local_gap']), 6),
        'crop_rerank_score': None if item.get('crop_rerank_score') is None else round(float(item['crop_rerank_score']), 6),
        'crop_rerank_score_weight_used': round(float(item.get('crop_rerank_score_weight_used', 0.0)), 6),
        'crop_rerank_gate_reason': item.get('crop_rerank_gate_reason'),
        'crop_rerank_pair_similarity': None if item.get('crop_rerank_pair_similarity') is None else round(float(item['crop_rerank_pair_similarity']), 6),
        'crop_rerank_local_gap': None if item.get('crop_rerank_local_gap') is None else round(float(item['crop_rerank_local_gap']), 6),
        'crop_rerank_crop_count': int(item.get('crop_rerank_crop_count', 0)),
        'concept_score': None if item.get('concept_score') is None else round(float(item['concept_score']), 6),
        'concept_score_weight_used': round(float(item.get('concept_score_weight_used', 0.0)), 6),
        'concept_gate_reason': item.get('concept_gate_reason'),
        'online_experience_delta': round(float(item.get('online_experience_delta', 0.0)), 6),
        'online_experience_gate_reason': item.get('online_experience_gate_reason'),
        'online_experience_evidence': item.get('online_experience_evidence', {}),
        'visual_rule_delta': round(float(item.get('visual_rule_delta', 0.0)), 6),
        'visual_rule_gate_reason': item.get('visual_rule_gate_reason'),
        'visual_rule_evidence': item.get('visual_rule_evidence', {}),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Ask2Know on datasets/unlabeled/<class_name>/ images.'
    )
    parser.add_argument('--config', required=True, help='Project task_config.yaml')
    parser.add_argument('--top-k', type=int, default=3, help='Number of predictions to keep per sample.')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle evaluation samples before streaming evaluation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used with --shuffle.')
    parser.add_argument('--online-experience', action='store_true', help='Learn from earlier revealed mistakes and rerank later low-margin confusion pairs.')
    parser.add_argument('--online-max-margin', type=float, default=0.035, help='Only apply online experience when top-2 score margin is at most this value.')
    parser.add_argument('--online-adjustment-weight', type=float, default=0.02, help='Weight for each learned pair/source signal.')
    parser.add_argument('--online-max-adjustment', type=float, default=0.04, help='Maximum score delta from online experience per candidate.')
    parser.add_argument('--online-min-observations', type=int, default=1, help='Minimum earlier pair errors before online experience can apply.')
    parser.add_argument('--disable-visual-rules', action='store_true', help='Disable pair-specific visual rule memory during online evaluation.')
    parser.add_argument('--visual-rule-weight', type=float, default=0.035, help='Weight for pair-specific concept visual rules.')
    parser.add_argument('--visual-rule-max-margin', type=float, default=0.04, help='Only apply visual rules when top-2 margin is at most this value.')
    parser.add_argument('--visual-rule-max-adjustment', type=float, default=0.05, help='Maximum score delta from visual rules per candidate.')
    parser.add_argument('--visual-rule-min-concept-gap', type=float, default=0.10, help='Minimum class concept gap required to learn a visual rule.')
    parser.add_argument('--visual-rule-min-match-gap', type=float, default=0.04, help='Minimum sample-to-class match gap required to apply a visual rule.')
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
    if args.shuffle:
        rng = random.Random(args.seed)
        eval_samples = list(eval_samples)
        rng.shuffle(eval_samples)

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

    online_memory = None
    visual_rule_memory = None
    if args.online_experience:
        online_memory = OnlineConfusionExperience(
            weak_signal_threshold=weak_signal_threshold,
            max_margin=args.online_max_margin,
            adjustment_weight=args.online_adjustment_weight,
            max_adjustment=args.online_max_adjustment,
            min_observations=args.online_min_observations,
        )
        if not args.disable_visual_rules:
            visual_rule_memory = PairVisualRuleMemory(
                weak_signal_threshold=weak_signal_threshold,
                max_margin=args.visual_rule_max_margin,
                rule_weight=args.visual_rule_weight,
                max_adjustment=args.visual_rule_max_adjustment,
                min_observations=args.online_min_observations,
                min_concept_gap=args.visual_rule_min_concept_gap,
                min_match_gap=args.visual_rule_min_match_gap,
            )

    rows = []
    confusion = defaultdict(lambda: defaultdict(int))
    per_class = defaultdict(lambda: {'total': 0, 'correct': 0})
    online_summary = {
        'raw_correct_count': 0,
        'online_correct_count': 0,
        'changed_top1_count': 0,
        'helped_count': 0,
        'hurt_count': 0,
        'applied_count': 0,
        'source_changed_top1_count': 0,
        'visual_rule_applied_count': 0,
        'visual_rule_changed_top1_count': 0,
        'visual_rule_helped_count': 0,
        'visual_rule_hurt_count': 0,
    }

    for idx, sample in enumerate(eval_samples):
        raw_results = model.predict(sample['path'], weights)
        raw_predicted = raw_results[0]['label'] if raw_results else None
        raw_correct = raw_predicted == sample['label']
        results = raw_results
        online_adjustment = {'applied': False, 'changed_top1': False, 'reason': 'disabled', 'deltas': {}}
        visual_rule_adjustment = {'applied': False, 'changed_top1': False, 'reason': 'disabled', 'deltas': {}}
        if online_memory is not None:
            results, online_adjustment = online_memory.apply(raw_results)
        after_source_predicted = results[0]['label'] if results else None
        if visual_rule_memory is not None:
            results, visual_rule_adjustment = visual_rule_memory.apply(results)
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
        after_source_correct = after_source_predicted == true_label
        final_changed = predicted != raw_predicted
        if raw_correct:
            online_summary['raw_correct_count'] += 1
        if correct:
            online_summary['online_correct_count'] += 1
        if final_changed:
            online_summary['changed_top1_count'] += 1
            if correct and not raw_correct:
                online_summary['helped_count'] += 1
            elif raw_correct and not correct:
                online_summary['hurt_count'] += 1
        if online_adjustment.get('applied'):
            online_summary['applied_count'] += 1
        if online_adjustment.get('changed_top1'):
            online_summary['source_changed_top1_count'] += 1
        if visual_rule_adjustment.get('applied'):
            online_summary['visual_rule_applied_count'] += 1
        if visual_rule_adjustment.get('changed_top1'):
            online_summary['visual_rule_changed_top1_count'] += 1
            if correct and not after_source_correct:
                online_summary['visual_rule_helped_count'] += 1
            elif after_source_correct and not correct:
                online_summary['visual_rule_hurt_count'] += 1

        row = {
            'stream_index': idx,
            'path': sample['path'],
            'true_label': true_label,
            'raw_predicted_label': raw_predicted,
            'after_source_predicted_label': after_source_predicted,
            'predicted_label': predicted,
            'raw_correct': raw_correct,
            'after_source_correct': after_source_correct,
            'correct': correct,
            'online_adjustment': online_adjustment,
            'visual_rule_adjustment': visual_rule_adjustment,
            'top_predictions': [rounded_prediction(item) for item in results[:max(1, args.top_k)]],
            'raw_top_predictions': [rounded_prediction(item) for item in raw_results[:max(1, args.top_k)]],
            'diagnosis': diagnosis,
        }
        rows.append(row)
        learn_row = dict(row)
        learn_row['top_predictions'] = results[:max(1, args.top_k)]
        if online_memory is not None:
            online_memory.record_outcome(raw_correct, correct, bool(final_changed))
            online_memory.observe(learn_row)
        if visual_rule_memory is not None:
            visual_rule_memory.record_outcome(after_source_correct, correct, bool(visual_rule_adjustment.get('changed_top1')))
            visual_rule_memory.observe(learn_row)

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
    confusion_experience = build_confusion_experience_report(
        rows,
        model=model,
        weak_signal_threshold=weak_signal_threshold,
    )
    report = {
        'schema_version': VERSION,
        'dataset_dir': str(dataset_dir),
        'eval_dir': str(Path(dataset_dir) / 'unlabeled'),
        'train_sample_count': len(train_samples),
        'eval_sample_count': total,
        'accuracy': correct_count / max(1, total),
        'correct_count': correct_count,
        'evaluation_mode': {
            'shuffle': bool(args.shuffle),
            'seed': args.seed if args.shuffle else None,
            'online_experience': bool(args.online_experience),
            'visual_rules': visual_rule_memory is not None,
        },
        'online_experience_summary': online_summary,
        'online_experience': None if online_memory is None else online_memory.export(),
        'visual_rule_memory': None if visual_rule_memory is None else visual_rule_memory.export(),
        'diagnostics': {
            'low_margin_threshold': low_margin_threshold,
            'weak_signal_threshold': weak_signal_threshold,
            'needs_review_count': review_count,
            'reason_counts': dict(sorted(reason_counts.items())),
        },
        'training_quality_report': model.training_quality_report,
        'confusion_experience_report': confusion_experience,
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
    confusion_report_path = output_dir / 'confusion_experience_report.json'
    save_json(report_path, report)
    save_json(confusion_report_path, confusion_experience)
    if online_memory is not None:
        save_json(output_dir / 'online_experience_report.json', online_memory.export())
    if visual_rule_memory is not None:
        save_json(output_dir / 'visual_rule_memory_report.json', visual_rule_memory.export())

    print(f'Evaluated {total} samples from {Path(dataset_dir) / "unlabeled"}')
    print(f'Accuracy: {correct_count}/{total} = {report["accuracy"]:.3f}')
    if args.online_experience:
        raw_accuracy = online_summary['raw_correct_count'] / max(1, total)
        print(f'Raw accuracy before online experience: {online_summary["raw_correct_count"]}/{total} = {raw_accuracy:.3f}')
        print(
            'Online changes: '
            f'applied={online_summary["applied_count"]}, '
            f'changed_top1={online_summary["changed_top1_count"]}, '
            f'helped={online_summary["helped_count"]}, '
            f'hurt={online_summary["hurt_count"]}'
        )
        if visual_rule_memory is not None:
            print(
                'Visual rule changes: '
                f'applied={online_summary["visual_rule_applied_count"]}, '
                f'changed_top1={online_summary["visual_rule_changed_top1_count"]}, '
                f'helped={online_summary["visual_rule_helped_count"]}, '
                f'hurt={online_summary["visual_rule_hurt_count"]}'
            )
    for label, item in report['per_class'].items():
        print(f'{label}: {item["correct"]}/{item["total"]} = {item["accuracy"]:.3f}')
    print('Report:', report_path)
    print('Confusion experience:', confusion_report_path)
    if online_memory is not None:
        print('Online experience:', output_dir / 'online_experience_report.json')
    if visual_rule_memory is not None:
        print('Visual rule memory:', output_dir / 'visual_rule_memory_report.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
