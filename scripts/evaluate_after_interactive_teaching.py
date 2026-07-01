import argparse
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ask2know.data.dataset_loader import DatasetLoader
from ask2know.experience.confusion import OnlineConfusionExperience, PairVisualRuleMemory
from ask2know.features.feature_config import (
    initial_feature_weights,
    parse_feature_config,
    resolve_deep_feature_config,
)
from ask2know.inference.prototype_model import PrototypeModel
from ask2know.learning.weights import AdaptiveWeights
from ask2know.questions.ask_resolver import (
    DEFAULT_ASK_CANDIDATE_TOP_K,
    DEFAULT_ASK_MAX_OPTIONS,
    DEFAULT_ASK_MAX_QUESTIONS,
    simulate_dynamic_dialog,
)
from ask2know.utils.io_utils import ensure_dir, load_json, load_yaml, save_json

VERSION = '0.5.0'
PAIR_FEATURE_NAMES = ('color', 'shape', 'texture', 'surface', 'part', 'size', 'quality')


def _class_names(objects):
    return [item.get('name') for item in objects if item.get('name')]


def _load_or_fit_model(cfg, train_samples, feature_spec, deep_cache_dir, model_cache_path, rebuild=False):
    deep_feature_config = resolve_deep_feature_config(cfg)
    if model_cache_path and model_cache_path.exists() and not rebuild:
        cached = load_json(model_cache_path)
        model_data = cached.get('model') if isinstance(cached, dict) and 'model' in cached else cached
        return PrototypeModel.from_export(
            model_data,
            deep_feature_config=deep_feature_config,
            deep_cache_dir=deep_cache_dir,
        )
    model = PrototypeModel(
        feature_spec['scoring_features'],
        augmentation_config=cfg.get('augmentation', {}),
        concept_config=cfg.get('concepts', {'enable': True, 'score_weight': 0.25}),
        system_feature_names=feature_spec['system_features'],
        feature_groups=feature_spec['group_features'],
        similarity_config=cfg.get('similarity', {}),
        deep_feature_config=deep_feature_config,
        deep_cache_dir=deep_cache_dir,
    ).fit(train_samples)
    return model


def _stratified_split(samples, teach_per_class, seed):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample['label']].append(sample)
    rng = random.Random(seed)
    teaching = []
    heldout = []
    for label in sorted(grouped):
        rows = list(grouped[label])
        rng.shuffle(rows)
        n = min(max(0, int(teach_per_class)), len(rows))
        teaching.extend(rows[:n])
        heldout.extend(rows[n:])
    rng.shuffle(teaching)
    rng.shuffle(heldout)
    return teaching, heldout


def _pair_key(label_a, label_b):
    labels = sorted([str(label_a), str(label_b)])
    return f'{labels[0]} <-> {labels[1]}'


def _feature_value(row, feature):
    for bucket in ('group_detail', 'system_detail', 'detail'):
        data = row.get(bucket) or {}
        if feature in data:
            try:
                return float(data[feature])
            except (TypeError, ValueError):
                return None
    return None


def _best_pair_feature(true_item, wrong_item, *, min_gap=0.01):
    best = None
    for feature in PAIR_FEATURE_NAMES:
        true_value = _feature_value(true_item, feature)
        wrong_value = _feature_value(wrong_item, feature)
        if true_value is None or wrong_value is None:
            continue
        gap = true_value - wrong_value
        if gap <= min_gap:
            continue
        if best is None or gap > best['gap']:
            best = {
                'feature': feature,
                'true_value': true_value,
                'wrong_value': wrong_value,
                'gap': gap,
            }
    return best


class PairFeatureMemory:
    """Pair-local feature hints learned from dynamic corrections."""

    def __init__(
        self,
        feature_bonus=0.02,
        max_adjustment=0.04,
        max_margin=0.035,
        min_observations=1,
        min_feature_gap=0.01,
        min_sources_for_flip=1,
        max_examples=8,
    ):
        self.feature_bonus = float(feature_bonus)
        self.max_adjustment = float(max_adjustment)
        self.max_margin = float(max_margin)
        self.min_observations = int(min_observations)
        self.min_feature_gap = float(min_feature_gap)
        self.min_sources_for_flip = max(1, int(min_sources_for_flip))
        self.max_examples = int(max_examples)
        self.pairs = {}
        self.stats = {
            'observed': 0,
            'observed_pairs': 0,
            'feature_votes': 0,
            'applied': 0,
            'changed_top1': 0,
        }

    def observe(self, sample):
        true_label = sample.get('true_label')
        predicted = sample.get('predicted_label')
        if not true_label or not predicted or true_label == predicted:
            return False
        top_predictions = sample.get('top_predictions') or []
        true_item = next((item for item in top_predictions if item.get('label') == true_label), None)
        wrong_item = next((item for item in top_predictions if item.get('label') == predicted), None)
        if not true_item or not wrong_item:
            return False
        best = _best_pair_feature(true_item, wrong_item, min_gap=self.min_feature_gap)
        if not best:
            return False
        key = _pair_key(true_label, predicted)
        pair = self.pairs.setdefault(key, {
            'classes': sorted([str(true_label), str(predicted)]),
            'observations': 0,
            'supports': defaultdict(Counter),
            'examples': [],
        })
        pair['observations'] += 1
        pair['supports'][str(true_label)][best['feature']] += 1
        self.stats['observed'] += 1
        self.stats['observed_pairs'] = len(self.pairs)
        self.stats['feature_votes'] += 1
        if len(pair['examples']) < self.max_examples:
            pair['examples'].append({
                'path': sample.get('path'),
                'true_label': true_label,
                'predicted_label': predicted,
                'feature': best['feature'],
                'gap': round(float(best['gap']), 6),
            })
        return True

    def apply(self, results):
        adjusted = [dict(item) for item in (results or [])]
        for item in adjusted:
            item['pair_feature_delta'] = 0.0
            item['pair_feature_gate_reason'] = 'disabled_or_no_pair_memory'
            item['pair_feature_evidence'] = {}
        info = {'applied': False, 'changed_top1': False, 'reason': 'not_enough_candidates', 'deltas': {}}
        if len(adjusted) < 2:
            return adjusted, info
        top = adjusted[0]
        second = adjusted[1]
        top_label = top.get('label')
        second_label = second.get('label')
        margin = float(top.get('score', 0.0)) - float(second.get('score', 0.0))
        if margin > self.max_margin:
            for item in adjusted[:2]:
                item['pair_feature_gate_reason'] = 'margin_too_large'
            info['reason'] = 'margin_too_large'
            return adjusted, info
        pair = self.pairs.get(_pair_key(top_label, second_label))
        if not pair or pair['observations'] < self.min_observations:
            for item in adjusted[:2]:
                item['pair_feature_gate_reason'] = 'no_pair_feature_memory'
            info['reason'] = 'no_pair_feature_memory'
            return adjusted, info

        row_by_label = {item.get('label'): item for item in adjusted[:2]}
        delta_by_label = {top_label: 0.0, second_label: 0.0}
        evidence_by_label = {top_label: {}, second_label: {}}
        for label in (top_label, second_label):
            row = row_by_label.get(label)
            other_label = second_label if label == top_label else top_label
            other = row_by_label.get(other_label)
            if not row or not other:
                continue
            votes_by_feature = pair['supports'].get(str(label), {})
            for feature, votes in votes_by_feature.items():
                row_value = _feature_value(row, feature)
                other_value = _feature_value(other, feature)
                if row_value is None or other_value is None:
                    continue
                feature_gap = row_value - other_value
                if feature_gap <= self.min_feature_gap:
                    continue
                vote_factor = min(3, int(votes))
                delta = self.feature_bonus * vote_factor * min(1.0, feature_gap)
                if delta <= 0.0:
                    continue
                delta_by_label[label] += delta
                evidence_by_label[label][feature] = {
                    'votes': int(votes),
                    'feature_gap': round(float(feature_gap), 6),
                    'delta': round(float(delta), 6),
                }

        support_source_count = {
            label: len(evidence)
            for label, evidence in evidence_by_label.items()
        }
        if (
            delta_by_label.get(second_label, 0.0) > delta_by_label.get(top_label, 0.0)
            and support_source_count.get(second_label, 0) < self.min_sources_for_flip
        ):
            delta_by_label[second_label] = 0.0

        for item in adjusted[:2]:
            label = item.get('label')
            delta = max(-self.max_adjustment, min(self.max_adjustment, delta_by_label.get(label, 0.0)))
            item['pair_feature_delta'] = delta
            item['pair_feature_gate_reason'] = 'applied'
            item['pair_feature_evidence'] = evidence_by_label.get(label, {})
            item['score'] = float(item.get('score', 0.0)) + delta
        adjusted.sort(key=lambda item: item.get('score', 0.0), reverse=True)
        changed = adjusted[0].get('label') != top_label
        self.stats['applied'] += 1
        self.stats['changed_top1'] += 1 if changed else 0
        info.update({
            'applied': True,
            'changed_top1': changed,
            'reason': 'applied',
            'deltas': {label: round(float(delta), 6) for label, delta in delta_by_label.items()},
        })
        return adjusted, info

    def export(self):
        rows = []
        for key, pair in sorted(self.pairs.items()):
            rows.append({
                'pair': key,
                'classes': pair['classes'],
                'observations': pair['observations'],
                'supports': {
                    label: dict(counter.most_common())
                    for label, counter in pair['supports'].items()
                },
                'examples': pair['examples'],
            })
        return {
            'schema_version': 'pair_feature_memory_v1',
            'settings': {
                'feature_bonus': self.feature_bonus,
                'max_adjustment': self.max_adjustment,
                'max_margin': self.max_margin,
                'min_observations': self.min_observations,
                'min_feature_gap': self.min_feature_gap,
                'min_sources_for_flip': self.min_sources_for_flip,
            },
            'stats': dict(self.stats),
            'pairs': rows,
        }


def _evaluate_unattended(model, samples, weights, top_k, online_memory=None, visual_rule_memory=None, pair_feature_memory=None):
    rows = []
    per_class = defaultdict(lambda: {'total': 0, 'correct': 0})
    confusion = defaultdict(Counter)
    for sample in samples:
        raw_results = model.predict(sample['path'], weights)
        results = raw_results
        online_adjustment = None
        visual_rule_adjustment = None
        if online_memory is not None:
            results, online_adjustment = online_memory.apply(results)
        if visual_rule_memory is not None:
            results, visual_rule_adjustment = visual_rule_memory.apply(results)
        pair_feature_adjustment = None
        if pair_feature_memory is not None:
            results, pair_feature_adjustment = pair_feature_memory.apply(results)
        predicted = results[0]['label'] if results else None
        correct = predicted == sample['label']
        per_class[sample['label']]['total'] += 1
        per_class[sample['label']]['correct'] += 1 if correct else 0
        confusion[sample['label']][predicted or 'none'] += 1
        rows.append({
            'path': sample['path'],
            'true_label': sample['label'],
            'predicted_label': predicted,
            'raw_predicted_label': raw_results[0]['label'] if raw_results else None,
            'correct': correct,
            'online_adjustment': online_adjustment,
            'visual_rule_adjustment': visual_rule_adjustment,
            'pair_feature_adjustment': pair_feature_adjustment,
            'top_predictions': [
                {
                    'label': item.get('label'),
                    'score': round(float(item.get('score', 0.0)), 6),
                }
                for item in results[:max(1, int(top_k))]
            ],
            'raw_top_predictions': [
                {
                    'label': item.get('label'),
                    'score': round(float(item.get('score', 0.0)), 6),
                }
                for item in raw_results[:max(1, int(top_k))]
            ],
        })
    correct_count = sum(1 for row in rows if row['correct'])
    return {
        'sample_count': len(rows),
        'correct_count': correct_count,
        'accuracy': correct_count / max(1, len(rows)),
        'per_class': {
            label: {
                'total': item['total'],
                'correct': item['correct'],
                'accuracy': item['correct'] / max(1, item['total']),
            }
            for label, item in sorted(per_class.items())
        },
        'confusion': {label: dict(counts) for label, counts in sorted(confusion.items())},
        'errors': [row for row in rows if not row['correct']],
        'samples': rows,
    }


def _score_margin(results):
    if not results or len(results) < 2:
        return float('inf')
    return float(results[0].get('score', 0.0)) - float(results[1].get('score', 0.0))


def _teach_with_dynamic_oracle(
    model,
    samples,
    weights,
    top_k,
    max_questions,
    max_options,
    *,
    learning_policy='all_prototype',
    prototype_min_margin=0.02,
    online_memory=None,
    visual_rule_memory=None,
    pair_feature_memory=None,
):
    summary = {
        'sample_count': 0,
        'raw_correct_count': 0,
        'dynamic_correct_count': 0,
        'asked_count': 0,
        'converted_count': 0,
        'true_label_not_in_options_count': 0,
        'prototype_update_count': 0,
        'memory_observation_count': 0,
        'pair_feature_observation_count': 0,
        'skipped_prototype_count': 0,
    }
    examples = []
    for sample in samples:
        results = model.predict(sample['path'], weights)
        raw_label = results[0]['label'] if results else None
        raw_correct = raw_label == sample['label']
        margin = _score_margin(results)
        dialog = simulate_dynamic_dialog(
            results[:max(1, int(top_k))],
            sample['label'],
            max_questions=max_questions,
            max_options=max_options,
            candidate_top_k=top_k,
        )
        dynamic_correct = bool(dialog.get('correct'))
        summary['sample_count'] += 1
        summary['raw_correct_count'] += 1 if raw_correct else 0
        summary['dynamic_correct_count'] += 1 if dynamic_correct else 0
        summary['asked_count'] += int(dialog.get('asked') or 0)
        summary['converted_count'] += 1 if dynamic_correct and not raw_correct else 0
        summary['true_label_not_in_options_count'] += 1 if dialog.get('reason') == 'true_label_not_in_options' else 0
        if len(examples) < 20 and (not raw_correct or dialog.get('asked')):
            examples.append({
                'path': sample['path'],
                'true_label': sample['label'],
                'raw_label': raw_label,
                'resolved_label': dialog.get('resolved_label'),
                'reason': dialog.get('reason'),
                'asked': dialog.get('asked'),
                'margin': margin,
            })
        should_update_prototype = False
        if learning_policy == 'all_prototype':
            should_update_prototype = dynamic_correct
        elif learning_policy == 'gated_prototype':
            should_update_prototype = raw_correct and margin >= prototype_min_margin
        elif learning_policy in ('gated_memory', 'memory_only', 'pair_feature_memory', 'hybrid_memory'):
            should_update_prototype = raw_correct and margin >= prototype_min_margin
            if learning_policy in ('memory_only', 'pair_feature_memory', 'hybrid_memory'):
                should_update_prototype = False
            if dynamic_correct and not raw_correct and online_memory is not None:
                learn_row = {
                    'path': sample['path'],
                    'true_label': sample['label'],
                    'predicted_label': raw_label,
                    'correct': False,
                    'top_predictions': results[:max(1, int(top_k))],
                    'diagnosis': {'score_margin': margin},
                }
                online_memory.observe(learn_row)
                if visual_rule_memory is not None:
                    visual_rule_memory.observe(learn_row)
                summary['memory_observation_count'] += 1
            if dynamic_correct and not raw_correct and pair_feature_memory is not None:
                learn_row = {
                    'path': sample['path'],
                    'true_label': sample['label'],
                    'predicted_label': raw_label,
                    'correct': False,
                    'top_predictions': results[:max(1, int(top_k))],
                    'diagnosis': {'score_margin': margin},
                }
                if pair_feature_memory.observe(learn_row):
                    summary['pair_feature_observation_count'] += 1
        else:
            raise ValueError(f'Unknown learning_policy: {learning_policy}')

        if should_update_prototype:
            model.add_confirmed_sample(sample['label'], sample['path'])
            summary['prototype_update_count'] += 1
        else:
            summary['skipped_prototype_count'] += 1
    total = max(1, summary['sample_count'])
    summary['raw_accuracy'] = summary['raw_correct_count'] / total
    summary['dynamic_accuracy'] = summary['dynamic_correct_count'] / total
    summary['examples'] = examples
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Simulate interactive teaching, then measure unattended accuracy on held-out samples.'
    )
    parser.add_argument('--config', required=True, help='Project task_config.yaml')
    parser.add_argument('--output-dir', required=True, help='Directory for the experiment report.')
    parser.add_argument('--deep-cache-dir', help='CLIP feature cache directory.')
    parser.add_argument('--model-cache', help='Existing fitted PrototypeModel cache.')
    parser.add_argument('--rebuild-model-cache', action='store_true')
    parser.add_argument('--teach-per-class', type=int, default=10)
    parser.add_argument('--learning-policy', choices=['all_prototype', 'gated_prototype', 'gated_memory', 'memory_only', 'pair_feature_memory', 'hybrid_memory'], default='all_prototype')
    parser.add_argument('--prototype-min-margin', type=float, default=0.02)
    parser.add_argument('--disable-visual-rules', action='store_true')
    parser.add_argument('--memory-min-observations', type=int, default=1)
    parser.add_argument('--memory-max-margin', type=float, default=0.035)
    parser.add_argument('--memory-adjustment-weight', type=float, default=0.02)
    parser.add_argument('--memory-max-adjustment', type=float, default=0.04)
    parser.add_argument('--memory-min-sources-for-flip', type=int, default=2)
    parser.add_argument('--visual-rule-allow-rank-flip', action='store_true')
    parser.add_argument('--pair-feature-bonus', type=float, default=0.02)
    parser.add_argument('--pair-feature-max-adjustment', type=float, default=0.04)
    parser.add_argument('--pair-feature-min-gap', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--top-k', type=int, default=DEFAULT_ASK_CANDIDATE_TOP_K)
    parser.add_argument('--max-questions', type=int, default=DEFAULT_ASK_MAX_QUESTIONS)
    parser.add_argument('--max-options', type=int, default=DEFAULT_ASK_MAX_OPTIONS)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()

    started = time.perf_counter()

    def profile(message):
        if args.profile:
            print(f'[profile] {time.perf_counter() - started:.1f}s {message}', flush=True)

    cfg = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    loader = DatasetLoader(cfg['paths']['dataset_dir'])
    objects = loader.load_objects()
    labels = _class_names(objects)
    train_samples = loader.load_train_samples()
    eval_samples = loader.load_eval_samples()
    teaching_samples, heldout_samples = _stratified_split(eval_samples, args.teach_per_class, args.seed)
    profile(f'loaded data: train={len(train_samples)}, teaching={len(teaching_samples)}, heldout={len(heldout_samples)}')

    feature_spec = parse_feature_config(cfg, classes=labels or cfg.get('classes', []))
    weights = AdaptiveWeights(
        initial_feature_weights(cfg, feature_spec),
        cfg['learning'].get('update_step', 0.07),
        cfg['learning'].get('min_weight', 0.05),
        cfg['learning'].get('max_weight', 0.70),
    ).export()
    deep_cache_dir = Path(args.deep_cache_dir) if args.deep_cache_dir else output_dir / '.cache' / 'deep_features'
    model_cache_path = Path(args.model_cache) if args.model_cache else None
    model = _load_or_fit_model(
        cfg,
        train_samples,
        feature_spec,
        deep_cache_dir,
        model_cache_path,
        rebuild=args.rebuild_model_cache,
    )
    profile('loaded model')

    baseline = _evaluate_unattended(model, heldout_samples, weights, args.top_k)
    profile('baseline heldout evaluated')
    online_memory = None
    visual_rule_memory = None
    pair_feature_memory = None
    if args.learning_policy in ('gated_memory', 'memory_only', 'hybrid_memory'):
        online_memory = OnlineConfusionExperience(
            max_margin=args.memory_max_margin,
            adjustment_weight=args.memory_adjustment_weight,
            max_adjustment=args.memory_max_adjustment,
            min_observations=args.memory_min_observations,
            min_sources_for_flip=args.memory_min_sources_for_flip,
        )
        if not args.disable_visual_rules:
            visual_rule_memory = PairVisualRuleMemory(
                max_margin=args.memory_max_margin,
                min_observations=args.memory_min_observations,
                allow_rank_flip=args.visual_rule_allow_rank_flip,
            )
    if args.learning_policy in ('pair_feature_memory', 'hybrid_memory'):
        pair_feature_memory = PairFeatureMemory(
            feature_bonus=args.pair_feature_bonus,
            max_adjustment=args.pair_feature_max_adjustment,
            max_margin=args.memory_max_margin,
            min_observations=args.memory_min_observations,
            min_feature_gap=args.pair_feature_min_gap,
        )
    teaching_summary = _teach_with_dynamic_oracle(
        model,
        teaching_samples,
        weights,
        args.top_k,
        args.max_questions,
        args.max_options,
        learning_policy=args.learning_policy,
        prototype_min_margin=args.prototype_min_margin,
        online_memory=online_memory,
        visual_rule_memory=visual_rule_memory,
        pair_feature_memory=pair_feature_memory,
    )
    profile('interactive teaching simulated')
    after = _evaluate_unattended(
        model,
        heldout_samples,
        weights,
        args.top_k,
        online_memory=online_memory,
        visual_rule_memory=visual_rule_memory,
        pair_feature_memory=pair_feature_memory,
    )
    profile('post-teaching heldout evaluated')

    report = {
        'schema_version': VERSION,
        'dataset_dir': cfg['paths']['dataset_dir'],
        'seed': args.seed,
        'teach_per_class': args.teach_per_class,
        'class_count': len(labels),
        'initial_train_sample_count': len(train_samples),
        'teaching_sample_count': len(teaching_samples),
        'heldout_sample_count': len(heldout_samples),
        'top_k': args.top_k,
        'learning_policy': args.learning_policy,
        'prototype_min_margin': args.prototype_min_margin,
        'teaching': teaching_summary,
        'online_memory': None if online_memory is None else online_memory.export(),
        'visual_rule_memory': None if visual_rule_memory is None else visual_rule_memory.export(),
        'pair_feature_memory': None if pair_feature_memory is None else pair_feature_memory.export(),
        'heldout_baseline_unattended': baseline,
        'heldout_after_teaching_unattended': after,
        'delta_accuracy': after['accuracy'] - baseline['accuracy'],
    }
    report_path = output_dir / 'interactive_then_unattended_report.json'
    save_json(report_path, report)
    print(f'Teaching samples: {len(teaching_samples)}')
    print(f'Held-out samples: {len(heldout_samples)}')
    print(
        'Teaching dynamic accuracy: '
        f'{teaching_summary["dynamic_correct_count"]}/{teaching_summary["sample_count"]} '
        f'= {teaching_summary["dynamic_accuracy"]:.3f}'
    )
    print(
        'Teaching updates: '
        f'prototype={teaching_summary["prototype_update_count"]}, '
        f'memory={teaching_summary["memory_observation_count"]}, '
        f'pair_feature={teaching_summary["pair_feature_observation_count"]}, '
        f'skipped_prototype={teaching_summary["skipped_prototype_count"]}'
    )
    print(
        'Held-out unattended before teaching: '
        f'{baseline["correct_count"]}/{baseline["sample_count"]} = {baseline["accuracy"]:.3f}'
    )
    print(
        'Held-out unattended after teaching: '
        f'{after["correct_count"]}/{after["sample_count"]} = {after["accuracy"]:.3f}'
    )
    print(f'Delta: {report["delta_accuracy"]:+.3f}')
    print('Report:', report_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
