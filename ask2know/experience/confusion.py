from collections import Counter, defaultdict


SOURCE_KEYS = (
    'prototype_score',
    'subprototype_score',
    'knn_score',
    'text_semantic_score',
    'hierarchy_score',
    'pairwise_score',
    'crop_rerank_score',
    'pair_confusion_score',
    'concept_score',
)


def _as_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _source_name(key):
    return str(key).replace('_score', '').replace('_', ' ')


def _true_rank(sample):
    true_label = sample.get('true_label')
    for idx, item in enumerate(sample.get('top_predictions') or [], 1):
        if item.get('label') == true_label:
            return idx
    return None


def _pair_key(true_label, predicted_label):
    return f'{true_label} -> {predicted_label}'


def _pair_recommendations(pair, contrast):
    true_sources = pair.get('true_supported_sources', {})
    wrong_sources = pair.get('wrong_supported_sources', {})
    recommendations = []

    local_votes = sum(int(true_sources.get(key, 0)) for key in (
        'knn_score',
        'hierarchy_score',
        'pairwise_score',
        'crop_rerank_score',
        'pair_confusion_score',
    ))
    if local_votes:
        recommendations.append('Local evidence sometimes supports the true class; use pair-specific nearest samples before changing the final label.')
    if int(wrong_sources.get('text_semantic_score', 0)):
        recommendations.append('Text prompts have misled this pair; treat text_semantic_score as weak evidence for this pair.')
    if int(wrong_sources.get('concept_score', 0)):
        recommendations.append('Concept score has supported the wrong class; ask the user for pair-specific visual criteria instead of relying on generic concepts.')

    top_groups = (contrast or {}).get('top_group_differences') or []
    if top_groups:
        groups = ', '.join(item['group'] for item in top_groups[:3])
        recommendations.append(f'Ask about these candidate discriminative groups first: {groups}.')

    top_concepts = (contrast or {}).get('top_concept_differences') or []
    if top_concepts:
        concepts = ', '.join(item['concept'] for item in top_concepts[:3])
        recommendations.append(f'Check whether these concept differences are visible in the current image: {concepts}.')

    if not recommendations:
        recommendations.append('No stable source explains this pair yet; collect a user note during the next correction.')
    return recommendations


def build_confusion_experience_report(samples, model=None, weak_signal_threshold=0.005, max_examples=8):
    pair_map = {}
    low_accuracy = defaultdict(lambda: {'total': 0, 'correct': 0})
    source_totals = {
        'wrong_supported_sources': Counter(),
        'true_supported_sources': Counter(),
    }

    for sample in samples or []:
        true_label = sample.get('true_label')
        predicted = sample.get('predicted_label')
        if true_label:
            low_accuracy[true_label]['total'] += 1
            low_accuracy[true_label]['correct'] += 1 if sample.get('correct') else 0
        if sample.get('correct') or not true_label or not predicted:
            continue

        key = _pair_key(true_label, predicted)
        pair = pair_map.setdefault(key, {
            'true_label': true_label,
            'predicted_label': predicted,
            'error_count': 0,
            'true_in_top_k_count': 0,
            'score_margins': [],
            'reason_counts': Counter(),
            'wrong_supported_sources': Counter(),
            'true_supported_sources': Counter(),
            'examples': [],
        })
        pair['error_count'] += 1
        diagnosis = sample.get('diagnosis') or {}
        margin = _as_float(diagnosis.get('score_margin'))
        if margin is not None:
            pair['score_margins'].append(margin)
        rank = _true_rank(sample)
        if rank is not None:
            pair['true_in_top_k_count'] += 1
        for reason in diagnosis.get('reason_codes') or []:
            pair['reason_counts'][reason] += 1

        compare = diagnosis.get('true_label_compare') or {}
        deltas = compare.get('source_deltas_top_minus_true') or {}
        for source in SOURCE_KEYS:
            delta = _as_float(deltas.get(source))
            if delta is None:
                continue
            if delta > weak_signal_threshold:
                pair['wrong_supported_sources'][source] += 1
                source_totals['wrong_supported_sources'][source] += 1
            elif delta < -weak_signal_threshold:
                pair['true_supported_sources'][source] += 1
                source_totals['true_supported_sources'][source] += 1

        if len(pair['examples']) < max_examples:
            pair['examples'].append({
                'path': sample.get('path'),
                'true_rank': rank,
                'score_margin': margin,
                'top_predictions': sample.get('top_predictions', [])[:3],
            })

    pairs = []
    for pair in pair_map.values():
        margins = pair.pop('score_margins')
        reason_counts = pair.pop('reason_counts')
        wrong_sources = pair.pop('wrong_supported_sources')
        true_sources = pair.pop('true_supported_sources')
        contrast = None
        if model is not None:
            contrast = model.pair_discriminative_summary(pair['true_label'], pair['predicted_label'])
        pair['avg_score_margin'] = None if not margins else float(sum(margins) / len(margins))
        pair['reason_counts'] = dict(reason_counts.most_common())
        pair['wrong_supported_sources'] = dict(wrong_sources.most_common())
        pair['true_supported_sources'] = dict(true_sources.most_common())
        pair['model_contrast_summary'] = contrast
        pair['recommendations'] = _pair_recommendations(pair, contrast)
        pairs.append(pair)

    pairs.sort(key=lambda item: item['error_count'], reverse=True)
    low_rows = []
    for label, item in low_accuracy.items():
        total = int(item['total'])
        correct = int(item['correct'])
        if total:
            low_rows.append({
                'label': label,
                'total': total,
                'correct': correct,
                'accuracy': correct / total,
            })
    low_rows.sort(key=lambda item: (item['accuracy'], -item['total'], item['label']))

    source_summary = {
        name: {
            _source_name(key): value
            for key, value in counter.most_common()
        }
        for name, counter in source_totals.items()
    }
    return {
        'schema_version': 'confusion_experience_v1',
        'summary': {
            'confusion_pair_count': len(pairs),
            'error_count': sum(item['error_count'] for item in pairs),
            'lowest_accuracy_classes': low_rows[:10],
            'source_summary': source_summary,
        },
        'pairs': pairs,
    }


def _unordered_pair_key(label_a, label_b):
    labels = sorted([str(label_a), str(label_b)])
    return f'{labels[0]} <-> {labels[1]}'


class OnlineConfusionExperience:
    """Use earlier revealed mistakes as weak pair-specific evidence for later samples."""

    def __init__(
        self,
        weak_signal_threshold=0.005,
        max_margin=0.035,
        adjustment_weight=0.02,
        max_adjustment=0.04,
        min_observations=1,
        max_examples=8,
        allow_negative_adjustments=False,
        min_sources_for_flip=2,
    ):
        self.weak_signal_threshold = float(weak_signal_threshold)
        self.max_margin = float(max_margin)
        self.adjustment_weight = float(adjustment_weight)
        self.max_adjustment = float(max_adjustment)
        self.min_observations = int(min_observations)
        self.max_examples = int(max_examples)
        self.allow_negative_adjustments = bool(allow_negative_adjustments)
        self.min_sources_for_flip = max(1, int(min_sources_for_flip))
        self.pairs = {}
        self.stats = {'observed_errors': 0, 'observed_pairs': 0, 'applied': 0, 'changed_top1': 0, 'helped': 0, 'hurt': 0}

    def _pair(self, label_a, label_b):
        key = _unordered_pair_key(label_a, label_b)
        if key not in self.pairs:
            labels = sorted([str(label_a), str(label_b)])
            self.pairs[key] = {
                'classes': labels,
                'observations': 0,
                'support_sources': defaultdict(Counter),
                'misleading_sources': defaultdict(Counter),
                'examples': [],
            }
            self.stats['observed_pairs'] = len(self.pairs)
        return self.pairs[key]

    def observe(self, sample):
        true_label = sample.get('true_label')
        predicted = sample.get('predicted_label')
        if sample.get('correct') or not true_label or not predicted or true_label == predicted:
            return
        top_predictions = sample.get('top_predictions') or []
        predicted_item = next((item for item in top_predictions if item.get('label') == predicted), None)
        true_item = next((item for item in top_predictions if item.get('label') == true_label), None)
        if not predicted_item or not true_item:
            return
        pair = self._pair(true_label, predicted)
        pair['observations'] += 1
        self.stats['observed_errors'] += 1
        for source in SOURCE_KEYS:
            pred_value = _as_float(predicted_item.get(source))
            true_value = _as_float(true_item.get(source))
            if pred_value is None or true_value is None:
                continue
            delta = pred_value - true_value
            if delta < -self.weak_signal_threshold:
                pair['support_sources'][true_label][source] += 1
            elif delta > self.weak_signal_threshold:
                pair['misleading_sources'][predicted][source] += 1
        if len(pair['examples']) < self.max_examples:
            pair['examples'].append({
                'path': sample.get('path'),
                'true_label': true_label,
                'predicted_label': predicted,
                'score_margin': (sample.get('diagnosis') or {}).get('score_margin'),
            })

    def apply(self, results):
        adjusted = [dict(item) for item in (results or [])]
        for item in adjusted:
            item['online_experience_delta'] = 0.0
            item['online_experience_gate_reason'] = 'disabled_or_no_pair_memory'
            item['online_experience_evidence'] = {}
        info = {'applied': False, 'changed_top1': False, 'reason': 'not_enough_candidates', 'deltas': {}}
        if len(adjusted) < 2:
            return adjusted, info

        top = adjusted[0]
        second = adjusted[1]
        top_label = top.get('label')
        second_label = second.get('label')
        margin = float(top.get('score', 0.0)) - float(second.get('score', 0.0))
        if margin > self.max_margin:
            reason = 'margin_too_large'
            for item in adjusted[:2]:
                item['online_experience_gate_reason'] = reason
            info['reason'] = reason
            return adjusted, info

        pair = self.pairs.get(_unordered_pair_key(top_label, second_label))
        if not pair or pair['observations'] < self.min_observations:
            reason = 'no_pair_memory'
            for item in adjusted[:2]:
                item['online_experience_gate_reason'] = reason
            info['reason'] = reason
            return adjusted, info

        delta_by_label = {top_label: 0.0, second_label: 0.0}
        evidence_by_label = {top_label: {}, second_label: {}}
        row_by_label = {item.get('label'): item for item in adjusted[:2]}
        local_sources = {'knn_score', 'hierarchy_score', 'pairwise_score', 'crop_rerank_score', 'pair_confusion_score'}

        for label in (top_label, second_label):
            row = row_by_label.get(label)
            other_label = second_label if label == top_label else top_label
            other = row_by_label.get(other_label)
            if not row or not other:
                continue
            for source in SOURCE_KEYS:
                row_value = _as_float(row.get(source))
                other_value = _as_float(other.get(source))
                if row_value is None or other_value is None:
                    continue
                source_gap = row_value - other_value
                if abs(source_gap) <= self.weak_signal_threshold:
                    continue
                factor = 1.35 if source in local_sources else 1.0
                support_count = int(pair['support_sources'].get(label, {}).get(source, 0))
                misleading_count = int(pair['misleading_sources'].get(label, {}).get(source, 0))
                source_delta = 0.0
                if support_count and source_gap > self.weak_signal_threshold:
                    source_delta += self.adjustment_weight * factor * min(support_count, 3) * source_gap
                if self.allow_negative_adjustments and misleading_count and source_gap > 0:
                    source_delta -= self.adjustment_weight * factor * min(misleading_count, 3) * source_gap
                if source_delta:
                    delta_by_label[label] += source_delta
                    evidence_by_label[label][source] = {
                        'support_count': support_count,
                        'misleading_count': misleading_count,
                        'source_gap': round(float(source_gap), 6),
                        'delta': round(float(source_delta), 6),
                    }

        support_source_count_by_label = {
            label: sum(1 for evidence in sources.values() if evidence.get('support_count', 0) > 0)
            for label, sources in evidence_by_label.items()
        }
        if (
            delta_by_label.get(second_label, 0.0) > delta_by_label.get(top_label, 0.0)
            and support_source_count_by_label.get(second_label, 0) < self.min_sources_for_flip
        ):
            delta_by_label[second_label] = 0.0

        for item in adjusted[:2]:
            label = item.get('label')
            delta = max(-self.max_adjustment, min(self.max_adjustment, delta_by_label.get(label, 0.0)))
            item['online_experience_delta'] = delta
            item['online_experience_gate_reason'] = 'applied'
            item['online_experience_evidence'] = evidence_by_label.get(label, {})
            item['score'] = float(item.get('score', 0.0)) + delta

        adjusted.sort(key=lambda item: item.get('score', 0.0), reverse=True)
        self.stats['applied'] += 1
        changed_top1 = adjusted[0].get('label') != top_label
        if changed_top1:
            self.stats['changed_top1'] += 1
        info.update({
            'applied': True,
            'changed_top1': changed_top1,
            'reason': 'applied',
            'deltas': {label: round(float(delta), 6) for label, delta in delta_by_label.items()},
        })
        return adjusted, info

    def record_outcome(self, raw_correct, online_correct, changed_top1):
        if not changed_top1:
            return
        if online_correct and not raw_correct:
            self.stats['helped'] += 1
        elif raw_correct and not online_correct:
            self.stats['hurt'] += 1

    def export(self):
        pair_rows = []
        for key, pair in sorted(self.pairs.items()):
            support = {label: dict(counter.most_common()) for label, counter in pair['support_sources'].items()}
            misleading = {label: dict(counter.most_common()) for label, counter in pair['misleading_sources'].items()}
            pair_rows.append({
                'pair': key,
                'classes': pair['classes'],
                'observations': pair['observations'],
                'support_sources': support,
                'misleading_sources': misleading,
                'examples': pair['examples'],
            })
        return {
            'schema_version': 'online_confusion_experience_v1',
            'settings': {
                'weak_signal_threshold': self.weak_signal_threshold,
                'max_margin': self.max_margin,
                'adjustment_weight': self.adjustment_weight,
                'max_adjustment': self.max_adjustment,
                'min_observations': self.min_observations,
                'allow_negative_adjustments': self.allow_negative_adjustments,
                'min_sources_for_flip': self.min_sources_for_flip,
            },
            'stats': dict(self.stats),
            'pairs': pair_rows,
        }


def _concept_match_gap(sample_concepts, label_concepts, other_concepts, concept):
    value = _as_float((sample_concepts or {}).get(concept))
    label_value = _as_float((label_concepts or {}).get(concept))
    other_value = _as_float((other_concepts or {}).get(concept))
    if value is None or label_value is None or other_value is None:
        return None
    return abs(value - other_value) - abs(value - label_value)


class PairVisualRuleMemory:
    """Learn pair-specific concept rules from earlier revealed mistakes."""

    def __init__(
        self,
        weak_signal_threshold=0.005,
        max_margin=0.04,
        rule_weight=0.035,
        max_adjustment=0.05,
        min_observations=1,
        min_concept_gap=0.10,
        min_match_gap=0.04,
        allow_rank_flip=False,
        max_rules_per_pair=6,
        max_examples=8,
    ):
        self.weak_signal_threshold = float(weak_signal_threshold)
        self.max_margin = float(max_margin)
        self.rule_weight = float(rule_weight)
        self.max_adjustment = float(max_adjustment)
        self.min_observations = int(min_observations)
        self.min_concept_gap = float(min_concept_gap)
        self.min_match_gap = float(min_match_gap)
        self.allow_rank_flip = bool(allow_rank_flip)
        self.max_rules_per_pair = int(max_rules_per_pair)
        self.max_examples = int(max_examples)
        self.pairs = {}
        self.stats = {
            'observed_errors': 0,
            'observed_pairs': 0,
            'learned_rule_votes': 0,
            'applied': 0,
            'changed_top1': 0,
            'helped': 0,
            'hurt': 0,
        }

    def _pair(self, label_a, label_b):
        key = _unordered_pair_key(label_a, label_b)
        if key not in self.pairs:
            labels = sorted([str(label_a), str(label_b)])
            self.pairs[key] = {
                'classes': labels,
                'observations': 0,
                'rules': defaultdict(Counter),
                'concept_stats': defaultdict(lambda: defaultdict(float)),
                'examples': [],
            }
            self.stats['observed_pairs'] = len(self.pairs)
        return self.pairs[key]

    def _candidate_rules(self, true_item, wrong_item):
        true_label = true_item.get('label')
        sample_concepts = true_item.get('concepts') or wrong_item.get('concepts') or {}
        true_concepts = true_item.get('class_concepts') or {}
        wrong_concepts = wrong_item.get('class_concepts') or {}
        rows = []
        for concept in sorted(set(true_concepts) | set(wrong_concepts)):
            tv = _as_float(true_concepts.get(concept), 0.0)
            wv = _as_float(wrong_concepts.get(concept), 0.0)
            gap = abs(tv - wv)
            if gap < self.min_concept_gap:
                continue
            match_gap = _concept_match_gap(sample_concepts, true_concepts, wrong_concepts, concept)
            if match_gap is None or match_gap < self.min_match_gap:
                continue
            rows.append({
                'concept': concept,
                'supports': true_label,
                'direction': 'high' if tv >= wv else 'low',
                'class_gap': float(gap),
                'match_gap': float(match_gap),
                'sample_value': _as_float(sample_concepts.get(concept), 0.0),
                'support_value': float(tv),
                'other_value': float(wv),
            })
        rows.sort(key=lambda item: (item['match_gap'], item['class_gap']), reverse=True)
        return rows[:self.max_rules_per_pair]

    def observe(self, sample):
        true_label = sample.get('true_label')
        predicted = sample.get('predicted_label')
        if sample.get('correct') or not true_label or not predicted or true_label == predicted:
            return
        top_predictions = sample.get('top_predictions') or []
        predicted_item = next((item for item in top_predictions if item.get('label') == predicted), None)
        true_item = next((item for item in top_predictions if item.get('label') == true_label), None)
        if not predicted_item or not true_item:
            return
        rules = self._candidate_rules(true_item, predicted_item)
        if not rules:
            return
        pair = self._pair(true_label, predicted)
        pair['observations'] += 1
        self.stats['observed_errors'] += 1
        for rule in rules:
            concept = rule['concept']
            pair['rules'][true_label][concept] += 1
            stats = pair['concept_stats'][true_label + '|' + concept]
            stats['votes'] += 1.0
            stats['class_gap_sum'] += rule['class_gap']
            stats['match_gap_sum'] += rule['match_gap']
            stats['sample_value_sum'] += rule['sample_value']
            stats['support_value_sum'] += rule['support_value']
            stats['other_value_sum'] += rule['other_value']
            self.stats['learned_rule_votes'] += 1
        if len(pair['examples']) < self.max_examples:
            pair['examples'].append({
                'path': sample.get('path'),
                'true_label': true_label,
                'predicted_label': predicted,
                'rules': rules[:3],
            })

    def apply(self, results):
        adjusted = [dict(item) for item in (results or [])]
        for item in adjusted:
            item['visual_rule_delta'] = 0.0
            item['visual_rule_gate_reason'] = 'disabled_or_no_pair_rules'
            item['visual_rule_evidence'] = {}
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
                item['visual_rule_gate_reason'] = 'margin_too_large'
            info['reason'] = 'margin_too_large'
            return adjusted, info
        pair = self.pairs.get(_unordered_pair_key(top_label, second_label))
        if not pair or pair['observations'] < self.min_observations:
            for item in adjusted[:2]:
                item['visual_rule_gate_reason'] = 'no_pair_rules'
            info['reason'] = 'no_pair_rules'
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
            sample_concepts = row.get('concepts') or other.get('concepts') or {}
            label_concepts = row.get('class_concepts') or {}
            other_concepts = other.get('class_concepts') or {}
            rule_counter = pair['rules'].get(label)
            if not rule_counter:
                continue
            for concept, votes in rule_counter.most_common(self.max_rules_per_pair):
                match_gap = _concept_match_gap(sample_concepts, label_concepts, other_concepts, concept)
                if match_gap is None or match_gap < self.min_match_gap:
                    continue
                stat = pair['concept_stats'].get(label + '|' + concept, {})
                vote_count = int(votes)
                avg_class_gap = float(stat.get('class_gap_sum', 0.0)) / max(1.0, float(stat.get('votes', vote_count)))
                strength = min(3.0, float(vote_count)) * min(1.0, max(0.0, avg_class_gap)) * min(1.0, max(0.0, match_gap))
                delta = self.rule_weight * strength
                if delta <= 0.0:
                    continue
                delta_by_label[label] += delta
                evidence_by_label[label][concept] = {
                    'votes': vote_count,
                    'match_gap': round(float(match_gap), 6),
                    'avg_class_gap': round(float(avg_class_gap), 6),
                    'delta': round(float(delta), 6),
                }

        if not self.allow_rank_flip:
            top_delta = delta_by_label.get(top_label, 0.0)
            second_delta = delta_by_label.get(second_label, 0.0)
            if second_delta - top_delta >= margin:
                delta_by_label[second_label] = max(0.0, margin + top_delta - 1e-9)

        for item in adjusted[:2]:
            label = item.get('label')
            delta = max(-self.max_adjustment, min(self.max_adjustment, delta_by_label.get(label, 0.0)))
            item['visual_rule_delta'] = delta
            item['visual_rule_gate_reason'] = 'applied'
            item['visual_rule_evidence'] = evidence_by_label.get(label, {})
            item['score'] = float(item.get('score', 0.0)) + delta
        adjusted.sort(key=lambda item: item.get('score', 0.0), reverse=True)
        self.stats['applied'] += 1
        changed_top1 = adjusted[0].get('label') != top_label
        if changed_top1:
            self.stats['changed_top1'] += 1
        info.update({
            'applied': True,
            'changed_top1': changed_top1,
            'reason': 'applied',
            'deltas': {label: round(float(delta), 6) for label, delta in delta_by_label.items()},
        })
        return adjusted, info

    def record_outcome(self, raw_correct, visual_correct, changed_top1):
        if not changed_top1:
            return
        if visual_correct and not raw_correct:
            self.stats['helped'] += 1
        elif raw_correct and not visual_correct:
            self.stats['hurt'] += 1

    def export(self):
        rows = []
        for key, pair in sorted(self.pairs.items()):
            rule_rows = []
            for label, counter in pair['rules'].items():
                for concept, votes in counter.most_common():
                    stat = pair['concept_stats'].get(label + '|' + concept, {})
                    total = max(1.0, float(stat.get('votes', votes)))
                    rule_rows.append({
                        'supports': label,
                        'concept': concept,
                        'votes': int(votes),
                        'avg_class_gap': float(stat.get('class_gap_sum', 0.0)) / total,
                        'avg_match_gap': float(stat.get('match_gap_sum', 0.0)) / total,
                        'avg_sample_value': float(stat.get('sample_value_sum', 0.0)) / total,
                        'avg_support_value': float(stat.get('support_value_sum', 0.0)) / total,
                        'avg_other_value': float(stat.get('other_value_sum', 0.0)) / total,
                    })
            rule_rows.sort(key=lambda item: (item['votes'], item['avg_match_gap'], item['avg_class_gap']), reverse=True)
            rows.append({
                'pair': key,
                'classes': pair['classes'],
                'observations': pair['observations'],
                'rules': rule_rows,
                'examples': pair['examples'],
            })
        return {
            'schema_version': 'pair_visual_rule_memory_v1',
            'settings': {
                'weak_signal_threshold': self.weak_signal_threshold,
                'max_margin': self.max_margin,
                'rule_weight': self.rule_weight,
                'max_adjustment': self.max_adjustment,
                'min_observations': self.min_observations,
                'min_concept_gap': self.min_concept_gap,
                'min_match_gap': self.min_match_gap,
                'allow_rank_flip': self.allow_rank_flip,
                'max_rules_per_pair': self.max_rules_per_pair,
            },
            'stats': dict(self.stats),
            'pairs': rows,
        }
