from collections import Counter, defaultdict


SOURCE_KEYS = (
    'prototype_score',
    'subprototype_score',
    'knn_score',
    'text_semantic_score',
    'pairwise_score',
    'crop_rerank_score',
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

    local_votes = sum(int(true_sources.get(key, 0)) for key in ('knn_score', 'pairwise_score', 'crop_rerank_score'))
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
