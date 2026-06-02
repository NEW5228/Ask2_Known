SCORE_KEYS = (
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


def score_margin(results):
    if len(results or []) < 2:
        return None
    return float(results[0].get('score', 0.0)) - float(results[1].get('score', 0.0))


def source_margins(results):
    if len(results or []) < 2:
        return {}
    top = results[0]
    second = results[1]
    margins = {}
    for key in SCORE_KEYS:
        a = _as_float(top.get(key))
        b = _as_float(second.get(key))
        if a is not None and b is not None:
            margins[key] = a - b
    return margins


def diagnose_prediction(results, true_label=None, low_margin_threshold=0.015, weak_signal_threshold=0.005):
    results = list(results or [])
    if not results:
        return {
            'top_label': None,
            'runner_up_label': None,
            'score_margin': None,
            'needs_review': True,
            'reason_codes': ['no_prediction'],
            'source_margins': {},
        }

    top = results[0]
    second = results[1] if len(results) > 1 else None
    margin = score_margin(results)
    margins = source_margins(results)
    reason_codes = []
    if margin is None or margin < float(low_margin_threshold):
        reason_codes.append('low_score_margin')

    if second is not None:
        for key, reason in (
            ('prototype_score', 'prototype_led'),
            ('subprototype_score', 'subprototype_led'),
            ('knn_score', 'knn_led'),
            ('text_semantic_score', 'text_semantic_led'),
            ('hierarchy_score', 'hierarchy_led'),
            ('pairwise_score', 'pairwise_led'),
            ('crop_rerank_score', 'crop_rerank_led'),
            ('pair_confusion_score', 'pair_confusion_led'),
            ('concept_score', 'concept_led'),
        ):
            delta = margins.get(key)
            if delta is not None and delta > float(weak_signal_threshold):
                reason_codes.append(reason)
        for key, reason in (
            ('prototype_score', 'prototype_weak_or_against_top'),
            ('subprototype_score', 'subprototype_conflicts_with_top'),
            ('knn_score', 'knn_conflicts_with_top'),
            ('text_semantic_score', 'text_semantic_conflicts_with_top'),
            ('hierarchy_score', 'hierarchy_conflicts_with_top'),
            ('pairwise_score', 'pairwise_conflicts_with_top'),
            ('crop_rerank_score', 'crop_rerank_conflicts_with_top'),
            ('pair_confusion_score', 'pair_confusion_conflicts_with_top'),
            ('concept_score', 'concept_conflicts_with_top'),
        ):
            delta = margins.get(key)
            if delta is not None and delta < -float(weak_signal_threshold):
                reason_codes.append(reason)

    true_compare = None
    if true_label is not None:
        true_item = next((item for item in results if item.get('label') == true_label), None)
        correct = top.get('label') == true_label
        if correct:
            reason_codes.append('correct')
        else:
            reason_codes.append('misclassified')
            if true_item is None:
                reason_codes.append('true_label_not_in_top_predictions')
            else:
                deltas = {}
                for key in ('score',) + SCORE_KEYS:
                    a = _as_float(top.get(key))
                    b = _as_float(true_item.get(key))
                    if a is not None and b is not None:
                        deltas[key] = a - b
                true_compare = {
                    'true_rank_label': true_item.get('label'),
                    'score_delta_top_minus_true': deltas.get('score'),
                    'source_deltas_top_minus_true': {
                        key: value for key, value in deltas.items() if key != 'score'
                    },
                }
                positive = [
                    key for key, value in true_compare['source_deltas_top_minus_true'].items()
                    if value is not None and value > float(weak_signal_threshold)
                ]
                negative = [
                    key for key, value in true_compare['source_deltas_top_minus_true'].items()
                    if value is not None and value < -float(weak_signal_threshold)
                ]
                if positive:
                    reason_codes.append('wrong_label_supported_by_' + '+'.join(positive))
                if negative:
                    reason_codes.append('true_label_supported_by_' + '+'.join(negative))

    return {
        'top_label': top.get('label'),
        'runner_up_label': second.get('label') if second else None,
        'score_margin': margin,
        'needs_review': margin is None or margin < float(low_margin_threshold),
        'reason_codes': list(dict.fromkeys(reason_codes)),
        'source_margins': margins,
        'true_label_compare': true_compare,
    }
