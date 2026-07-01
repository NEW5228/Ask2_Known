from collections import Counter

DEFAULT_ASK_CANDIDATE_TOP_K = 10
DEFAULT_ASK_MAX_OPTIONS = 8
DEFAULT_ASK_MAX_QUESTIONS = 2
DEFAULT_DYNAMIC_SCORE_BONUS = 1.0


def _path(prediction):
    return [str(item) for item in (prediction.get('taxonomy_path') or [])]


def _prefix(path, level_index):
    if not path or level_index < 0:
        return ()
    return tuple(path[:min(len(path), level_index + 1)])


def select_ask_candidates(predictions, *, candidate_top_k=DEFAULT_ASK_CANDIDATE_TOP_K):
    """Keep ASK resolution bounded to a small candidate set."""
    limit = max(1, int(candidate_top_k or DEFAULT_ASK_CANDIDATE_TOP_K))
    return list(predictions or [])[:limit]


def suggest_taxonomy_question(
    predictions,
    *,
    max_options=DEFAULT_ASK_MAX_OPTIONS,
    min_options=2,
    candidate_top_k=None,
):
    """Pick one taxonomy level that best separates the current candidates."""
    if candidate_top_k is not None:
        predictions = select_ask_candidates(predictions, candidate_top_k=candidate_top_k)
    preds = [item for item in (predictions or []) if _path(item)]
    if len(preds) < 2:
        return None
    max_depth = max(len(_path(item)) for item in preds)
    best = None
    # Do not ask the leaf class directly; ASK should ask an attribute/field question.
    for level_index in range(1, max(1, max_depth - 1)):
        groups = {}
        for item in preds:
            path = _path(item)
            if level_index >= len(path):
                continue
            key = _prefix(path, level_index)
            group = groups.setdefault(key, {
                'node': path[level_index],
                'path_prefix': list(key),
                'labels': [],
                'best_score': None,
            })
            group['labels'].append(item.get('label'))
            score = float(item.get('score', 0.0))
            group['best_score'] = score if group['best_score'] is None else max(group['best_score'], score)
        if not (min_options <= len(groups) <= max_options):
            continue
        counts = Counter()
        for item in preds:
            path = _path(item)
            if level_index < len(path):
                counts[path[level_index]] += 1
        singletons = sum(1 for count in counts.values() if count == 1)
        option_count = len(groups)
        top_prefix = _prefix(_path(preds[0]), level_index)
        top_group_size = len(groups.get(top_prefix, {}).get('labels') or [])
        score = (
            level_index * 10.0
            + option_count * 2.0
            + singletons
            - max(0, top_group_size - 2) * 0.5
        )
        if best is None or score > best['selection_score']:
            options = sorted(
                groups.values(),
                key=lambda item: float(item.get('best_score') or 0.0),
                reverse=True,
            )
            best = {
                'type': 'taxonomy_disambiguation',
                'level_index': level_index,
                'level_name': f'level_{level_index}',
                'question': '请选择当前图片更符合哪个分层属性。',
                'options': options,
                'selection_score': score,
            }
    return best


def build_runtime_taxonomy_question(
    predictions,
    *,
    max_options=DEFAULT_ASK_MAX_OPTIONS,
    candidate_top_k=DEFAULT_ASK_CANDIDATE_TOP_K,
):
    candidates = select_ask_candidates(predictions, candidate_top_k=candidate_top_k)
    question = suggest_taxonomy_question(candidates, max_options=max_options)
    if not question:
        return None, None
    keys = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    options = []
    for idx, option in enumerate(question.get('options') or []):
        if idx >= len(keys):
            break
        labels = [str(label) for label in option.get('labels', []) if label]
        node = str(option.get('node') or '')
        label_hint = ', '.join(labels[:3])
        text = node if not label_hint else f'{node} ({label_hint})'
        options.append((
            keys[idx],
            text,
            {
                'kind': 'taxonomy_resolution',
                'path_prefix': list(option.get('path_prefix') or []),
                'node': node,
                'labels': labels,
            },
        ))
    if len(options) < 2:
        return None, None
    level_index = int(question.get('level_index', 0))
    pending_question = {
        'id': f'ASK_TAXONOMY_LEVEL_{level_index}',
        'kind': 'taxonomy_resolution',
        'feature': f'taxonomy_level_{level_index}',
        'options': options,
        '_selection': {
            'feature': f'taxonomy_level_{level_index}',
            'level_index': level_index,
            'score': question.get('selection_score', 0.0),
        },
    }
    generated = {
        'question': '我需要确认当前图片更符合哪个分层属性。请选择最接近当前图片的选项。',
        'context': '',
        'evidence': '这些候选在同一层级属性上分歧明显，回答后系统会只在对应分支内重新排序。',
        'concept_evidence': '',
        'selected_feature': pending_question['feature'],
        'selected_reason': '候选类别的 taxonomy path 在这一层开始分开。',
        'selection_debug': pending_question['_selection'],
    }
    return pending_question, generated


def _display_label(label):
    return str(label or '').replace('_', ' ')


def _best_source_hint(prediction):
    sources = [
        ('nearest samples', prediction.get('knn_score')),
        ('prototype memory', prediction.get('prototype_score')),
        ('local crop evidence', prediction.get('crop_rerank_score')),
        ('pairwise experience', prediction.get('pairwise_score')),
        ('text semantics', prediction.get('text_semantic_score')),
        ('concept evidence', prediction.get('concept_score')),
        ('subprototype memory', prediction.get('subprototype_score')),
    ]
    scored = [
        (name, float(value))
        for name, value in sources
        if value is not None
    ]
    if not scored:
        return 'overall visual similarity'
    name, value = max(scored, key=lambda item: item[1])
    return f'{name} {value:.3f}'


def suggest_dynamic_question(
    predictions,
    *,
    max_options=DEFAULT_ASK_MAX_OPTIONS,
    min_options=2,
    candidate_top_k=None,
):
    """Build a candidate-specific question without requiring a predefined taxonomy."""
    if candidate_top_k is not None:
        predictions = select_ask_candidates(predictions, candidate_top_k=candidate_top_k)
    preds = [item for item in (predictions or []) if item.get('label')]
    if len(preds) < min_options:
        return None
    labels_seen = set()
    options = []
    for item in preds:
        label = str(item.get('label'))
        if label in labels_seen:
            continue
        labels_seen.add(label)
        if len(options) >= max(1, int(max_options)):
            break
        score = float(item.get('score', 0.0))
        options.append({
            'node': label,
            'labels': [label],
            'best_score': score,
            'hint': _best_source_hint(item),
        })
    if len(options) < min_options:
        return None
    top = preds[0]
    runner_up = preds[1] if len(preds) > 1 else {}
    margin = float(top.get('score', 0.0)) - float(runner_up.get('score', 0.0))
    return {
        'type': 'dynamic_disambiguation',
        'question': '请选择当前图片最接近哪一种候选外观。',
        'options': options,
        'selection_score': max(0.0, 1.0 - margin),
        'score_margin': margin,
    }


def build_runtime_dynamic_question(
    predictions,
    *,
    max_options=DEFAULT_ASK_MAX_OPTIONS,
    candidate_top_k=DEFAULT_ASK_CANDIDATE_TOP_K,
):
    candidates = select_ask_candidates(predictions, candidate_top_k=candidate_top_k)
    question = suggest_dynamic_question(candidates, max_options=max_options)
    if not question:
        return None, None
    keys = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    options = []
    for idx, option in enumerate(question.get('options') or []):
        if idx >= len(keys) - 1:
            break
        labels = [str(label) for label in option.get('labels', []) if label]
        if not labels:
            continue
        label = labels[0]
        text = f'更像 {_display_label(label)}（依据：{option.get("hint", "visual similarity")}）'
        options.append((
            keys[idx],
            text,
            {
                'kind': 'dynamic_disambiguation',
                'labels': labels,
                'score_bonus': DEFAULT_DYNAMIC_SCORE_BONUS,
            },
        ))
    unknown_key = keys[len(options)] if len(options) < len(keys) else 'Z'
    options.append((
        unknown_key,
        '不确定，保留当前排序',
        {
            'kind': 'dynamic_disambiguation',
            'labels': [],
            'score_bonus': 0.0,
            'no_change': True,
        },
    ))
    if len(options) < 3:
        return None, None
    pending_question = {
        'id': 'ASK_DYNAMIC_CANDIDATE',
        'kind': 'dynamic_disambiguation',
        'feature': 'dynamic_candidate',
        'options': options,
        '_selection': {
            'feature': 'dynamic_candidate',
            'score': question.get('selection_score', 0.0),
            'score_margin': question.get('score_margin', 0.0),
        },
    }
    generated = {
        'question': (
            '我会根据当前图片和候选类别动态追问。'
            '请选择最接近这张图的候选外观；如果看不出来就选“不确定”。'
        ),
        'context': '',
        'evidence': (
            '当前没有可用的固定分层问题。'
            '系统改为围绕本次 top-k 候选生成候选级区分问题，回答后只调整本次候选排序。'
        ),
        'concept_evidence': '',
        'selected_feature': pending_question['feature'],
        'selected_reason': '候选分数接近，且没有 taxonomy 问题可用。',
        'selection_debug': pending_question['_selection'],
    }
    return pending_question, generated


def apply_taxonomy_answer_to_predictions(predictions, path_prefix):
    prefix = tuple(str(item) for item in (path_prefix or []))
    if not prefix:
        return list(predictions or []), []
    matched = []
    unmatched = []
    for item in predictions or []:
        path = tuple(_path(item)[:len(prefix)])
        if path == prefix:
            copied = dict(item)
            copied['ask_resolution_delta'] = 1.0
            copied['ask_resolution_gate_reason'] = 'selected_taxonomy_branch'
            matched.append(copied)
        else:
            copied = dict(item)
            copied['ask_resolution_delta'] = 0.0
            copied['ask_resolution_gate_reason'] = 'outside_selected_taxonomy_branch'
            unmatched.append(copied)
    return matched + unmatched, matched


def apply_dynamic_answer_to_predictions(predictions, labels, *, score_bonus=DEFAULT_DYNAMIC_SCORE_BONUS):
    selected_labels = {str(label) for label in (labels or []) if label}
    if not selected_labels:
        copied = []
        for item in predictions or []:
            row = dict(item)
            row['ask_resolution_delta'] = 0.0
            row['ask_resolution_gate_reason'] = 'dynamic_no_change'
            copied.append(row)
        return copied, copied
    matched = []
    unmatched = []
    for item in predictions or []:
        row = dict(item)
        if str(row.get('label')) in selected_labels:
            row['ask_resolution_delta'] = float(score_bonus)
            row['ask_resolution_gate_reason'] = 'selected_dynamic_candidate'
            row['score'] = float(row.get('score', 0.0)) + float(score_bonus)
            matched.append(row)
        else:
            row['ask_resolution_delta'] = 0.0
            row['ask_resolution_gate_reason'] = 'outside_dynamic_candidate'
            unmatched.append(row)
    reranked = sorted(matched + unmatched, key=lambda item: float(item.get('score', 0.0)), reverse=True)
    return reranked, matched


def simulate_taxonomy_answer(predictions, true_label, question):
    """Simulate an oracle user answer using the true label inside top-k predictions."""
    if not question:
        return {
            'asked': False,
            'resolved_label': (predictions or [{}])[0].get('label'),
            'correct': False,
            'reason': 'no_question',
        }
    preds = list(predictions or [])
    true_item = next((item for item in preds if item.get('label') == true_label), None)
    if true_item is None:
        return {
            'asked': True,
            'resolved_label': preds[0].get('label') if preds else None,
            'correct': False,
            'reason': 'true_label_not_in_candidates',
            'answer_prefix': None,
        }
    level_index = int(question.get('level_index', 0))
    answer_prefix = _prefix(_path(true_item), level_index)
    filtered = [
        item for item in preds
        if _prefix(_path(item), level_index) == answer_prefix
    ]
    resolved = filtered[0] if filtered else (preds[0] if preds else {})
    return {
        'asked': True,
        'resolved_label': resolved.get('label'),
        'correct': resolved.get('label') == true_label,
        'reason': 'answered',
        'answer_prefix': list(answer_prefix),
        'remaining_labels': [item.get('label') for item in filtered],
    }


def simulate_taxonomy_dialog(
    predictions,
    true_label,
    *,
    max_questions=DEFAULT_ASK_MAX_QUESTIONS,
    max_options=DEFAULT_ASK_MAX_OPTIONS,
    candidate_top_k=DEFAULT_ASK_CANDIDATE_TOP_K,
):
    preds = select_ask_candidates(predictions, candidate_top_k=candidate_top_k)
    raw_label = preds[0].get('label') if preds else None
    if raw_label == true_label:
        return {
            'asked': 0,
            'resolved_label': raw_label,
            'correct': True,
            'reason': 'already_correct',
            'steps': [],
            'remaining_labels': [item.get('label') for item in preds],
        }
    current = preds
    steps = []
    asked = 0
    for _ in range(max(1, int(max_questions))):
        question = suggest_taxonomy_question(current, max_options=max_options)
        if not question:
            break
        result = simulate_taxonomy_answer(current, true_label, question)
        asked += 1 if result.get('asked') else 0
        steps.append({
            'level_index': question.get('level_index'),
            'options': [
                {'node': item.get('node'), 'labels': item.get('labels', [])}
                for item in question.get('options', [])
            ],
            'answer_prefix': result.get('answer_prefix'),
            'reason': result.get('reason'),
            'resolved_label': result.get('resolved_label'),
            'remaining_labels': result.get('remaining_labels', []),
        })
        if result.get('reason') == 'true_label_not_in_candidates':
            return {
                'asked': asked,
                'resolved_label': result.get('resolved_label'),
                'correct': False,
                'reason': 'true_label_not_in_candidates',
                'steps': steps,
                'remaining_labels': result.get('remaining_labels', []),
            }
        answer_prefix = tuple(result.get('answer_prefix') or [])
        if not answer_prefix:
            break
        current = [
            item for item in current
            if _prefix(_path(item), len(answer_prefix) - 1) == answer_prefix
        ]
        if not current:
            break
        if current[0].get('label') == true_label:
            return {
                'asked': asked,
                'resolved_label': current[0].get('label'),
                'correct': True,
                'reason': 'answered',
                'steps': steps,
                'remaining_labels': [item.get('label') for item in current],
            }
        if len(current) == 1:
            break
    resolved = current[0].get('label') if current else raw_label
    return {
        'asked': asked,
        'resolved_label': resolved,
        'correct': resolved == true_label,
        'reason': 'answered' if asked else 'no_question',
        'steps': steps,
        'remaining_labels': [item.get('label') for item in current],
    }


def simulate_dynamic_dialog(
    predictions,
    true_label,
    *,
    max_questions=DEFAULT_ASK_MAX_QUESTIONS,
    max_options=DEFAULT_ASK_MAX_OPTIONS,
    candidate_top_k=DEFAULT_ASK_CANDIDATE_TOP_K,
):
    preds = select_ask_candidates(predictions, candidate_top_k=candidate_top_k)
    raw_label = preds[0].get('label') if preds else None
    if raw_label == true_label:
        return {
            'asked': 0,
            'resolved_label': raw_label,
            'correct': True,
            'reason': 'already_correct',
            'steps': [],
            'remaining_labels': [item.get('label') for item in preds],
        }
    current = preds
    steps = []
    asked = 0
    for _ in range(max(1, int(max_questions))):
        question = suggest_dynamic_question(
            current,
            max_options=max_options,
            candidate_top_k=candidate_top_k,
        )
        if not question:
            break
        asked += 1
        options = question.get('options') or []
        selected = next(
            (item for item in options if true_label in set(item.get('labels') or [])),
            None,
        )
        steps.append({
            'level_index': 'dynamic',
            'options': [
                {'node': item.get('node'), 'labels': item.get('labels', [])}
                for item in options
            ],
            'answer_prefix': None,
            'answer_labels': selected.get('labels') if selected else [],
            'reason': 'answered' if selected else 'true_label_not_in_options',
            'resolved_label': (selected.get('labels') or [raw_label])[0] if selected else raw_label,
            'remaining_labels': selected.get('labels') if selected else [item.get('label') for item in current],
        })
        if selected is None:
            return {
                'asked': asked,
                'resolved_label': raw_label,
                'correct': False,
                'reason': 'true_label_not_in_options',
                'steps': steps,
                'remaining_labels': [item.get('label') for item in current],
            }
        labels = set(selected.get('labels') or [])
        current = [item for item in current if item.get('label') in labels]
        resolved = current[0].get('label') if current else (selected.get('labels') or [raw_label])[0]
        return {
            'asked': asked,
            'resolved_label': resolved,
            'correct': resolved == true_label,
            'reason': 'answered',
            'steps': steps,
            'remaining_labels': [item.get('label') for item in current],
        }
    resolved = current[0].get('label') if current else raw_label
    return {
        'asked': asked,
        'resolved_label': resolved,
        'correct': resolved == true_label,
        'reason': 'answered' if asked else 'no_question',
        'steps': steps,
        'remaining_labels': [item.get('label') for item in current],
    }


def summarize_ask_resolution(
    samples,
    *,
    top_k=DEFAULT_ASK_CANDIDATE_TOP_K,
    max_questions=DEFAULT_ASK_MAX_QUESTIONS,
    max_options=DEFAULT_ASK_MAX_OPTIONS,
    candidate_top_k=DEFAULT_ASK_CANDIDATE_TOP_K,
    question_mode='taxonomy',
):
    total = len(samples or [])
    raw_correct = 0
    topk_contains_true = 0
    candidate_contains_true = 0
    ask_correct = 0
    asked = 0
    converted = 0
    hurt = 0
    reason_counts = Counter()
    level_counts = Counter()
    examples = []
    for sample in samples or []:
        true_label = sample.get('true_label')
        preds = list(sample.get('top_predictions') or [])[:max(1, int(top_k))]
        candidates = select_ask_candidates(preds, candidate_top_k=candidate_top_k)
        raw_label = preds[0].get('label') if preds else None
        raw_ok = raw_label == true_label
        raw_correct += 1 if raw_ok else 0
        labels = [item.get('label') for item in preds]
        candidate_labels = [item.get('label') for item in candidates]
        in_topk = true_label in labels
        in_candidates = true_label in candidate_labels
        topk_contains_true += 1 if in_topk else 0
        candidate_contains_true += 1 if in_candidates else 0
        if question_mode == 'dynamic':
            result = simulate_dynamic_dialog(
                preds,
                true_label,
                max_questions=max_questions,
                max_options=max_options,
                candidate_top_k=candidate_top_k,
            )
        elif question_mode == 'auto':
            result = simulate_taxonomy_dialog(
                preds,
                true_label,
                max_questions=max_questions,
                max_options=max_options,
                candidate_top_k=candidate_top_k,
            )
            if result.get('reason') == 'no_question':
                result = simulate_dynamic_dialog(
                    preds,
                    true_label,
                    max_questions=max_questions,
                    max_options=max_options,
                    candidate_top_k=candidate_top_k,
                )
        else:
            result = simulate_taxonomy_dialog(
                preds,
                true_label,
                max_questions=max_questions,
                max_options=max_options,
                candidate_top_k=candidate_top_k,
            )
        asked += int(result.get('asked') or 0)
        for step in result.get('steps') or []:
            level_counts[step.get('level_index')] += 1
        reason_counts[result.get('reason')] += 1
        ask_ok = bool(result.get('correct'))
        ask_correct += 1 if ask_ok else 0
        if ask_ok and not raw_ok:
            converted += 1
        if raw_ok and not ask_ok:
            hurt += 1
        if len(examples) < 20 and (not raw_ok or result.get('asked')):
            examples.append({
                'path': sample.get('path'),
                'true_label': true_label,
                'raw_label': raw_label,
                'resolved_label': result.get('resolved_label'),
                'raw_rank': (labels.index(true_label) + 1) if in_topk else None,
                'candidate_rank': (candidate_labels.index(true_label) + 1) if in_candidates else None,
                'steps': result.get('steps', []),
                'remaining_labels': result.get('remaining_labels', []),
                'reason': result.get('reason'),
            })
    return {
        'schema_version': 'ask_resolution_summary_v1',
        'total': total,
        'raw_correct_count': raw_correct,
        'raw_accuracy': raw_correct / max(1, total),
        'top_k': int(top_k),
        'ask_candidate_top_k': int(candidate_top_k),
        'max_questions': int(max_questions),
        'max_options': int(max_options),
        'question_mode': question_mode,
        'top_k_contains_true_count': topk_contains_true,
        'top_k_contains_true_accuracy': topk_contains_true / max(1, total),
        'candidate_contains_true_count': candidate_contains_true,
        'candidate_contains_true_accuracy': candidate_contains_true / max(1, total),
        'ask_correct_count': ask_correct,
        'ask_accuracy_after_one_question': ask_correct / max(1, total),
        'ask_accuracy_after_questions': ask_correct / max(1, total),
        'asked_count': asked,
        'converted_count': converted,
        'hurt_count': hurt,
        'reason_counts': dict(reason_counts.most_common()),
        'question_level_counts': dict(level_counts.most_common()),
        'examples': examples,
    }
