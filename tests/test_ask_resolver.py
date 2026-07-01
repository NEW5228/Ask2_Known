from ask2know.questions.ask_resolver import (
    DEFAULT_ASK_CANDIDATE_TOP_K,
    DEFAULT_ASK_MAX_OPTIONS,
    DEFAULT_ASK_MAX_QUESTIONS,
    apply_dynamic_answer_to_predictions,
    apply_taxonomy_answer_to_predictions,
    build_runtime_dynamic_question,
    build_runtime_taxonomy_question,
    select_ask_candidates,
    summarize_ask_resolution,
)


def _pred(label, score, path):
    return {'label': label, 'score': score, 'taxonomy_path': path}


def test_runtime_taxonomy_question_filters_selected_branch():
    predictions = [
        _pred('turn_right_ahead', 0.92, ['traffic_sign', 'mandatory', 'circle', 'blue', 'arrow', 'arrow_right', 'turn_right_ahead']),
        _pred('ahead_only', 0.90, ['traffic_sign', 'mandatory', 'circle', 'blue', 'arrow', 'arrow_straight', 'ahead_only']),
        _pred('keep_left', 0.88, ['traffic_sign', 'mandatory', 'circle', 'blue', 'arrow', 'arrow_keep_left', 'keep_left']),
    ]
    question, generated = build_runtime_taxonomy_question(predictions)
    assert question is not None
    assert question['kind'] == 'taxonomy_resolution'
    assert generated['question']

    straight_option = next(
        action
        for _, text, action in question['options']
        if text.startswith('arrow_straight')
    )
    reranked, matched = apply_taxonomy_answer_to_predictions(predictions, straight_option['path_prefix'])
    assert matched
    assert reranked[0]['label'] == 'ahead_only'
    assert reranked[0]['ask_resolution_gate_reason'] == 'selected_taxonomy_branch'


def test_summarize_ask_resolution_counts_one_question_conversion():
    samples = [{
        'true_label': 'ahead_only',
        'top_predictions': [
            _pred('turn_right_ahead', 0.92, ['traffic_sign', 'mandatory', 'circle', 'blue', 'arrow', 'arrow_right', 'turn_right_ahead']),
            _pred('ahead_only', 0.90, ['traffic_sign', 'mandatory', 'circle', 'blue', 'arrow', 'arrow_straight', 'ahead_only']),
        ],
    }]
    summary = summarize_ask_resolution(samples, top_k=5)
    assert summary['raw_correct_count'] == 0
    assert summary['converted_count'] == 1
    assert summary['ask_correct_count'] == 1


def test_select_ask_candidates_limits_to_top10():
    predictions = [
        _pred(f'label_{idx}', 1.0 - idx * 0.01, ['root', f'branch_{idx}', f'label_{idx}'])
        for idx in range(12)
    ]

    candidates = select_ask_candidates(predictions)

    assert len(candidates) == DEFAULT_ASK_CANDIDATE_TOP_K
    assert candidates[0]['label'] == 'label_0'
    assert candidates[-1]['label'] == 'label_9'


def test_summarize_ask_resolution_defaults_to_top10_8_options_2_questions():
    samples = [{
        'true_label': 'label_9',
        'top_predictions': [
            _pred(f'label_{idx}', 1.0 - idx * 0.01, ['root', f'branch_{idx}', f'label_{idx}'])
            for idx in range(12)
        ],
    }]

    summary = summarize_ask_resolution(samples)

    assert summary['top_k'] == DEFAULT_ASK_CANDIDATE_TOP_K
    assert summary['ask_candidate_top_k'] == DEFAULT_ASK_CANDIDATE_TOP_K
    assert summary['max_options'] == DEFAULT_ASK_MAX_OPTIONS
    assert summary['max_questions'] == DEFAULT_ASK_MAX_QUESTIONS
    assert summary['candidate_contains_true_count'] == 1


def test_runtime_dynamic_question_reranks_selected_candidate():
    predictions = [
        {'label': 'birman', 'score': 0.91, 'knn_score': 0.90},
        {'label': 'ragdoll', 'score': 0.90, 'knn_score': 0.92},
        {'label': 'persian', 'score': 0.86, 'concept_score': 0.88},
    ]

    question, generated = build_runtime_dynamic_question(predictions)

    assert question is not None
    assert question['kind'] == 'dynamic_disambiguation'
    assert generated['question']
    ragdoll_action = next(
        action
        for _, text, action in question['options']
        if 'ragdoll' in text
    )
    reranked, matched = apply_dynamic_answer_to_predictions(
        predictions,
        ragdoll_action['labels'],
        score_bonus=ragdoll_action['score_bonus'],
    )
    assert matched
    assert reranked[0]['label'] == 'ragdoll'
    assert reranked[0]['ask_resolution_gate_reason'] == 'selected_dynamic_candidate'


def test_summarize_ask_resolution_dynamic_converts_candidate_error():
    samples = [{
        'true_label': 'ragdoll',
        'top_predictions': [
            {'label': 'birman', 'score': 0.91, 'knn_score': 0.90},
            {'label': 'ragdoll', 'score': 0.90, 'knn_score': 0.92},
        ],
    }]

    summary = summarize_ask_resolution(samples, top_k=5, question_mode='dynamic')

    assert summary['question_mode'] == 'dynamic'
    assert summary['raw_correct_count'] == 0
    assert summary['asked_count'] == 1
    assert summary['converted_count'] == 1
    assert summary['ask_correct_count'] == 1
