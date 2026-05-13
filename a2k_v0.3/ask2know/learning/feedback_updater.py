def apply_answer_to_weights(adaptive_weights, question, selected_key):
    selected = None
    for key, text, action in question['options']:
        if key.upper() == selected_key.upper():
            selected = (key, text, action)
            break
    if selected is None:
        return None, None, None
    key, text, action = selected
    before, after = adaptive_weights.update(action.get('increase', []), action.get('decrease', []))
    return text, before, after

def update_question_reward(question_weights, question_id, was_helpful):
    old = question_weights.get(question_id, 1.0)
    new = min(2.0, old + 0.1) if was_helpful else max(0.2, old - 0.1)
    question_weights[question_id] = new
    return old, new
