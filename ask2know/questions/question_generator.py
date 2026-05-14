def _feature_gap(top_a, top_b, feature):
    return abs(float(top_a.get('detail', {}).get(feature, 0.0)) - float(top_b.get('detail', {}).get(feature, 0.0)))


def _feature_sentence(gaps):
    close = [k for k, v in gaps.items() if v < 0.05]
    different = [k for k, v in gaps.items() if v >= 0.12]
    parts = []
    if close:
        parts.append('、'.join(close) + ' 的分数很接近')
    if different:
        parts.append('、'.join(different) + ' 可能有一定区分作用')
    if not parts:
        parts.append('各项特征差异都不明显')
    return '；'.join(parts)


def generate_question_context(top_a, top_b, sample_path=None, true_label=None, phase='pre_confirm'):
    current = sample_path or '当前未标注图片'
    a = top_a.get('label') if isinstance(top_a, dict) else str(top_a)
    b = top_b.get('label') if isinstance(top_b, dict) else str(top_b)
    lines = []
    lines.append('【问题上下文】')
    lines.append(f'当前图片：{current}')
    if phase == 'post_error' and true_label:
        lines.append(f'系统预测：{a}')
        lines.append(f'用户确认真实类别：{true_label}')
        lines.append('当前状态：系统识别错误，下面的问题是在问“真实类别”和“系统误判类别”的主要差异。')
    else:
        lines.append(f'系统当前第一候选：{a}')
        lines.append(f'系统当前第二候选：{b}')
        lines.append('当前状态：真实类别尚未确认。请根据“当前这张图片”回答，不是单纯比较两个类别的理论区别。')
    return '\n'.join(lines)


def generate_natural_question(top_a, top_b, question, weights, sample_path=None, pairwise_manager=None):
    a = top_a['label']
    b = top_b['label']
    qid = question['id']
    feature = question.get('feature', '')

    gaps = {
        'color': _feature_gap(top_a, top_b, 'color'),
        'size': _feature_gap(top_a, top_b, 'size'),
        'contour': _feature_gap(top_a, top_b, 'contour'),
        'texture': _feature_gap(top_a, top_b, 'texture'),
    }

    evidence = _feature_sentence(gaps)
    pair_hint = ''
    specific_hint = ''
    if pairwise_manager is not None:
        pair_hint = pairwise_manager.suggest_pair_prompt(a, b)
        specific_hint = pairwise_manager.pair_specific_question_hint(a, b)
    score_gap = float(top_a['score']) - float(top_b['score'])
    context = generate_question_context(top_a, top_b, sample_path=sample_path, phase='pre_confirm')

    intro = (
        f'{context}\n\n'
        f'系统现在最纠结的是 {a} 和 {b}。'
        f'两者总分差距只有 {score_gap:.3f}，说明当前判断并不稳。'
        f'从特征上看：{evidence}。'
        + (pair_hint if pair_hint else '')
    )
    if specific_hint:
        intro += '历史经验提示：' + specific_hint

    if qid == 'Q_COLOR_IMPORTANCE':
        question_text = intro + '请判断：当前这张图片的颜色/色系是否足以帮助排除错误候选？'
    elif qid == 'Q_SIZE_RELIABILITY':
        question_text = intro + '这里要特别注意：单张图里的“大小”可能只是拍摄距离造成的。当前图片里的大小是否可靠？'
    elif qid == 'Q_CONTOUR_IMPORTANCE':
        question_text = intro + '请判断：当前这张图片的整体形状、长宽比例、圆润/弯曲程度是否能帮助区分？'
    elif qid == 'Q_TEXTURE_IMPORTANCE':
        question_text = intro + '请判断：当前这张图片的表面纹理、颗粒感、局部重复结构是否能帮助区分？'
    elif qid == 'Q_SAMPLE_QUALITY':
        question_text = intro + '如果你觉得系统候选明显不靠谱，请优先考虑当前图片是否主体清晰、背景是否干扰。'
    else:
        question_text = context + '\n\n' + question['template'].format(a=a, b=b)

    return {
        'question': question_text,
        'context': context,
        'evidence': evidence,
        'feature_gaps': gaps,
        'selected_feature': feature,
        'history_hint': specific_hint,
    }
