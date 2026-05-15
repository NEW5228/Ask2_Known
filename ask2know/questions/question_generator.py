from ask2know.concepts.basic_concepts import DISPLAY_NAMES, summarize_concepts


def _feature_gap(top_a, top_b, feature):
    a = top_a.get('group_detail', {}).get(
        feature,
        top_a.get('system_detail', {}).get(feature, top_a.get('detail', {}).get(feature, 0.0))
    )
    b = top_b.get('group_detail', {}).get(
        feature,
        top_b.get('system_detail', {}).get(feature, top_b.get('detail', {}).get(feature, 0.0))
    )
    return abs(float(a) - float(b))


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


def _selection_sentence(top_a, top_b, question):
    sel = question.get('_selection') or {}
    feature = sel.get('feature') or question.get('feature', '')
    if not feature:
        return ''
    a = top_a.get('label')
    b = top_b.get('label')
    a_score = float(sel.get('a_score', 0.0))
    b_score = float(sel.get('b_score', 0.0))
    gap = float(sel.get('gap', abs(a_score - b_score)))
    if question.get('kind') == 'sample_quality':
        return '这次优先问样本质量，是因为当前关键特征信号偏弱或过于饱和，继续比较类别差异可能会误导学习。'
    if gap >= 0.12:
        stronger = a if a_score >= b_score else b
        return f'这次优先问 {feature}，因为它当前明显更支持 {stronger}，需要确认这个差异是否真实可靠。'
    return f'这次优先问 {feature}，因为它在当前两个候选之间分不开，但对这组判断仍可能很关键。'


def _concept_observation_sentence(top_a, top_b):
    sample_concepts = top_a.get('concepts') or {}
    if not sample_concepts:
        return ''
    observed = summarize_concepts(sample_concepts, top_n=5, min_score=0.38)
    if not observed:
        return ''
    obs_text = '、'.join([f'{x["name"]}({x["score"]:.2f})' for x in observed])

    a = top_a.get('label')
    b = top_b.get('label')
    a_concepts = top_a.get('class_concepts') or {}
    b_concepts = top_b.get('class_concepts') or {}
    hints = []
    for cname, sample_score in sorted(sample_concepts.items(), key=lambda x: x[1], reverse=True):
        av = float(a_concepts.get(cname, 0.0))
        bv = float(b_concepts.get(cname, 0.0))
        if max(av, bv, float(sample_score)) < 0.35:
            continue
        if abs(av - bv) >= 0.18:
            stronger = a if av > bv else b
            hints.append(f'{DISPLAY_NAMES.get(cname, cname)}更接近 {stronger} 的已确认经验')
        if len(hints) >= 2:
            break
    if hints:
        return f'从基础概念看，我观察到：{obs_text}。其中' + '，'.join(hints) + '。'
    return f'从基础概念看，我观察到：{obs_text}。'


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

    enabled_features = list(top_a.get('group_detail', {}).keys())
    gaps = {}
    for name in enabled_features:
        if name in top_a.get('group_detail', {}) or name in top_b.get('group_detail', {}):
            gaps[name] = _feature_gap(top_a, top_b, name)

    evidence = _feature_sentence(gaps)
    selected_reason = _selection_sentence(top_a, top_b, question)
    concept_evidence = _concept_observation_sentence(top_a, top_b)
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
        + (concept_evidence if concept_evidence else '')
        + (selected_reason if selected_reason else '')
        + (pair_hint if pair_hint else '')
    )
    if specific_hint:
        intro += '历史经验提示：' + specific_hint

    if qid == 'Q_COLOR_IMPORTANCE':
        question_text = intro + '请你帮我确认：这里真正可靠的是主体本身的颜色，还是光线/背景造成的颜色相似？'
    elif qid == 'Q_SIZE_RELIABILITY':
        question_text = intro + '这里要特别注意：单张图里的“大小”可能只是拍摄距离造成的。当前图片里的大小是否可靠？'
    elif qid == 'Q_CONTOUR_IMPORTANCE':
        question_text = intro + '请你帮我确认：当前主体的整体形状概念，比如圆形、长条形、上窄下宽，是否比颜色更能区分？'
    elif qid == 'Q_TEXTURE_IMPORTANCE':
        question_text = intro + '请你帮我确认：这里的纹理、颗粒感、重复结构或局部图案，是否是区分类别的关键？'
    elif qid == 'Q_TEXT_IMPORTANCE':
        question_text = intro + '请你帮我确认：这里的文字、数字或字符笔画区域，是否是区分类别的关键？'
    elif qid == 'Q_SIGN_IMPORTANCE':
        question_text = intro + '请你帮我确认：这里的箭头、禁止符号、方向性结构或标识图案，是否是区分类别的关键？'
    elif qid == 'Q_SAMPLE_QUALITY':
        question_text = intro + '如果你觉得系统候选明显不靠谱，请优先考虑当前图片是否主体清晰、背景是否干扰。'
    else:
        question_text = context + '\n\n' + question['template'].format(a=a, b=b)

    return {
        'question': question_text,
        'context': context,
        'evidence': evidence,
        'concept_evidence': concept_evidence,
        'feature_gaps': gaps,
        'sample_concepts': top_a.get('concepts', {}),
        'top_a_class_concepts': top_a.get('class_concepts', {}),
        'top_b_class_concepts': top_b.get('class_concepts', {}),
        'selected_feature': feature,
        'selected_reason': selected_reason,
        'selection_debug': question.get('_selection', {}),
        'history_hint': specific_hint,
    }
