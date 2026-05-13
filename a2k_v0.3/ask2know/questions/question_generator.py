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


def generate_natural_question(top_a, top_b, question, weights, sample_path=None):
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
    score_gap = float(top_a['score']) - float(top_b['score'])

    intro = (
        f'系统现在最纠结的是 {a} 和 {b}。'
        f'两者总分差距只有 {score_gap:.3f}，说明当前判断并不稳。'
        f'从特征上看：{evidence}。'
    )

    if qid == 'Q_COLOR_IMPORTANCE':
        question_text = (
            intro
            + f'当前 {a} 和 {b} 在颜色上可能容易混淆。'
            + '你认为这个任务里，颜色能不能作为主要判断依据？'
        )
    elif qid == 'Q_SIZE_RELIABILITY':
        question_text = (
            intro
            + '这里要特别注意：单独图片里的“大小”可能只是拍摄距离造成的，不一定代表真实大小。'
            + '请判断当前数据里，图片中的大小是否可靠。'
        )
    elif qid == 'Q_CONTOUR_IMPORTANCE':
        question_text = (
            intro
            + '系统想确认：整体轮廓、形状、圆润程度这类信息，是否适合用来区分这两个对象？'
        )
    elif qid == 'Q_TEXTURE_IMPORTANCE':
        question_text = (
            intro
            + '系统想确认：表面纹理、颗粒感、边缘变化这些信息，是否比颜色或大小更有用？'
        )
    elif qid == 'Q_SAMPLE_QUALITY':
        question_text = (
            intro
            + '由于各项特征都没有明显拉开，系统怀疑这张图可能不是很适合作为学习样本。'
            + '请判断它是否清晰、主体是否明显、是否值得加入学习。'
        )
    else:
        question_text = question['template'].format(a=a, b=b)

    return {
        'question': question_text,
        'evidence': evidence,
        'feature_gaps': gaps,
        'selected_feature': feature,
    }
