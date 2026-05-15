QUESTION_BANK = [
    {
        'id': 'Q_COLOR_IMPORTANCE',
        'feature': 'color',
        'kind': 'feature_importance',
        'template': '{a} 和 {b} 是否主要通过颜色区分？',
        'options': [
            ('A', '颜色差异明显，颜色很重要', {'increase': ['color'], 'decrease': []}),
            ('B', '颜色相似，不能主要靠颜色', {'increase': [], 'decrease': ['color']}),
            ('C', '不确定', {'increase': [], 'decrease': []}),
        ]
    },
    {
        'id': 'Q_SIZE_RELIABILITY',
        'feature': 'size',
        'kind': 'feature_reliability',
        'template': '当前数据里，图片中的大小是否能代表真实大小？',
        'options': [
            ('A', '可以，拍摄距离/比例基本一致，大小可以参考', {'increase': ['size'], 'decrease': []}),
            ('B', '不可以，拍摄距离不同，图片大小不可靠', {'increase': [], 'decrease': ['size']}),
            ('C', '只有同一张图中多个物体比较时才可靠', {'increase': [], 'decrease': ['size']}),
            ('D', '不确定', {'increase': [], 'decrease': []}),
        ]
    },
    {
        'id': 'Q_CONTOUR_IMPORTANCE',
        'feature': 'shape',
        'kind': 'feature_importance',
        'template': '{a} 和 {b} 是否主要通过轮廓/形状区分？',
        'options': [
            ('A', '轮廓差异明显，形状很重要', {'increase': ['shape'], 'decrease': []}),
            ('B', '轮廓差异不明显', {'increase': [], 'decrease': ['shape']}),
            ('C', '当前图片角度/遮挡影响轮廓判断', {'increase': [], 'decrease': ['shape']}),
            ('D', '不确定', {'increase': [], 'decrease': []}),
        ]
    },
    {
        'id': 'Q_TEXTURE_IMPORTANCE',
        'feature': 'texture',
        'kind': 'feature_importance',
        'template': '{a} 和 {b} 是否主要通过表面纹理区分？',
        'options': [
            ('A', '纹理差异明显，纹理很重要', {'increase': ['texture'], 'decrease': []}),
            ('B', '纹理差异不明显', {'increase': [], 'decrease': ['texture']}),
            ('C', '当前图片太糊，纹理不可靠', {'increase': [], 'decrease': ['texture']}),
            ('D', '不确定', {'increase': [], 'decrease': []}),
        ]
    },
    {
        'id': 'Q_TEXT_IMPORTANCE',
        'feature': 'text',
        'kind': 'feature_importance',
        'template': '{a} 和 {b} 是否主要通过文字/数字区域区分？',
        'options': [
            ('A', '文字/数字差异明显，文字特征很重要', {'increase': ['text'], 'decrease': []}),
            ('B', '文字/数字不稳定，不能主要靠文字特征', {'increase': [], 'decrease': ['text']}),
            ('C', '当前图片太糊或太小，文字特征不可靠', {'increase': [], 'decrease': ['text']}),
            ('D', '不确定', {'increase': [], 'decrease': []}),
        ]
    },
    {
        'id': 'Q_SIGN_IMPORTANCE',
        'feature': 'sign',
        'kind': 'feature_importance',
        'template': '{a} 和 {b} 是否主要通过箭头/禁止/标识符号区分？',
        'options': [
            ('A', '箭头/禁止/符号差异明显，标识特征很重要', {'increase': ['sign'], 'decrease': []}),
            ('B', '符号差异不稳定，不能主要靠标识特征', {'increase': [], 'decrease': ['sign']}),
            ('C', '当前图片角度/遮挡影响标识判断', {'increase': [], 'decrease': ['sign']}),
            ('D', '不确定', {'increase': [], 'decrease': []}),
        ]
    },
    {
        'id': 'Q_SAMPLE_QUALITY',
        'feature': 'quality',
        'kind': 'sample_quality',
        'template': '这张图片是否适合作为学习样本？',
        'options': [
            ('A', '适合，主体清晰，背景干扰少', {'increase': ['texture', 'shape'], 'decrease': []}),
            ('B', '不适合，太糊/遮挡/主体不明显', {'increase': [], 'decrease': ['texture', 'shape']}),
            ('C', '可以判断，但不要加入正式样本库', {'increase': [], 'decrease': []}),
            ('D', '不确定', {'increase': [], 'decrease': []}),
        ]
    },
]
