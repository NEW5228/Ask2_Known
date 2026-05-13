QUESTION_BANK = [
    {
        'id': 'Q_COLOR_IMPORTANCE',
        'feature': 'color',
        'template': '{a} 和 {b} 是否主要通过颜色区分？',
        'options': [
            ('A', '颜色差异明显，颜色很重要', {'increase': ['color'], 'decrease': []}),
            ('B', '颜色相似，不能主要靠颜色', {'increase': [], 'decrease': ['color']}),
            ('C', '不确定', {'increase': [], 'decrease': []}),
        ]
    },
    {
        'id': 'Q_SIZE_IMPORTANCE',
        'feature': 'size',
        'template': '{a} 和 {b} 是否主要通过大小/体积区分？',
        'options': [
            ('A', '{a} 通常更大', {'increase': ['size'], 'decrease': []}),
            ('B', '{b} 通常更大', {'increase': ['size'], 'decrease': []}),
            ('C', '大小差异不明显', {'increase': [], 'decrease': ['size']}),
            ('D', '不确定', {'increase': [], 'decrease': []}),
        ]
    },
    {
        'id': 'Q_CONTOUR_IMPORTANCE',
        'feature': 'contour',
        'template': '{a} 和 {b} 是否主要通过轮廓/形状区分？',
        'options': [
            ('A', '轮廓差异明显，形状很重要', {'increase': ['contour'], 'decrease': []}),
            ('B', '轮廓差异不明显', {'increase': [], 'decrease': ['contour']}),
            ('C', '不确定', {'increase': [], 'decrease': []}),
        ]
    },
    {
        'id': 'Q_TEXTURE_IMPORTANCE',
        'feature': 'texture',
        'template': '{a} 和 {b} 是否主要通过表面纹理区分？',
        'options': [
            ('A', '纹理差异明显，纹理很重要', {'increase': ['texture'], 'decrease': []}),
            ('B', '纹理差异不明显', {'increase': [], 'decrease': ['texture']}),
            ('C', '不确定', {'increase': [], 'decrease': []}),
        ]
    },
]
