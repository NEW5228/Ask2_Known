import math
import numpy as np


CONCEPT_NAMES = [
    'red',
    'orange',
    'yellow',
    'green',
    'blue',
    'purple',
    'color_family',
    'dark',
    'bright',
    'round',
    'elongated',
    'pear_like',
    'rectangular_like',
    'smooth_surface',
    'texture_rich',
    'edge_rich',
    'single_object',
    'cluster_like',
    'repeated_parts',
    'clear_foreground',
    'background_interference',
]

DISPLAY_NAMES = {
    'red': '偏红',
    'orange': '偏橙',
    'yellow': '偏黄',
    'green': '偏绿',
    'blue': '偏蓝',
    'purple': '偏紫',
    'color_family': '颜色类别明显',
    'dark': '偏暗',
    'bright': '偏亮',
    'round': '接近圆形',
    'elongated': '长条形',
    'pear_like': '上窄下宽',
    'rectangular_like': '接近矩形',
    'smooth_surface': '表面较平滑',
    'texture_rich': '纹理较丰富',
    'edge_rich': '边缘/局部线条明显',
    'single_object': '更像单体',
    'cluster_like': '有聚集感',
    'repeated_parts': '有重复结构',
    'clear_foreground': '主体较清晰',
    'background_interference': '背景/主体干扰较强',
}


def _clip01(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _safe_array(value):
    if value is None:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(value, dtype=np.float32).flatten()


def _concept_dict():
    return {name: 0.0 for name in CONCEPT_NAMES}


def concepts_from_features(features):
    """Map shallow visual features to reusable, human-nameable concepts.

    This is intentionally lightweight and generic. It is not a final truth
    source; it gives Ask2Know a first concept vocabulary that can be refined by
    user feedback and future modules.
    """
    concepts = _concept_dict()

    color = _safe_array(features.get('color'))
    if color.size >= 8:
        red, orange, yellow, green, blue, purple, dark, bright = color[-8:]
        concepts.update({
            'red': _clip01(red),
            'orange': _clip01(orange),
            'yellow': _clip01(yellow),
            'green': _clip01(green),
            'blue': _clip01(blue),
            'purple': _clip01(purple),
            'color_family': _clip01(max(red, orange, yellow, green, blue, purple)),
            'dark': _clip01(dark),
            'bright': _clip01(bright),
        })

    contour = _safe_array(features.get('contour'))
    if contour.size >= 15:
        circularity = _clip01(contour[0])
        extent = _clip01(contour[1])
        solidity = _clip01(contour[2])
        aspect = max(1e-6, float(contour[3]))
        elongated_raw = max(1.0, float(contour[5]))
        complexity = _clip01(contour[7])
        pear_ratio = _clip01(contour[14])
        aspect_balance = 1.0 - _clip01(abs(math.log(aspect)) / math.log(3.0))
        concepts['round'] = _clip01(circularity * aspect_balance)
        concepts['elongated'] = _clip01((elongated_raw - 1.15) / 2.2)
        concepts['pear_like'] = pear_ratio
        concepts['rectangular_like'] = _clip01(extent * solidity * (1.0 - complexity * 0.35))

    texture = _safe_array(features.get('texture'))
    if texture.size >= 5:
        edge_density = _clip01(texture[0] * 2.2)
        lap_var = _clip01(texture[1])
        entropy = _clip01(texture[4])
        texture_rich = _clip01(0.34 * edge_density + 0.33 * lap_var + 0.33 * entropy)
        concepts['edge_rich'] = edge_density
        concepts['texture_rich'] = texture_rich
        concepts['smooth_surface'] = _clip01(1.0 - texture_rich)

    quality = _safe_array(features.get('quality'))
    if quality.size >= 2:
        area_ratio = _clip01(quality[0])
        blur_score = _clip01(quality[1])
        area_ok = _clip01(1.0 - abs(area_ratio - 0.42) / 0.42)
        concepts['clear_foreground'] = _clip01(0.45 * area_ok + 0.55 * blur_score)
        concepts['background_interference'] = _clip01(1.0 - area_ok)

    cluster_signal = _clip01(
        0.45 * concepts['texture_rich']
        + 0.35 * concepts['edge_rich']
        + 0.20 * (1.0 - concepts['smooth_surface'])
    )
    concepts['cluster_like'] = cluster_signal
    concepts['repeated_parts'] = _clip01(0.75 * cluster_signal + 0.25 * concepts['edge_rich'])
    concepts['single_object'] = _clip01((1.0 - cluster_signal) * (0.55 + 0.45 * concepts['clear_foreground']))
    return concepts


def concept_similarity(a, b):
    keys = [k for k in CONCEPT_NAMES if k in a and k in b]
    if not keys:
        return 0.0
    dist = 0.0
    for key in keys:
        dist += (float(a.get(key, 0.0)) - float(b.get(key, 0.0))) ** 2
    rmse = math.sqrt(dist / max(1, len(keys)))
    return _clip01(math.exp(-3.2 * rmse))


def summarize_concepts(concepts, top_n=5, min_score=0.35):
    ranked = sorted(
        [(name, float(score)) for name, score in (concepts or {}).items() if float(score) >= min_score],
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        {'id': name, 'name': DISPLAY_NAMES.get(name, name), 'score': round(score, 3)}
        for name, score in ranked[:top_n]
    ]
