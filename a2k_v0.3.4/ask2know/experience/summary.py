from pathlib import Path
import json
from datetime import datetime

FEATURE_WORDS = {
    'color': '颜色/色系',
    'contour': '形状/轮廓',
    'shape': '形状/结构',
    'texture': '纹理/局部重复',
    'cluster': '聚集结构',
    'size': '大小/尺度',
}


def _load_json(path, default):
    path = Path(path)
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return default
    return default


class ExperienceSummarizer:
    """Weak self-summary module for v0.3.4.

    It does not claim to understand concepts. It summarizes accumulated pairwise
    corrections into readable task experience so later versions can reuse it as a
    bridge toward a real visual concept layer.
    """

    def __init__(self, metadata_dir):
        self.metadata_dir = Path(metadata_dir)
        self.path = self.metadata_dir / 'experience_summary.json'
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _class_counts(self, pairwise_state):
        counts = {}
        for pair in pairwise_state.get('pairs', {}).values():
            classes = pair.get('classes', [])
            useful = pair.get('useful_features', {})
            weak = pair.get('weak_features', {})
            for cls in classes:
                info = counts.setdefault(cls, {'useful_features': {}, 'weak_features': {}, 'confused_with': {}})
                for f, n in useful.items():
                    info['useful_features'][f] = info['useful_features'].get(f, 0) + int(n)
                for f, n in weak.items():
                    info['weak_features'][f] = info['weak_features'].get(f, 0) + int(n)
                others = [x for x in classes if x != cls]
                for other in others:
                    info['confused_with'][other] = info['confused_with'].get(other, 0) + int(pair.get('confused_count', 0))
        return counts

    def update_from_project(self, objects, pairwise_state, model=None, logs=None):
        objects = objects or []
        logs = logs or []
        classes = [o.get('name') for o in objects if o.get('name')]
        class_counts = self._class_counts(pairwise_state or {})
        class_summaries = {}
        for cls in classes:
            info = class_counts.get(cls, {'useful_features': {}, 'weak_features': {}, 'confused_with': {}})
            useful_sorted = sorted(info['useful_features'].items(), key=lambda x: x[1], reverse=True)
            weak_sorted = sorted(info['weak_features'].items(), key=lambda x: x[1], reverse=True)
            confused_sorted = sorted(info['confused_with'].items(), key=lambda x: x[1], reverse=True)
            class_summaries[cls] = {
                'useful_features': [{'feature': f, 'name': FEATURE_WORDS.get(f, f), 'count': n} for f, n in useful_sorted],
                'weak_features': [{'feature': f, 'name': FEATURE_WORDS.get(f, f), 'count': n} for f, n in weak_sorted],
                'often_confused_with': [{'class': c, 'count': n} for c, n in confused_sorted],
                'summary_text': self._make_text(cls, useful_sorted, weak_sorted, confused_sorted),
            }

        data = {
            'schema_version': '0.3.4',
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'purpose': 'weak self-summary; used to reduce repetitive questions and prepare future concept layer',
            'classes': class_summaries,
            'global_notes': self._global_notes(pairwise_state or {}, logs),
            'future_reserved': {
                'visual_concept_layer': 'reserved for v0.4.x: concepts such as round, elongated, clustered, symmetric',
                'crawler_external_candidates': 'reserved for later: web images must enter external_candidate, never confirmed directly',
            }
        }
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return data

    def _make_text(self, cls, useful, weak, confused):
        parts = []
        if useful:
            parts.append('当前更值得参考：' + '、'.join(FEATURE_WORDS.get(f, f) for f, _ in useful[:3]))
        if weak:
            parts.append('当前可能不稳定：' + '、'.join(FEATURE_WORDS.get(f, f) for f, _ in weak[:3]))
        if confused:
            parts.append('常与这些类别混淆：' + '、'.join(c for c, _ in confused[:3]))
        return f'{cls}: ' + ('；'.join(parts) if parts else '经验还少，暂时没有稳定总结。')

    def _global_notes(self, pairwise_state, logs):
        pair_count = len(pairwise_state.get('pairs', {}))
        correction_count = sum(int(p.get('correction_count', 0)) for p in pairwise_state.get('pairs', {}).values())
        return [
            f'当前累计类别对经验 {pair_count} 组。',
            f'当前累计纠错原因 {correction_count} 次。',
            '总结仅作为弱经验，不等同于真正视觉概念；后续可迁移到概念层。'
        ]
