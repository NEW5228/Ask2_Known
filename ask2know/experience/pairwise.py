from pathlib import Path
import json
from datetime import datetime

FEATURE_TO_QUESTION = {
    'color': 'Q_COLOR_IMPORTANCE',
    'shape': 'Q_CONTOUR_IMPORTANCE',
    'contour': 'Q_CONTOUR_IMPORTANCE',
    'texture': 'Q_TEXTURE_IMPORTANCE',
    'cluster': 'Q_TEXTURE_IMPORTANCE',
    'quality': 'Q_SAMPLE_QUALITY',
    'background': 'Q_SAMPLE_QUALITY',
    'size': 'Q_SIZE_RELIABILITY',
}

REASON_TO_FEATURES = {
    'color': ['color'],
    'shape': ['contour'],
    'texture': ['texture'],
    'cluster': ['texture', 'contour'],
    'background': [],
    'quality': [],
    'size': ['size'],
    'other': [],
}

REASON_TO_CONCEPTS = {
    'color': ['color_family'],
    'shape': ['round', 'elongated', 'pear_like', 'rectangular_like'],
    'texture': ['smooth_surface', 'texture_rich', 'edge_rich'],
    'cluster': ['single_object', 'cluster_like', 'repeated_parts'],
    'background': ['background_interference', 'clear_foreground'],
    'quality': ['clear_foreground', 'background_interference'],
    'size': [],
    'other': [],
}

# These are optional task hints. They are examples, not hard-coded final truth.
FRUIT_PAIR_HINTS = {
    frozenset(['apple', 'grape']): {
        'color': '葡萄通常更偏紫色/深色，苹果通常更偏红、绿或黄。',
        'shape': '葡萄常是多个小圆聚集，苹果通常更像单个较大的圆形水果。',
        'texture': '葡萄可能有成串/多颗粒聚集感，苹果表面通常更连续、更平滑。',
        'cluster': '葡萄常是多个小圆聚集，苹果多为单体。',
    },
    frozenset(['apple', 'pear']): {
        'color': '苹果和梨颜色可能接近，颜色不一定可靠。',
        'shape': '梨通常更接近上窄下宽，苹果更圆润。',
        'texture': '苹果和梨的表面纹理差异可能较弱，需要结合局部形状。',
        'cluster': '二者通常都是单体水果，不适合靠聚集结构区分。',
    },
    frozenset(['orange', 'apple']): {
        'color': '桔子/橙子更偏橙色，苹果可能偏红、绿或黄。',
        'shape': '二者都可能接近圆形，单纯轮廓可能不够。',
        'texture': '桔子/橙子表面通常更粗糙，苹果表面更平滑。',
        'cluster': '二者通常都是单体水果，不适合靠聚集结构区分。',
    },
    frozenset(['banana', 'orange']): {
        'color': '香蕉和桔子都可能偏黄/橙，颜色有时会混。',
        'shape': '香蕉通常是细长/弯曲形状，桔子更圆。',
        'texture': '纹理不是主要判断点，轮廓形态通常更关键。',
        'cluster': '二者通常不是多个小圆聚集结构。',
    },
    frozenset(['banana', 'apple']): {
        'color': '香蕉和苹果颜色可能有重叠，但不是主要依据。',
        'shape': '香蕉通常细长弯曲，苹果通常圆润。',
        'texture': '纹理通常不是最关键，形状更重要。',
        'cluster': '香蕉和苹果通常都是单体，聚集结构不是主要点。',
    },
    frozenset(['banana', 'grape']): {
        'color': '香蕉通常黄，葡萄通常紫/深色或绿色，颜色可参考。',
        'shape': '香蕉是单个长条形，葡萄是多个小圆聚集。',
        'texture': '葡萄的聚集颗粒结构比香蕉明显。',
        'cluster': '葡萄常有多个小圆聚集，香蕉通常没有。',
    },
    frozenset(['pear', 'grape']): {
        'color': '梨和葡萄颜色可能不同，尤其紫葡萄更明显。',
        'shape': '梨通常是单个上窄下宽水果，葡萄常是多个小圆聚集。',
        'texture': '葡萄的成串/颗粒结构更重要。',
        'cluster': '葡萄常有多个小圆聚集，梨通常为单体。',
    },
    frozenset(['pear', 'banana']): {
        'color': '梨和香蕉都可能偏黄，颜色可能混。',
        'shape': '香蕉更细长弯曲，梨更接近上窄下宽。',
        'texture': '纹理不是主要判断点。',
        'cluster': '二者通常不是靠聚集结构区分。',
    },
    frozenset(['pear', 'orange']): {
        'color': '桔子/橙子更偏橙，梨可能偏黄绿。',
        'shape': '梨常上窄下宽，桔子/橙子更圆。',
        'texture': '桔子/橙子表面可能更粗糙。',
        'cluster': '二者通常都是单体水果。',
    },
}


def pair_key(a, b):
    a = str(a)
    b = str(b)
    return '__vs__'.join(sorted([a, b]))


class PairwiseExperienceManager:
    def __init__(self, metadata_dir=None, path=None, version='0.3.6'):
        self.version = version
        if path is not None:
            self.path = Path(path)
        else:
            self.path = Path(metadata_dir or '.') / 'pairwise_experience.json'
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and 'pairs' in data:
                    data.setdefault('schema_version', self.version)
                    return data
            except Exception:
                pass
        return {'schema_version': self.version, 'pairs': {}}

    def save(self):
        self.data['schema_version'] = self.version
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get_pair(self, a, b):
        key = pair_key(a, b)
        pair = self.data['pairs'].setdefault(key, {
            'classes': sorted([str(a), str(b)]),
            'confused_count': 0,
            'correction_count': 0,
            'reason_counts': {},
            'useful_features': {},
            'weak_features': {},
            'useful_concepts': {},
            'weak_concepts': {},
            'question_counts': {},
            'question_helpful': {},
            'notes': [],
            'last_updated': None,
        })
        return pair

    def get_feature_bias(self, a, b):
        pair = self.get_pair(a, b)
        return dict(pair.get('useful_features', {})), dict(pair.get('weak_features', {}))

    def get_question_bias(self, a, b):
        useful, weak = self.get_feature_bias(a, b)
        q_bonus = {}
        q_penalty = {}
        for feature, count in useful.items():
            qid = FEATURE_TO_QUESTION.get(feature)
            if qid:
                q_bonus[qid] = q_bonus.get(qid, 0.0) + min(float(count) * 0.18, 0.55)
        for feature, count in weak.items():
            qid = FEATURE_TO_QUESTION.get(feature)
            if qid:
                q_penalty[qid] = q_penalty.get(qid, 0.0) + min(float(count) * 0.12, 0.36)
        return q_bonus, q_penalty

    def record_question_result(self, a, b, question_id, helpful):
        pair = self.get_pair(a, b)
        pair['question_counts'][question_id] = int(pair['question_counts'].get(question_id, 0)) + 1
        if helpful:
            pair['question_helpful'][question_id] = int(pair['question_helpful'].get(question_id, 0)) + 1
        pair['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save()

    def record_corrections(self, predicted, true_label, reason_items, free_note=None):
        pair = self.get_pair(predicted, true_label)
        pair['confused_count'] = int(pair.get('confused_count', 0)) + 1
        pair['correction_count'] = int(pair.get('correction_count', 0)) + 1
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        saved_reasons = []
        pair.setdefault('useful_concepts', {})
        pair.setdefault('weak_concepts', {})
        for reason_id, reason_text in reason_items:
            pair['reason_counts'][reason_id] = int(pair['reason_counts'].get(reason_id, 0)) + 1
            features = REASON_TO_FEATURES.get(reason_id, [])
            concepts = REASON_TO_CONCEPTS.get(reason_id, [])
            if reason_id in ('background', 'quality', 'other'):
                for concept in concepts:
                    pair['weak_concepts'][concept] = int(pair['weak_concepts'].get(concept, 0)) + 1
            elif reason_id == 'size':
                pair['weak_features']['size'] = int(pair['weak_features'].get('size', 0)) + 1
            else:
                for feature in features:
                    pair['useful_features'][feature] = int(pair['useful_features'].get(feature, 0)) + 1
                for concept in concepts:
                    pair['useful_concepts'][concept] = int(pair['useful_concepts'].get(concept, 0)) + 1
            saved_reasons.append({'reason_id': reason_id, 'reason_text': reason_text})
        note = {
            'time': now,
            'predicted': predicted,
            'true_label': true_label,
            'reasons': saved_reasons,
            'note': free_note or ''
        }
        pair['notes'].append(note)
        if len(pair['notes']) > 50:
            pair['notes'] = pair['notes'][-50:]
        pair['last_updated'] = now
        self.save()
        return pair

    def record_correction(self, predicted, true_label, reason_id, reason_text=None, free_note=None):
        return self.record_corrections(predicted, true_label, [(reason_id, reason_text or reason_id)], free_note=free_note)

    def suggest_pair_prompt(self, a, b):
        pair = self.get_pair(a, b)
        useful = sorted(pair.get('useful_features', {}).items(), key=lambda x: x[1], reverse=True)
        weak = sorted(pair.get('weak_features', {}).items(), key=lambda x: x[1], reverse=True)
        parts = []
        if useful:
            parts.append('历史经验里，这两个类别更适合参考：' + '、'.join([x[0] for x in useful[:3]]) + '。')
        if weak:
            parts.append('历史经验里，这些特征可能不可靠：' + '、'.join([x[0] for x in weak[:3]]) + '。')
        return ''.join(parts)

    def pair_specific_question_hint(self, a, b):
        pair = self.get_pair(a, b)
        useful = sorted(pair.get('useful_features', {}).items(), key=lambda x: x[1], reverse=True)
        if useful:
            top = useful[0][0]
            hints = FRUIT_PAIR_HINTS.get(frozenset([a, b]), {})
            if top in hints:
                return hints[top]
            if top == 'color':
                return '请重点观察当前图片的颜色/色系是否和两个候选类别存在明显差异。'
            if top == 'contour':
                return '请重点观察当前图片的整体形状、长宽比例、圆润程度或弯曲程度。'
            if top == 'texture':
                return '请重点观察当前图片的表面纹理、颗粒感、局部重复结构。'
        return ''

    def correction_options(self, predicted, true_label):
        hints = FRUIT_PAIR_HINTS.get(frozenset([predicted, true_label]), {})
        color_hint = hints.get('color', '颜色或色系差异明显。')
        shape_hint = hints.get('shape', '整体形状、结构或轮廓不同。')
        texture_hint = hints.get('texture', '表面纹理、颗粒感或局部重复结构不同。')
        cluster_hint = hints.get('cluster', '是否为单体物体、多个小物体聚集、重复结构不同。')
        return [
            ('A', 'color', color_hint, {'increase': ['color'], 'decrease': []}),
            ('B', 'shape', shape_hint, {'increase': ['contour'], 'decrease': []}),
            ('C', 'texture', texture_hint, {'increase': ['texture'], 'decrease': []}),
            ('D', 'cluster', cluster_hint, {'increase': ['texture', 'contour'], 'decrease': []}),
            ('E', 'background', '背景、光线、遮挡或主体不清楚影响了判断。', {'increase': [], 'decrease': ['color', 'texture']}),
            ('F', 'other', '不确定 / 其他原因。', {'increase': [], 'decrease': []}),
        ]

    def export(self):
        return self.data
