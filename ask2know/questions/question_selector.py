from ask2know.questions.question_bank import QUESTION_BANK


def _clip01(value):
    return max(0.0, min(1.0, float(value)))


class QuestionSelector:
    def __init__(self, question_weights=None, pairwise_manager=None):
        self.question_weights = question_weights or {q['id']: 1.0 for q in QUESTION_BANK}
        self.last_question_id = None
        self.ask_counts = {q['id']: 0 for q in QUESTION_BANK}
        self.pairwise_manager = pairwise_manager

    def select(self, top_a, top_b, weights=None):
        detail_a = top_a.get('detail', {})
        detail_b = top_b.get('detail', {})
        label_a = top_a.get('label')
        label_b = top_b.get('label')
        weights = weights or {}
        candidates = []

        q_bonus = {}
        q_penalty = {}
        if self.pairwise_manager is not None and label_a and label_b:
            q_bonus, q_penalty = self.pairwise_manager.get_question_bias(label_a, label_b)

        for q in QUESTION_BANK:
            feature = q['feature']
            qid = q['id']
            feature_available = feature in detail_a or feature in detail_b
            a_score = float(detail_a.get(feature, 0.0))
            b_score = float(detail_b.get(feature, 0.0))
            gap = abs(a_score - b_score)
            top_feature_score = max(a_score, b_score)
            bottom_feature_score = min(a_score, b_score)
            feature_weight = float(weights.get(feature, 0.08))

            # 高价值问题应抓住当前混淆的关键：
            # 1. 某个特征明显支持一个候选，适合问“这个差异是否真实可靠”。
            # 2. 历史经验认为某特征重要，但当前两者很接近，适合问“是否被背景/角度/质量干扰”。
            # 3. 已有权重高的特征如果表现不稳定，优先验证，避免继续放大错误方向。
            decisive_signal = _clip01(gap / 0.22)
            close_but_relevant = _clip01((1.0 - gap / 0.10) * (feature_weight / 0.30))
            low_signal = _clip01((0.72 - top_feature_score) / 0.72)
            saturation = 1.0 if gap < 0.04 and bottom_feature_score > 0.92 else 0.0

            history_weight = float(self.question_weights.get(qid, 1.0))
            repeat_penalty = 0.60 if qid == self.last_question_id else 0.0
            count_penalty = min(self.ask_counts.get(qid, 0) * 0.08, 0.40)
            pair_bonus = float(q_bonus.get(qid, 0.0))
            pair_penalty = float(q_penalty.get(qid, 0.0))

            if q.get('kind') == 'sample_quality':
                quality_pressure = max(low_signal, saturation)
                base_score = 0.58 * quality_pressure + 0.18 * close_but_relevant
            elif q.get('kind') == 'feature_reliability':
                base_score = 0.48 * decisive_signal + 0.30 * close_but_relevant + 0.12 * low_signal
            else:
                base_score = 0.56 * decisive_signal + 0.24 * close_but_relevant + 0.08 * low_signal
            if not feature_available:
                base_score = 0.02

            # 历史权重只做轻微校正，不能盖过当前图片的证据。
            score = (
                base_score
                + 0.10 * history_weight
                + pair_bonus
                - pair_penalty
                - repeat_penalty
                - count_penalty
            )
            selected = dict(q)
            selected['_selection'] = {
                'feature': feature,
                'a_score': a_score,
                'b_score': b_score,
                'gap': gap,
                'feature_weight': feature_weight,
                'decisive_signal': decisive_signal,
                'close_but_relevant': close_but_relevant,
                'low_signal': low_signal,
                'saturation': saturation,
                'score': score,
            }
            candidates.append((score, selected))

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[0][1]
        self.last_question_id = selected['id']
        self.ask_counts[selected['id']] = self.ask_counts.get(selected['id'], 0) + 1
        return selected

    def export(self):
        return {
            'question_weights': dict(self.question_weights),
            'ask_counts': dict(self.ask_counts),
            'last_question_id': self.last_question_id,
        }
