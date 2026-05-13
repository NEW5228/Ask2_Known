from ask2know.questions.question_bank import QUESTION_BANK


class QuestionSelector:
    def __init__(self, question_weights=None):
        self.question_weights = question_weights or {q['id']: 1.0 for q in QUESTION_BANK}
        self.last_question_id = None
        self.ask_counts = {q['id']: 0 for q in QUESTION_BANK}

    def select(self, top_a, top_b):
        detail_a = top_a.get('detail', {})
        detail_b = top_b.get('detail', {})
        candidates = []

        for q in QUESTION_BANK:
            feature = q['feature']
            qid = q['id']
            a_score = float(detail_a.get(feature, 0.0))
            b_score = float(detail_b.get(feature, 0.0))
            gap = abs(a_score - b_score)

            # 分数很接近：说明这个特征当前分不开，适合问“是否可靠/是否重要”
            confusion_score = 1.0 - min(gap, 1.0)

            # 分数有差异：说明这个特征也可能有区分潜力
            difference_score = min(gap, 1.0)

            history_weight = float(self.question_weights.get(qid, 1.0))
            repeat_penalty = 0.60 if qid == self.last_question_id else 0.0
            count_penalty = min(self.ask_counts.get(qid, 0) * 0.08, 0.40)

            # 对样本质量问题稍微保留一点优先级：当所有候选都很像时，它可能更重要
            quality_bonus = 0.10 if q.get('kind') == 'sample_quality' else 0.0

            score = (
                0.40 * confusion_score
                + 0.20 * difference_score
                + 0.30 * history_weight
                + quality_bonus
                - repeat_penalty
                - count_penalty
            )
            candidates.append((score, q))

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
