from ask2know.questions.question_bank import QUESTION_BANK


class QuestionSelector:
    def __init__(self, question_weights=None):
        self.question_weights = question_weights or {q["id"]: 1.0 for q in QUESTION_BANK}
        self.last_question_id = None
        self.ask_counts = {q["id"]: 0 for q in QUESTION_BANK}

    def select(self, top_a, top_b):
        detail_a = top_a.get("detail", {})
        detail_b = top_b.get("detail", {})

        candidates = []

        for q in QUESTION_BANK:
            feature = q["feature"]
            question_id = q["id"]

            score_a = detail_a.get(feature, 0.0)
            score_b = detail_b.get(feature, 0.0)

            feature_gap = abs(score_a - score_b)

            # 差距越小，说明这个特征越分不清；差距越大，说明这个特征可能更有区分价值
            uncertainty_score = 1.0 - min(feature_gap, 1.0)
            difference_score = min(feature_gap, 1.0)

            # 历史问题权重：以前有用的问题优先
            history_weight = self.question_weights.get(question_id, 1.0)

            # 防止连续问同一个问题
            repeat_penalty = 0.45 if question_id == self.last_question_id else 0.0

            # 问得太多的问题稍微降权，避免一直问同类问题
            count_penalty = min(self.ask_counts.get(question_id, 0) * 0.08, 0.35)

            # 综合得分：
            # 既考虑“不确定”，也考虑“差异潜力”，再结合历史收益
            final_score = (
                0.35 * uncertainty_score
                + 0.35 * difference_score
                + 0.30 * history_weight
                - repeat_penalty
                - count_penalty
            )

            candidates.append((final_score, q))

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[0][1]

        self.last_question_id = selected["id"]
        self.ask_counts[selected["id"]] = self.ask_counts.get(selected["id"], 0) + 1

        return selected

    def export(self):
        return {
            "question_weights": dict(self.question_weights),
            "ask_counts": dict(self.ask_counts),
            "last_question_id": self.last_question_id,
        }