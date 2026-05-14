class AdaptiveWeights:
    def __init__(self, weights, step=0.07, min_weight=0.05, max_weight=0.70):
        self.weights = dict(weights)
        self.step = float(step)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self._normalize()

    def apply_concepts(self, concepts):
        for c in concepts:
            for f, factor in self._adjustments(c.get('important_features', [])):
                if f in self.weights:
                    self.weights[f] += self.step * factor
            for f, factor in self._adjustments(c.get('weak_features', [])):
                if f in self.weights:
                    self.weights[f] -= self.step * factor
        self._clip()
        self._normalize()

    def update(self, increase=None, decrease=None):
        increase = increase or []
        decrease = decrease or []
        before = dict(self.weights)
        for f, factor in self._adjustments(increase):
            if f in self.weights:
                self.weights[f] += self.step * factor
        for f, factor in self._adjustments(decrease):
            if f in self.weights:
                self.weights[f] -= self.step * factor
        self._clip()
        self._normalize()
        return before, dict(self.weights)

    def _adjustments(self, items):
        if isinstance(items, dict):
            return [(k, float(v)) for k, v in items.items()]
        return [(k, 1.0) for k in (items or [])]

    def _clip(self):
        for k in self.weights:
            self.weights[k] = max(self.min_weight, min(self.max_weight, self.weights[k]))

    def _normalize(self):
        s = sum(self.weights.values())
        if s <= 0:
            n = len(self.weights)
            self.weights = {k: 1.0 / n for k in self.weights}
        else:
            self.weights = {k: v / s for k, v in self.weights.items()}

    def export(self):
        return dict(self.weights)
