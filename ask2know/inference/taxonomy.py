class TaxonomySpec:
    """Configuration-backed soft taxonomy for multilayer recognition."""

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get('enable', False))
        self.root = str(self.config.get('root', '')).strip()
        self.top_k_paths = max(1, int(self.config.get('top_k_paths', 5)))
        self.score_weight = max(0.0, min(0.5, float(self.config.get('score_weight', 0.08))))
        self.min_gap = max(0.0, float(self.config.get('min_gap', 0.0)))
        self.max_score_margin = max(0.0, float(self.config.get('max_score_margin', 0.05)))
        self.apply_to_score = bool(self.config.get('apply_to_score', True))
        self.label_paths = self._normalize_label_paths(
            self.config.get('label_paths')
            or self.config.get('paths')
            or {}
        )
        self.level_names = [str(item) for item in (self.config.get('levels') or [])]
        self.level_weights = {
            str(key): max(0.0, float(value))
            for key, value in dict(self.config.get('level_weights') or {}).items()
        }

    @staticmethod
    def _normalize_label_paths(raw_paths):
        paths = {}
        for label, path in dict(raw_paths or {}).items():
            if isinstance(path, str):
                nodes = [part.strip() for part in path.split('/') if part.strip()]
            else:
                nodes = [str(part).strip() for part in (path or []) if str(part).strip()]
            if nodes:
                paths[str(label)] = nodes
        return paths

    def path_for_label(self, label):
        label = str(label)
        path = list(self.label_paths.get(label) or [])
        if not path:
            return []
        if self.root and path[0] != self.root:
            path.insert(0, self.root)
        if path[-1] != label:
            path.append(label)
        return path

    def level_name(self, index):
        if 0 <= int(index) < len(self.level_names):
            return self.level_names[int(index)]
        return f'level_{int(index) + 1}'

    def weight_for_level(self, index, node):
        level = self.level_name(index)
        if level in self.level_weights:
            return self.level_weights[level]
        if str(node) in self.level_weights:
            return self.level_weights[str(node)]
        return 1.0

    def export(self):
        return {
            'enable': self.enabled,
            'root': self.root,
            'levels': list(self.level_names),
            'label_paths': {label: list(path) for label, path in self.label_paths.items()},
            'level_weights': dict(self.level_weights),
            'top_k_paths': self.top_k_paths,
            'score_weight': self.score_weight,
            'min_gap': self.min_gap,
            'max_score_margin': self.max_score_margin,
            'apply_to_score': self.apply_to_score,
        }


def taxonomy_level_summary(rows, taxonomy_config):
    spec = TaxonomySpec(taxonomy_config)
    if not spec.enabled:
        return {}
    totals = {}
    for row in rows or []:
        true_path = spec.path_for_label(row.get('true_label'))
        predicted_path = row.get('predicted_path') or []
        max_len = max(len(true_path), len(predicted_path))
        for idx in range(max_len):
            level = spec.level_name(idx)
            item = totals.setdefault(level, {'total': 0, 'correct': 0})
            true_node = true_path[idx] if idx < len(true_path) else None
            pred_node = predicted_path[idx] if idx < len(predicted_path) else None
            if true_node is None:
                continue
            item['total'] += 1
            if true_node == pred_node:
                item['correct'] += 1
    return {
        level: {
            'total': item['total'],
            'correct': item['correct'],
            'accuracy': item['correct'] / max(1, item['total']),
        }
        for level, item in totals.items()
    }
