from pathlib import Path
from ask2know.utils.io_utils import load_json

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


class DatasetLoader:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.objects_path = self.dataset_dir / 'objects.json'
        self.concepts_path = self.dataset_dir / 'concepts.json'
        self.train_dir = self.dataset_dir / 'train'
        self.unknown_dir = self.dataset_dir / 'unknown'
        self.unlabeled_dir = self.dataset_dir / 'unlabeled'

    def load_objects(self):
        if self.objects_path.exists():
            data = load_json(self.objects_path)
            return data.get('objects', [])
        # Fallback: infer classes from train folders to reduce user blocking.
        objects = []
        if self.train_dir.exists():
            for idx, class_dir in enumerate(sorted([p for p in self.train_dir.iterdir() if p.is_dir()]), 1):
                objects.append({
                    'object_id': f'C{idx:03d}',
                    'name': class_dir.name,
                    'display_name': class_dir.name,
                    'description': 'auto inferred from train folder'
                })
        return objects

    def load_concepts(self):
        if not self.concepts_path.exists():
            return []
        data = load_json(self.concepts_path)
        return data.get('concepts', [])

    def load_train_samples(self):
        samples = []
        if not self.train_dir.exists():
            return samples
        allowed_labels = None
        if self.objects_path.exists():
            allowed_labels = {item.get('name') for item in self.load_objects() if item.get('name')}
        for class_dir in sorted(self.train_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            if allowed_labels is not None and label not in allowed_labels:
                continue
            for img in sorted(class_dir.rglob('*')):
                if img.suffix.lower() in IMAGE_EXTS:
                    samples.append({'path': str(img), 'label': label})
        return samples

    def load_unlabeled_samples(self):
        return self.load_unknown_samples()

    def load_unknown_samples(self):
        samples = []
        if not self.unknown_dir.exists():
            return samples
        for img in sorted(self.unknown_dir.rglob('*')):
            if img.suffix.lower() in IMAGE_EXTS:
                samples.append({'path': str(img), 'label': None})
        return samples

    def load_legacy_unlabeled_flat_samples(self):
        samples = []
        if not self.unlabeled_dir.exists():
            return samples
        for img in sorted(self.unlabeled_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                samples.append({'path': str(img), 'label': None})
        return samples

    def load_eval_samples(self):
        samples = []
        if not self.unlabeled_dir.exists():
            return samples
        allowed_labels = None
        if self.objects_path.exists():
            allowed_labels = {item.get('name') for item in self.load_objects() if item.get('name')}
        for class_dir in sorted(self.unlabeled_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            if allowed_labels is not None and label not in allowed_labels:
                continue
            for img in sorted(class_dir.rglob('*')):
                if img.suffix.lower() in IMAGE_EXTS:
                    samples.append({'path': str(img), 'label': label})
        return samples
