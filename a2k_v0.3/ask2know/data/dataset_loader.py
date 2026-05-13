from pathlib import Path
from ask2know.utils.io_utils import load_json

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

class DatasetLoader:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.objects_path = self.dataset_dir / 'objects.json'
        self.concepts_path = self.dataset_dir / 'concepts.json'
        self.train_dir = self.dataset_dir / 'train'
        self.unlabeled_dir = self.dataset_dir / 'unlabeled'

    def load_objects(self):
        data = load_json(self.objects_path)
        return data.get('objects', [])

    def load_concepts(self):
        if not self.concepts_path.exists():
            return []
        data = load_json(self.concepts_path)
        return data.get('concepts', [])

    def load_train_samples(self):
        samples = []
        if not self.train_dir.exists():
            return samples
        for class_dir in sorted(self.train_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for img in sorted(class_dir.rglob('*')):
                if img.suffix.lower() in IMAGE_EXTS:
                    samples.append({'path': str(img), 'label': label})
        return samples

    def load_unlabeled_samples(self):
        samples = []
        if not self.unlabeled_dir.exists():
            return samples
        for img in sorted(self.unlabeled_dir.rglob('*')):
            if img.suffix.lower() in IMAGE_EXTS:
                samples.append({'path': str(img), 'label': None})
        return samples
