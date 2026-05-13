from pathlib import Path
import json
import re
import shutil
from datetime import datetime

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _safe_name(name):
    text = str(name).strip().replace(' ', '_')
    text = re.sub(r'[^0-9A-Za-z_\-]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text or 'unknown'


class SamplePoolManager:
    def __init__(self, project_root=None, output_dir=None, dataset_dir=None):
        self.project_root = Path(project_root) if project_root else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None

        if self.project_root:
            self.base_dir = self.project_root / 'sample_pools'
            self.metadata_dir = self.project_root / 'metadata'
        else:
            self.base_dir = self.output_dir / 'sample_pools'
            self.metadata_dir = self.output_dir / 'metadata'

        if self.dataset_dir is None:
            if self.project_root:
                self.dataset_dir = self.project_root / 'datasets'
            else:
                self.dataset_dir = self.output_dir / 'datasets'

        self.train_dir = self.dataset_dir / 'train'
        self.unlabeled_dir = self.dataset_dir / 'unlabeled'
        self.confirmed_dir = self.train_dir
        self.candidate_dir = self.base_dir / 'candidate'
        self.rejected_dir = self.base_dir / 'rejected'
        self.unknown_dir = self.base_dir / 'unknown'
        self.processed_dir = self.dataset_dir / 'processed'
        self.index_path = self.metadata_dir / 'dataset_index.json'
        self.history_path = self.metadata_dir / 'sample_history.jsonl'
        self.import_map_path = self.metadata_dir / 'unlabeled_import_map.jsonl'

        for d in [self.train_dir, self.unlabeled_dir, self.candidate_dir, self.rejected_dir, self.unknown_dir, self.processed_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_index(self):
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'classes' in data:
                    return data
            except Exception:
                pass
        return {'schema_version': '0.3.4', 'classes': {}}

    def _save_index(self, data):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _append_history(self, item):
        item = dict(item)
        item.setdefault('time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        item.setdefault('a2k_version', '0.3.4')
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def _scan_class_state(self, label):
        safe = _safe_name(label)
        class_dir = self.train_dir / safe
        max_id = 0
        count = 0
        if class_dir.exists():
            pattern = re.compile(rf'^{re.escape(safe)}_(\d+)\.[^.]+$', re.IGNORECASE)
            for p in class_dir.iterdir():
                if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                    continue
                count += 1
                m = pattern.match(p.name)
                if m:
                    max_id = max(max_id, int(m.group(1)))
        return {'next_id': max_id + 1 if max_id else count + 1, 'count': count}

    def ensure_for_classes(self, labels):
        index = self._load_index()
        for label in labels:
            safe = _safe_name(label)
            (self.train_dir / safe).mkdir(parents=True, exist_ok=True)
            scanned = self._scan_class_state(safe)
            if safe not in index['classes']:
                index['classes'][safe] = scanned
            else:
                current = index['classes'][safe]
                # If the user manually adds seed images after init_task, keep numbering safe.
                current['next_id'] = max(int(current.get('next_id', 1)), int(scanned.get('next_id', 1)))
                current['count'] = max(int(current.get('count', 0)), int(scanned.get('count', 0)))
                index['classes'][safe] = current
        self._save_index(index)

    def rebuild_index_from_train(self):
        index = {'schema_version': '0.3.4', 'classes': {}}
        if self.train_dir.exists():
            for class_dir in sorted([p for p in self.train_dir.iterdir() if p.is_dir()]):
                index['classes'][class_dir.name] = self._scan_class_state(class_dir.name)
        self._save_index(index)
        return index

    def _next_class_file(self, label, ext):
        safe = _safe_name(label)
        index = self._load_index()
        if safe not in index['classes']:
            index['classes'][safe] = self._scan_class_state(safe)
        next_id = int(index['classes'][safe].get('next_id', 1))
        dst_dir = self.train_dir / safe
        dst_dir.mkdir(parents=True, exist_ok=True)
        width = 3 if next_id < 1000 else len(str(next_id))
        dst = dst_dir / f'{safe}_{next_id:0{width}d}{ext.lower()}'
        while dst.exists():
            next_id += 1
            width = 3 if next_id < 1000 else len(str(next_id))
            dst = dst_dir / f'{safe}_{next_id:0{width}d}{ext.lower()}'
        index['classes'][safe]['next_id'] = next_id + 1
        index['classes'][safe]['count'] = int(index['classes'][safe].get('count', 0)) + 1
        self._save_index(index)
        return dst

    def _unique_pool_file(self, base_dir, prefix, ext):
        prefix = _safe_name(prefix)
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        dst = base_dir / f'{prefix}_{ts}{ext.lower()}'
        i = 1
        while dst.exists():
            dst = base_dir / f'{prefix}_{ts}_{i}{ext.lower()}'
            i += 1
        return dst

    def _move_file(self, src, dst):
        src = Path(src)
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return str(dst)

    def add_confirmed(self, image_path, label):
        src = Path(image_path)
        dst = self._next_class_file(label, src.suffix)
        saved = self._move_file(src, dst)
        self._append_history({
            'action': 'confirmed',
            'class': _safe_name(label),
            'old_name': src.name,
            'saved_as': saved,
        })
        return saved

    def add_candidate(self, image_path, label):
        src = Path(image_path)
        safe = _safe_name(label)
        dst = self._unique_pool_file(self.candidate_dir / safe, f'{safe}_candidate', src.suffix)
        saved = self._move_file(src, dst)
        self._append_history({
            'action': 'candidate',
            'class': safe,
            'old_name': src.name,
            'saved_as': saved,
        })
        return saved

    def add_rejected(self, image_path, reason='rejected'):
        src = Path(image_path)
        prefix = _safe_name(reason or 'rejected')
        dst = self._unique_pool_file(self.rejected_dir, prefix, src.suffix)
        saved = self._move_file(src, dst)
        self._append_history({
            'action': 'rejected',
            'reason': prefix,
            'old_name': src.name,
            'saved_as': saved,
        })
        return saved

    def add_unknown(self, image_path):
        src = Path(image_path)
        dst = self._unique_pool_file(self.unknown_dir, 'unknown', src.suffix)
        saved = self._move_file(src, dst)
        self._append_history({
            'action': 'unknown',
            'old_name': src.name,
            'saved_as': saved,
        })
        return saved

    def normalize_unlabeled(self):
        """Rename active unlabeled images to img_001, img_002... for stable processing."""
        if not self.unlabeled_dir.exists():
            return []
        files = sorted([p for p in self.unlabeled_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        if not files:
            return []

        temp_files = []
        for i, src in enumerate(files, 1):
            temp = self.unlabeled_dir / f'__a2k_tmp_{i:06d}{src.suffix.lower()}'
            while temp.exists():
                temp = self.unlabeled_dir / f'__a2k_tmp_{i:06d}_{datetime.now().strftime("%f")}{src.suffix.lower()}'
            shutil.move(str(src), str(temp))
            temp_files.append((src, temp))

        mappings = []
        for i, (old_src, temp) in enumerate(temp_files, 1):
            ext = temp.suffix.lower()
            dst = self.unlabeled_dir / f'img_{i:03d}{ext}'
            while dst.exists():
                i += 1
                dst = self.unlabeled_dir / f'img_{i:03d}{ext}'
            shutil.move(str(temp), str(dst))
            record = {
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'old_name': old_src.name,
                'new_name': dst.name,
                'path': str(dst),
                'a2k_version': '0.3.4'
            }
            mappings.append(record)
            with open(self.import_map_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return mappings
