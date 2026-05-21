from pathlib import Path
import json
import re
import shutil
from datetime import datetime

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _safe_name(name):
    text = str(name).strip().replace(' ', '_')
    text = re.sub(r'[^\w\-]+', '_', text, flags=re.UNICODE)
    text = re.sub(r'_+', '_', text).strip('_')
    return text or 'unknown'


class SamplePoolManager:
    def __init__(self, project_root=None, output_dir=None, dataset_dir=None, version='0.4.4.1'):
        self.version = version
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
        self.learning_unknown_dir = self.dataset_dir / 'unknown'
        self.unlabeled_dir = self.dataset_dir / 'unlabeled'
        self.confirmed_dir = self.train_dir
        self.candidate_dir = self.base_dir / 'candidate'
        self.rejected_dir = self.base_dir / 'rejected'
        self.unknown_dir = self.base_dir / 'unknown'
        self.processed_dir = self.dataset_dir / 'processed'
        self.index_path = self.metadata_dir / 'dataset_index.json'
        self.history_path = self.metadata_dir / 'sample_history.jsonl'
        self.import_map_path = self.metadata_dir / 'unlabeled_import_map.jsonl'
        self.unknown_import_map_path = self.metadata_dir / 'unknown_import_map.jsonl'
        self.train_normalize_map_path = self.metadata_dir / 'train_normalize_map.jsonl'
        self.project_meta_path = self.metadata_dir / 'project_meta.json'

        for d in [self.train_dir, self.learning_unknown_dir, self.unlabeled_dir, self.candidate_dir, self.rejected_dir, self.unknown_dir, self.processed_dir, self.metadata_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_json(self, path, default):
        path = Path(path)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def _save_json(self, path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_index(self):
        data = self._load_json(self.index_path, {'schema_version': self.version, 'classes': {}})
        if 'classes' not in data:
            data = {'schema_version': self.version, 'classes': {}}
        return data

    def _save_index(self, data):
        data['schema_version'] = self.version
        self._save_json(self.index_path, data)

    def _append_jsonl(self, path, item):
        item = dict(item)
        item.setdefault('time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        item.setdefault('a2k_version', self.version)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def _append_history(self, item):
        self._append_jsonl(self.history_path, item)

    def update_project_meta(self, project_name=None, classes=None):
        data = self._load_json(self.project_meta_path, {})
        if project_name:
            data.setdefault('project_name', project_name)
        data['last_used_by'] = f'a2k_v{self.version}'
        data['schema_version'] = self.version
        if classes is not None:
            data['classes'] = list(classes)
        data['last_updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._save_json(self.project_meta_path, data)
        return data

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

    def _class_index_record(self, label, scanned=None, existing=None):
        safe = _safe_name(label)
        scanned = dict(scanned or self._scan_class_state(safe))
        existing = dict(existing or {})
        return {
            'storage_name': safe,
            'label': existing.get('label', str(label)),
            'display_name': existing.get('display_name', str(label)),
            'next_id': max(int(existing.get('next_id', 1)), int(scanned.get('next_id', 1))),
            'count': max(int(existing.get('count', 0)), int(scanned.get('count', 0))),
        }

    def ensure_for_classes(self, labels):
        index = self._load_index()
        for label in labels:
            safe = _safe_name(label)
            (self.train_dir / safe).mkdir(parents=True, exist_ok=True)
            scanned = self._scan_class_state(safe)
            index['classes'][safe] = self._class_index_record(
                label,
                scanned=scanned,
                existing=index['classes'].get(safe),
            )
        self._save_index(index)
        return index

    def rebuild_index_from_train(self):
        index = {'schema_version': self.version, 'classes': {}}
        if self.train_dir.exists():
            for class_dir in sorted([p for p in self.train_dir.iterdir() if p.is_dir()]):
                index['classes'][class_dir.name] = self._class_index_record(
                    class_dir.name,
                    scanned=self._scan_class_state(class_dir.name),
                )
        self._save_index(index)
        return index

    def _class_file_plan(self, label, ext):
        safe = _safe_name(label)
        index = self._load_index()
        index['classes'][safe] = self._class_index_record(
            label,
            scanned=self._scan_class_state(safe),
            existing=index['classes'].get(safe),
        )
        next_id = int(index['classes'][safe].get('next_id', 1))
        dst_dir = self.train_dir / safe
        dst_dir.mkdir(parents=True, exist_ok=True)
        while True:
            width = 3 if next_id < 1000 else len(str(next_id))
            dst = dst_dir / f'{safe}_{next_id:0{width}d}{ext.lower()}'
            if not dst.exists():
                break
            next_id += 1
        return safe, index, next_id, dst

    def _commit_class_file(self, safe, index, next_id):
        index['classes'][safe]['next_id'] = next_id + 1
        index['classes'][safe]['count'] = int(index['classes'][safe].get('count', 0)) + 1
        self._save_index(index)

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

    def _copy_file(self, src, dst):
        src = Path(src)
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return str(dst)

    def add_confirmed(self, image_path, label):
        src = Path(image_path)
        safe, index, next_id, dst = self._class_file_plan(label, src.suffix)
        saved = self._move_file(src, dst)
        self._commit_class_file(safe, index, next_id)
        self._append_history({'action': 'confirmed', 'class': safe, 'old_name': src.name, 'saved_as': saved})
        return saved

    def copy_confirmed(self, image_path, label, source='bootstrap_cluster'):
        src = Path(image_path)
        safe, index, next_id, dst = self._class_file_plan(label, src.suffix)
        saved = self._copy_file(src, dst)
        self._commit_class_file(safe, index, next_id)
        self._append_history({
            'action': 'confirmed_copy',
            'source': source,
            'class': safe,
            'old_name': src.name,
            'saved_as': saved,
        })
        return saved

    def add_candidate(self, image_path, label):
        src = Path(image_path)
        safe = _safe_name(label)
        dst = self._unique_pool_file(self.candidate_dir / safe, f'{safe}_candidate', src.suffix)
        saved = self._move_file(src, dst)
        self._append_history({'action': 'candidate', 'class': safe, 'old_name': src.name, 'saved_as': saved})
        return saved

    def add_rejected(self, image_path, reason='rejected'):
        src = Path(image_path)
        prefix = _safe_name(reason or 'rejected')
        dst = self._unique_pool_file(self.rejected_dir, prefix, src.suffix)
        saved = self._move_file(src, dst)
        self._append_history({'action': 'rejected', 'reason': prefix, 'old_name': src.name, 'saved_as': saved})
        return saved

    def add_unknown(self, image_path):
        src = Path(image_path)
        dst = self._unique_pool_file(self.unknown_dir, 'unknown', src.suffix)
        saved = self._move_file(src, dst)
        self._append_history({'action': 'unknown', 'old_name': src.name, 'saved_as': saved})
        return saved

    def normalize_unlabeled(self):
        return self._normalize_flat_images(
            self.unlabeled_dir,
            'img',
            self.import_map_path,
        )

    def normalize_unknown(self):
        """Rename new/non-standard learning samples in datasets/unknown."""
        return self._normalize_flat_images(
            self.learning_unknown_dir,
            'unknown',
            self.unknown_import_map_path,
        )

    def _normalize_flat_images(self, base_dir, prefix, map_path):
        """Rename only new/non-standard flat image files in a dataset folder."""
        base_dir = Path(base_dir)
        if not base_dir.exists():
            return []
        files = sorted([p for p in base_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        if not files:
            return []

        pattern = re.compile(rf'^{re.escape(prefix)}_(\d+)\.[^.]+$', re.IGNORECASE)
        used_ids = set()
        nonstandard = []
        for src in files:
            m = pattern.match(src.name)
            if m:
                used_ids.add(int(m.group(1)))
            else:
                nonstandard.append(src)
        if not nonstandard:
            return []

        next_id = max(used_ids) + 1 if used_ids else 1
        temp_files = []
        for i, src in enumerate(nonstandard, 1):
            temp = base_dir / f'__a2k_tmp_{i:06d}{src.suffix.lower()}'
            while temp.exists():
                temp = base_dir / f'__a2k_tmp_{i:06d}_{datetime.now().strftime("%f")}{src.suffix.lower()}'
            shutil.move(str(src), str(temp))
            temp_files.append((src, temp))

        mappings = []
        for old_src, temp in temp_files:
            ext = temp.suffix.lower()
            while next_id in used_ids:
                next_id += 1
            dst = base_dir / f'{prefix}_{next_id:03d}{ext}'
            while dst.exists():
                used_ids.add(next_id)
                next_id += 1
                dst = base_dir / f'{prefix}_{next_id:03d}{ext}'
            shutil.move(str(temp), str(dst))
            used_ids.add(next_id)
            record = {'old_name': old_src.name, 'new_name': dst.name, 'path': str(dst)}
            mappings.append(record)
            self._append_jsonl(map_path, record)
            next_id += 1
        return mappings

    def normalize_train_images(self, labels=None):
        """Normalize existing train/class files to class_001, class_002... safely.

        This lets users drag arbitrary filenames into train/<class>/ without manual renaming.
        Existing class_001 style files are preserved when possible; non-standard files are appended.
        """
        changed = []
        class_dirs = []
        if labels:
            class_dirs = [self.train_dir / _safe_name(x) for x in labels]
        elif self.train_dir.exists():
            class_dirs = [p for p in self.train_dir.iterdir() if p.is_dir()]

        for class_dir in sorted(class_dirs):
            if not class_dir.exists() or not class_dir.is_dir():
                continue
            safe = _safe_name(class_dir.name)
            pattern = re.compile(rf'^{re.escape(safe)}_(\d+)\.[^.]+$', re.IGNORECASE)
            files = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
            nonstandard = [p for p in files if not pattern.match(p.name)]
            if not nonstandard:
                continue

            # Scan current numbered files, then append non-standard files after max id.
            state = self._scan_class_state(safe)
            next_id = int(state.get('next_id', 1))
            for src in nonstandard:
                temp = class_dir / f'__a2k_train_tmp_{datetime.now().strftime("%H%M%S_%f")}{src.suffix.lower()}'
                shutil.move(str(src), str(temp))
                while True:
                    width = 3 if next_id < 1000 else len(str(next_id))
                    dst = class_dir / f'{safe}_{next_id:0{width}d}{temp.suffix.lower()}'
                    if not dst.exists():
                        break
                    next_id += 1
                shutil.move(str(temp), str(dst))
                record = {'action': 'normalize_train', 'class': safe, 'old_name': src.name, 'new_name': dst.name, 'saved_as': str(dst)}
                changed.append(record)
                self._append_jsonl(self.train_normalize_map_path, record)
                next_id += 1

        self.rebuild_index_from_train()
        return changed
