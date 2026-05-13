from pathlib import Path
import shutil
from datetime import datetime


class SamplePoolManager:
    def __init__(self, project_root=None, output_dir=None):
        if project_root:
            self.base_dir = Path(project_root) / 'sample_pools'
        else:
            self.base_dir = Path(output_dir) / 'sample_pools'
        self.confirmed_dir = self.base_dir / 'confirmed'
        self.candidate_dir = self.base_dir / 'candidate'
        self.rejected_dir = self.base_dir / 'rejected'
        for d in [self.confirmed_dir, self.candidate_dir, self.rejected_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _copy(self, src, dst_dir, label=None):
        src = Path(src)
        label_part = label if label else 'unknown'
        dst_dir = Path(dst_dir) / label_part
        dst_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        dst = dst_dir / f'{label_part}_{ts}{src.suffix.lower()}'
        try:
            shutil.copy2(src, dst)
            return str(dst)
        except Exception:
            return str(src)

    def add_confirmed(self, image_path, label):
        return self._copy(image_path, self.confirmed_dir, label)

    def add_candidate(self, image_path, label):
        return self._copy(image_path, self.candidate_dir, label)

    def add_rejected(self, image_path, reason='rejected'):
        safe = str(reason).replace(' ', '_')[:32] or 'rejected'
        return self._copy(image_path, self.rejected_dir, safe)
