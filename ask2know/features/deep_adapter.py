import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_DEEP_FEATURE_NAME = 'image_embedding'


def _l2_normalize(vec):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm


class DeepFeatureAdapter:
    """Required CLIP image embedding adapter for Ask2Know v0.4.2.

    v0.4.2 keeps OpenCLIP as the production embedding path. OpenCV embedding from
    v0.4.0 is kept only as private legacy code and is not an accepted provider.
    """

    def __init__(self, config=None, cache_dir=None):
        self.config = dict(config or {})
        self.enabled = bool(self.config.get('enable', False))
        self.provider = str(self.config.get('provider', 'open_clip')).strip().lower()
        self.feature_name = str(self.config.get('feature_name', DEFAULT_DEEP_FEATURE_NAME))
        self.fallback_to_opencv = bool(self.config.get('fallback_to_opencv', False))
        self.include_augmented = bool(self.config.get('include_augmented', False))
        self.cache_enabled = bool(self.config.get('cache', True))
        raw_cache_dir = self.config.get('cache_dir') or cache_dir
        self.cache_dir = Path(raw_cache_dir) if raw_cache_dir else None
        self._runtime = None
        self._runtime_error = None
        if self.enabled and self.cache_enabled and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def is_enabled(self):
        return self.enabled

    def metadata(self):
        return {
            'enable': self.enabled,
            'provider': self.provider,
            'feature_name': self.feature_name,
            'fallback_to_opencv': self.fallback_to_opencv,
            'include_augmented': self.include_augmented,
            'cache': self.cache_enabled and self.cache_dir is not None,
            'runtime_error': self._runtime_error,
        }

    def extract_path(self, path):
        if not self.enabled:
            return {}
        path = Path(path)
        cached = self._load_cache(path)
        if cached is not None:
            return {self.feature_name: cached}
        img = cv2.imread(str(path))
        if img is None:
            return {}
        vec = self.extract_image_vector(img)
        self._save_cache(path, vec)
        return {self.feature_name: vec}

    def extract_image(self, img):
        if not self.enabled:
            return {}
        return {self.feature_name: self.extract_image_vector(img)}

    def extract_image_vector(self, img):
        if img is None:
            return np.zeros(1, dtype=np.float32)
        if self.provider in ('clip', 'open_clip'):
            return self._try_external_provider(img, self._open_clip_embedding)
        raise ValueError(
            f'Deep feature provider {self.provider!r} is not supported in Ask2Know v0.4.2. '
            'Use provider: open_clip. OpenCV fallback is intentionally disabled.'
        )

    def _try_external_provider(self, img, extractor):
        try:
            return extractor(img)
        except Exception as exc:
            self._runtime_error = str(exc)
            raise RuntimeError(
                'CLIP provider is required but unavailable. Install torch, torchvision, '
                'pillow, and open_clip_torch, and make sure the configured OpenCLIP '
                'model weights are cached or downloadable.'
            ) from exc

    def _open_clip_embedding(self, img):
        if self._runtime is None:
            import torch
            import open_clip
            from PIL import Image

            requested = str(self.config.get('device', 'auto')).lower()
            if requested == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = requested
            model_name = self.config.get('model_name', 'ViT-B-32')
            pretrained = self.config.get('pretrained', 'laion2b_s34b_b79k')
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            model.eval()
            model.to(device)
            self._runtime = {
                'torch': torch,
                'image_cls': Image,
                'model': model,
                'preprocess': preprocess,
                'device': device,
            }

        rt = self._runtime
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = rt['image_cls'].fromarray(rgb)
        tensor = rt['preprocess'](pil).unsqueeze(0).to(rt['device'])
        with rt['torch'].no_grad():
            vec = rt['model'].encode_image(tensor)
        return _l2_normalize(vec.detach().cpu().numpy()[0])

    def _transformers_embedding(self, img):
        if self._runtime is None:
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModel

            requested = str(self.config.get('device', 'auto')).lower()
            if requested == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = requested
            model_name = self.config.get('model_name', 'facebook/dinov2-base')
            local_only = bool(self.config.get('local_files_only', True))
            processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=local_only)
            model = AutoModel.from_pretrained(model_name, local_files_only=local_only)
            model.eval()
            model.to(device)
            self._runtime = {
                'torch': torch,
                'image_cls': Image,
                'processor': processor,
                'model': model,
                'device': device,
            }

        rt = self._runtime
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = rt['image_cls'].fromarray(rgb)
        inputs = rt['processor'](images=pil, return_tensors='pt')
        inputs = {k: v.to(rt['device']) for k, v in inputs.items()}
        with rt['torch'].no_grad():
            outputs = rt['model'](**inputs)
        if getattr(outputs, 'pooler_output', None) is not None:
            vec = outputs.pooler_output[0]
        else:
            vec = outputs.last_hidden_state[:, 0, :][0]
        return _l2_normalize(vec.detach().cpu().numpy())

    def _opencv_embedding(self, img):
        resized = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] /= 179.0
        hsv[:, :, 1:] /= 255.0

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 140).astype(np.float32) / 255.0
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap = np.clip((lap + 255.0) / 510.0, 0.0, 1.0)

        color_hist = []
        full_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for channel, bins, rng in ((0, 12, (0, 180)), (1, 8, (0, 256)), (2, 8, (0, 256))):
            hist = cv2.calcHist([full_hsv], [channel], None, [bins], rng).reshape(-1)
            hist = hist / max(float(hist.sum()), 1.0)
            color_hist.append(hist.astype(np.float32))

        vec = np.concatenate([
            rgb.reshape(-1),
            hsv.reshape(-1),
            edges.reshape(-1),
            lap.reshape(-1),
            np.concatenate(color_hist),
        ]).astype(np.float32)
        return _l2_normalize(vec)

    def _cache_key(self, path):
        if not path.exists():
            return None
        stat = path.stat()
        raw = '|'.join([
            str(path.resolve()),
            str(int(stat.st_mtime_ns)),
            str(stat.st_size),
            self.provider,
            self.feature_name,
        ])
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()

    def _cache_path(self, path):
        if not (self.cache_enabled and self.cache_dir):
            return None
        key = self._cache_key(path)
        if not key:
            return None
        return self.cache_dir / f'{key}.json'

    def _load_cache(self, path):
        cache_path = self._cache_path(path)
        if not cache_path or not cache_path.exists():
            return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('provider') != self.provider or data.get('feature_name') != self.feature_name:
                return None
            return np.asarray(data.get('vector', []), dtype=np.float32)
        except Exception:
            return None

    def _save_cache(self, path, vec):
        cache_path = self._cache_path(path)
        if not cache_path:
            return
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'provider': self.provider,
                    'feature_name': self.feature_name,
                    'vector': np.asarray(vec, dtype=np.float32).tolist(),
                }, f)
        except Exception:
            return
