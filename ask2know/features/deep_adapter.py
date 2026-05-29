import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_DEEP_FEATURE_NAME = 'image_embedding'


def _clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    return x1, y1, x2, y2


def _expand_box(box, width, height, pad_ratio=0.10):
    x1, y1, x2, y2 = box
    bw = max(1, int(x2) - int(x1))
    bh = max(1, int(y2) - int(y1))
    pad_x = int(round(bw * float(pad_ratio)))
    pad_y = int(round(bh * float(pad_ratio)))
    return _clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), width, height)


def _foreground_box(img):
    h, w = img.shape[:2]
    if h <= 1 or w <= 1:
        return 0, 0, w, h

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    edges = cv2.Canny(gray, 50, 140)

    border = max(2, int(round(min(h, w) * 0.06)))
    border_pixels = np.concatenate([
        img[:border, :, :].reshape(-1, 3),
        img[h - border:, :, :].reshape(-1, 3),
        img[:, :border, :].reshape(-1, 3),
        img[:, w - border:, :].reshape(-1, 3),
    ], axis=0).astype(np.float32)
    bg_color = np.median(border_pixels, axis=0)
    color_dist = np.linalg.norm(img.astype(np.float32) - bg_color.reshape(1, 1, 3), axis=2)
    color_mask = color_dist > max(18.0, float(np.percentile(color_dist, 68)))

    signal = (
        (sat > np.percentile(sat, 58))
        | (val < np.percentile(val, 34))
        | (edges > 0)
        | color_mask
    ).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    signal = cv2.morphologyEx(signal, cv2.MORPH_CLOSE, kernel, iterations=2)
    signal = cv2.morphologyEx(signal, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(signal, 8)
    min_area = max(32, int(round(h * w * 0.04)))
    best = None
    best_score = None
    cx_img, cy_img = w / 2.0, h / 2.0
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        if int(area) < min_area:
            continue
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        center_penalty = abs(cx - cx_img) / max(1.0, w) + 0.6 * abs(cy - cy_img) / max(1.0, h)
        score = float(area) / max(1.0, h * w) - 0.25 * center_penalty
        if best_score is None or score > best_score:
            best_score = score
            best = (int(x), int(y), int(x + bw), int(y + bh))

    if best is None:
        mx = int(round(w * 0.08))
        my = int(round(h * 0.08))
        best = (mx, my, w - mx, h - my)
    return _expand_box(best, w, h, pad_ratio=0.12)


def _head_box_from_object_box(obj_box, width, height):
    x1, y1, x2, y2 = obj_box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    head_w = int(round(bw * 0.72))
    head_h = int(round(bh * 0.52))
    cx = x1 + bw // 2
    hx1 = cx - head_w // 2
    hy1 = y1
    return _expand_box((hx1, hy1, hx1 + head_w, hy1 + head_h), width, height, pad_ratio=0.08)


def _crop_image_specs(img, crop_names=None, center_ratio=0.86, corner_ratio=0.72):
    if img is None:
        return []
    h, w = img.shape[:2]
    if h <= 1 or w <= 1:
        return []
    requested = list(crop_names or ['full', 'center', 'five_crop', 'object', 'head'])
    specs = []
    seen = set()
    object_box = None

    def add(crop_id, box):
        if crop_id in seen:
            return
        x1, y1, x2, y2 = _clamp_box(box, w, h)
        specs.append({
            'crop_id': crop_id,
            'box': [x1, y1, x2, y2],
            'image': img[y1:y2, x1:x2],
        })
        seen.add(crop_id)

    for name in requested:
        name = str(name).strip().lower()
        if name == 'full':
            add('full', (0, 0, w, h))
        elif name == 'center':
            cw = max(1, int(round(w * float(center_ratio))))
            ch = max(1, int(round(h * float(center_ratio))))
            x1 = (w - cw) // 2
            y1 = (h - ch) // 2
            add('center', (x1, y1, x1 + cw, y1 + ch))
        elif name == 'five_crop':
            cw = max(1, int(round(w * float(corner_ratio))))
            ch = max(1, int(round(h * float(corner_ratio))))
            add('top_left', (0, 0, cw, ch))
            add('top_right', (w - cw, 0, w, ch))
            add('bottom_left', (0, h - ch, cw, h))
            add('bottom_right', (w - cw, h - ch, w, h))
            x1 = (w - cw) // 2
            y1 = (h - ch) // 2
            add('five_center', (x1, y1, x1 + cw, y1 + ch))
        elif name in {'object', 'object_crop', 'foreground'}:
            if object_box is None:
                object_box = _foreground_box(img)
            add('object', object_box)
        elif name in {'head', 'head_crop', 'face'}:
            if object_box is None:
                object_box = _foreground_box(img)
            add('head', _head_box_from_object_box(object_box, w, h))
    return specs


def _l2_normalize(vec):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm


class DeepFeatureAdapter:
    """Required CLIP image embedding adapter for Ask2Know v0.4.61.0.

    v0.4.61.0 keeps OpenCLIP as the production embedding path. OpenCV embedding from
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
        self.multi_crop_config = dict(self.config.get('multi_crop', {}))
        self.multi_crop_enabled = bool(self.multi_crop_config.get('enable', False))
        self.multi_crop_names = list(self.multi_crop_config.get('crops') or ['full', 'center', 'five_crop', 'object', 'head'])
        self.multi_crop_center_ratio = float(self.multi_crop_config.get('center_ratio', 0.86))
        self.multi_crop_corner_ratio = float(self.multi_crop_config.get('corner_ratio', 0.72))
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
            'cache_fingerprint': self._cache_fingerprint(),
            'fallback_to_opencv': self.fallback_to_opencv,
            'include_augmented': self.include_augmented,
            'cache': self.cache_enabled and self.cache_dir is not None,
            'multi_crop': {
                'enable': self.multi_crop_enabled,
                'crops': self.multi_crop_names,
                'center_ratio': self.multi_crop_center_ratio,
                'corner_ratio': self.multi_crop_corner_ratio,
            },
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

    def extract_multi_crop_path(self, path):
        if not (self.enabled and self.multi_crop_enabled):
            return []
        path = Path(path)
        img = cv2.imread(str(path))
        if img is None:
            return []
        rows = []
        for spec in _crop_image_specs(
            img,
            crop_names=self.multi_crop_names,
            center_ratio=self.multi_crop_center_ratio,
            corner_ratio=self.multi_crop_corner_ratio,
        ):
            crop_id = spec['crop_id']
            vec = None
            if crop_id == 'full':
                vec = self._load_cache(path)
                if vec is None:
                    vec = self.extract_image_vector(spec['image'])
                    self._save_cache(path, vec)
            else:
                vec = self._load_crop_cache(path, crop_id, spec['box'])
                if vec is None:
                    vec = self.extract_image_vector(spec['image'])
                    self._save_crop_cache(path, crop_id, spec['box'], vec)
            rows.append({
                'crop_id': crop_id,
                'box': list(spec['box']),
                'vector': vec,
            })
        return rows

    def extract_image_vector(self, img):
        if img is None:
            return np.zeros(1, dtype=np.float32)
        if self.provider in ('clip', 'open_clip'):
            return self._try_external_provider(img, self._open_clip_embedding)
        raise ValueError(
            f'Deep feature provider {self.provider!r} is not supported in Ask2Know v0.4.61.0. '
            'Use provider: open_clip. OpenCV fallback is intentionally disabled.'
        )

    def extract_text_vectors(self, texts):
        if not self.enabled:
            return []
        if self.provider in ('clip', 'open_clip'):
            try:
                return self._open_clip_text_embeddings(texts)
            except Exception as exc:
                self._runtime_error = str(exc)
                raise RuntimeError(
                    'CLIP text provider is required but unavailable. Install torch, torchvision, '
                    'pillow, and open_clip_torch, and make sure the configured OpenCLIP '
                    'model weights are cached or downloadable.'
                ) from exc
        raise ValueError(
            f'Deep feature provider {self.provider!r} is not supported in Ask2Know v0.4.61.0. '
            'Use provider: open_clip.'
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
        rt = self._ensure_open_clip_runtime()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = rt['image_cls'].fromarray(rgb)
        tensor = rt['preprocess'](pil).unsqueeze(0).to(rt['device'])
        with rt['torch'].no_grad():
            vec = rt['model'].encode_image(tensor)
        return _l2_normalize(vec.detach().cpu().numpy()[0])

    def _open_clip_text_embeddings(self, texts):
        prompts = [str(text) for text in texts or [] if str(text).strip()]
        if not prompts:
            return []
        rt = self._ensure_open_clip_runtime()
        tokens = rt['tokenizer'](prompts).to(rt['device'])
        with rt['torch'].no_grad():
            vecs = rt['model'].encode_text(tokens)
        arr = vecs.detach().cpu().numpy()
        return [_l2_normalize(row) for row in arr]

    def _ensure_open_clip_runtime(self):
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
            tokenizer = open_clip.get_tokenizer(model_name)
            model.eval()
            model.to(device)
            self._runtime = {
                'torch': torch,
                'image_cls': Image,
                'model': model,
                'preprocess': preprocess,
                'tokenizer': tokenizer,
                'device': device,
            }
        return self._runtime

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

    def _cache_fingerprint(self):
        parts = {
            'cache_schema': 'deep_features_v2',
            'provider': self.provider,
            'feature_name': self.feature_name,
            'model_name': str(self.config.get('model_name', '')),
            'pretrained': str(self.config.get('pretrained', '')),
        }
        raw = json.dumps(parts, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()

    def _cache_key(self, path):
        if not path.exists():
            return None
        stat = path.stat()
        raw = '|'.join([
            str(path.resolve()),
            str(int(stat.st_mtime_ns)),
            str(stat.st_size),
            self._cache_fingerprint(),
        ])
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()

    def _cache_path(self, path):
        if not (self.cache_enabled and self.cache_dir):
            return None
        key = self._cache_key(path)
        if not key:
            return None
        return self.cache_dir / f'{key}.json'

    def _crop_cache_key(self, path, crop_id, box):
        if not path.exists():
            return None
        stat = path.stat()
        raw = '|'.join([
            str(path.resolve()),
            str(int(stat.st_mtime_ns)),
            str(stat.st_size),
            self._cache_fingerprint(),
            str(crop_id),
            ','.join(str(int(v)) for v in box),
        ])
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()

    def _crop_cache_path(self, path, crop_id, box):
        if not (self.cache_enabled and self.cache_dir):
            return None
        key = self._crop_cache_key(path, crop_id, box)
        if not key:
            return None
        return self.cache_dir / f'{key}.crop.json'

    def _load_cache(self, path):
        cache_path = self._cache_path(path)
        if not cache_path or not cache_path.exists():
            return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('provider') != self.provider or data.get('feature_name') != self.feature_name:
                return None
            if data.get('cache_fingerprint') != self._cache_fingerprint():
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
                    'cache_fingerprint': self._cache_fingerprint(),
                    'vector': np.asarray(vec, dtype=np.float32).tolist(),
                }, f)
        except Exception:
            return

    def _load_crop_cache(self, path, crop_id, box):
        cache_path = self._crop_cache_path(path, crop_id, box)
        if not cache_path or not cache_path.exists():
            return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('provider') != self.provider or data.get('feature_name') != self.feature_name:
                return None
            if data.get('cache_fingerprint') != self._cache_fingerprint():
                return None
            if data.get('crop_id') != crop_id:
                return None
            return np.asarray(data.get('vector', []), dtype=np.float32)
        except Exception:
            return None

    def _save_crop_cache(self, path, crop_id, box, vec):
        cache_path = self._crop_cache_path(path, crop_id, box)
        if not cache_path:
            return
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'provider': self.provider,
                    'feature_name': self.feature_name,
                    'cache_fingerprint': self._cache_fingerprint(),
                    'crop_id': crop_id,
                    'box': list(box),
                    'vector': np.asarray(vec, dtype=np.float32).tolist(),
                }, f)
        except Exception:
            return
