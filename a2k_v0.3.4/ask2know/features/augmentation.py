import cv2
import numpy as np


def _enabled(cfg):
    return bool(cfg and cfg.get('enable', False))


def _limit(cfg):
    return int(cfg.get('max_augmented_per_image', 4))


def generate_augmented_images(img, cfg=None):
    """Return lightweight augmented image variants.

    v0.3.4 only uses gentle deterministic transforms. It does not create new files;
    it only provides extra views for prototype feature averaging.
    """
    if not _enabled(cfg):
        return []
    cfg = cfg or {}
    out = []
    h, w = img.shape[:2]

    if cfg.get('brightness', True):
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=22)
        dark = cv2.convertScaleAbs(img, alpha=1.0, beta=-22)
        out.extend([bright, dark])

    if cfg.get('rotation', True):
        for angle in (-8, 8):
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rot = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)
            out.append(rot)

    if cfg.get('scale_crop', True):
        margin = max(2, int(min(h, w) * 0.06))
        crop = img[margin:h - margin, margin:w - margin]
        if crop.size:
            crop = cv2.resize(crop, (w, h))
            out.append(crop)

    if cfg.get('blur', False):
        out.append(cv2.GaussianBlur(img, (3, 3), 0))

    return out[:_limit(cfg)]
