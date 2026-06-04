from pathlib import Path

import cv2
import numpy as np


def _l2_normalize(vec):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr
    return arr / norm


def _crop_relative(img, box):
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, int(round(w * float(box[0])))))
    y1 = max(0, min(h - 1, int(round(h * float(box[1])))))
    x2 = max(x1 + 1, min(w, int(round(w * float(box[2])))))
    y2 = max(y1 + 1, min(h, int(round(h * float(box[3])))))
    return img[y1:y2, x1:x2]


def _resize_flat(mask, size=28):
    resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (resized / 255.0).reshape(-1)


def _projection(mask, bins=28):
    small = cv2.resize(mask, (bins, bins), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    rows = small.mean(axis=1)
    cols = small.mean(axis=0)
    return np.concatenate([rows, cols]).astype(np.float32)


def _crop_descriptor(crop):
    if crop is None or crop.size == 0:
        return np.zeros(28 * 28 * 4 + 56 * 4, dtype=np.float32)

    crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    eq = cv2.equalizeHist(gray)

    dark = np.where((gray < 105) & (val < 150), 255, 0).astype(np.uint8)
    light = np.where((gray > 150) & (sat < 85), 255, 0).astype(np.uint8)
    saturated = np.where((sat > 80) & (val > 70), 255, 0).astype(np.uint8)
    edges = cv2.Canny(eq, 55, 145)

    kernel = np.ones((2, 2), np.uint8)
    masks = []
    for mask in (dark, light, saturated, edges):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        masks.append(mask)

    flats = [_resize_flat(mask) for mask in masks]
    projections = [_projection(mask) for mask in masks]
    return _l2_normalize(np.concatenate(flats + projections).astype(np.float32))


def extract_field_shape_descriptor(path):
    img = cv2.imread(str(Path(path)))
    if img is None:
        return None
    crops = [
        _crop_relative(img, (0.16, 0.16, 0.84, 0.84)),
        _crop_relative(img, (0.24, 0.22, 0.76, 0.78)),
        _crop_relative(img, (0.30, 0.26, 0.70, 0.74)),
    ]
    parts = [_crop_descriptor(crop) for crop in crops]
    return _l2_normalize(np.concatenate(parts).astype(np.float32))
