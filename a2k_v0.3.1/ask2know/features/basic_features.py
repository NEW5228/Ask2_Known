import cv2
import numpy as np


def _safe_read(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f'Cannot read image: {path}')
    return img


def _fallback_mask(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Try to remove very bright/very dark flat backgrounds.
    _, mask1 = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, mask2 = cv2.threshold(sat, 28, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    area_ratio = cv2.countNonZero(mask) / float(h * w)
    if area_ratio < 0.05 or area_ratio > 0.92:
        mask = np.zeros((h, w), dtype=np.uint8)
        margin_x = int(w * 0.08)
        margin_y = int(h * 0.08)
        mask[margin_y:h - margin_y, margin_x:w - margin_x] = 255
    return mask


def _main_mask(img):
    # GrabCut often works better than plain threshold on real images.
    h, w = img.shape[:2]
    rect = (int(w * 0.06), int(h * 0.06), int(w * 0.88), int(h * 0.88))
    try:
        mask = np.zeros((h, w), np.uint8)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        area_ratio = cv2.countNonZero(mask) / float(h * w)
        if 0.04 <= area_ratio <= 0.90:
            return mask
    except Exception:
        pass
    return _fallback_mask(img)


def _largest_component_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    c = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [c], -1, 255, -1)
    return out


def color_feature(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [18, 8], [0, 180, 0, 256])
    hist = hist.astype(np.float32).flatten()
    hist = hist / (hist.sum() + 1e-8)
    # Add rough color statistics to avoid pure histogram ties.
    pixels = hsv[mask > 0]
    if pixels.size == 0:
        stats = np.zeros(6, dtype=np.float32)
    else:
        mean = pixels.mean(axis=0) / np.array([180.0, 255.0, 255.0])
        std = pixels.std(axis=0) / np.array([180.0, 255.0, 255.0])
        stats = np.concatenate([mean, std]).astype(np.float32)
    return np.concatenate([hist, stats]).astype(np.float32)


def size_feature(img, mask):
    h, w = mask.shape[:2]
    area_pixels = float(cv2.countNonZero(mask))
    area = area_pixels / max(1.0, float(h * w))
    x, y, bw, bh = cv2.boundingRect(mask)
    aspect = float(bw) / max(1.0, float(bh))
    extent = area_pixels / max(1.0, float(bw * bh))
    box_w = float(bw) / max(1.0, float(w))
    box_h = float(bh) / max(1.0, float(h))
    # image size is not real-world size; this feature should usually keep low weight.
    return np.array([area, aspect, extent, box_w, box_h], dtype=np.float32)


def contour_feature(img, mask):
    mask = _largest_component_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(12, dtype=np.float32)
    c = max(contours, key=cv2.contourArea)
    area = max(1.0, cv2.contourArea(c))
    peri = max(1.0, cv2.arcLength(c, True))
    x, y, w, h = cv2.boundingRect(c)
    rect_area = max(1.0, float(w * h))
    hull = cv2.convexHull(c)
    hull_area = max(1.0, cv2.contourArea(hull))

    circularity = 4 * np.pi * area / (peri * peri + 1e-6)
    extent = area / rect_area
    solidity = area / hull_area
    aspect = float(w) / max(1.0, float(h))
    area_ratio = area / max(1.0, float(mask.shape[0] * mask.shape[1]))
    elongated = max(aspect, 1.0 / max(aspect, 1e-6))

    moments = cv2.HuMoments(cv2.moments(c)).flatten()
    hu = -np.sign(moments[:4]) * np.log10(np.abs(moments[:4]) + 1e-12)
    hu = np.clip(hu / 10.0, -1.0, 1.0)

    # Approx polygon complexity helps separate long/simple shapes from round/blob shapes.
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    complexity = min(len(approx) / 20.0, 1.0)
    bbox_fill = extent
    return np.array([
        circularity, extent, solidity, aspect, area_ratio, elongated,
        bbox_fill, complexity, hu[0], hu[1], hu[2], hu[3]
    ], dtype=np.float32)


def texture_feature(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_area = max(1.0, float(cv2.countNonZero(mask)))
    edges = cv2.Canny(gray, 60, 150)
    edge_density = float(cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))) / masked_area
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_vals = lap[mask > 0]
    lap_var = float(lap_vals.var()) if lap_vals.size else 0.0
    lap_var = min(lap_var / 1500.0, 1.0)
    vals = gray[mask > 0]
    if vals.size == 0:
        mean = std = entropy = 0.0
    else:
        mean = float(vals.mean()) / 255.0
        std = float(vals.std()) / 128.0
        hist = np.bincount(vals.astype(np.uint8), minlength=256).astype(np.float32)
        p = hist / (hist.sum() + 1e-8)
        entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum() / 8.0)
    return np.array([edge_density, lap_var, mean, min(std, 1.0), entropy], dtype=np.float32)


def quality_feature(img, mask):
    area_ratio = cv2.countNonZero(mask) / float(mask.shape[0] * mask.shape[1])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0, 1.0)
    return np.array([area_ratio, blur_score], dtype=np.float32)


def extract_features(path):
    img = _safe_read(path)
    img = cv2.resize(img, (256, 256))
    mask = _main_mask(img)
    mask = _largest_component_mask(mask)
    return {
        'color': color_feature(img, mask),
        'size': size_feature(img, mask),
        'contour': contour_feature(img, mask),
        'texture': texture_feature(img, mask),
        'quality': quality_feature(img, mask),
    }
