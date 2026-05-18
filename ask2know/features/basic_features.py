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
    # Disable GrabCut by default for stability.
    # It caused occasional Windows hangs in earlier tests when combined with preview/file moving.
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
    pixels = hsv[mask > 0]
    if pixels.size == 0:
        stats = np.zeros(6, dtype=np.float32)
        color_bins = np.zeros(14, dtype=np.float32)
    else:
        mean = pixels.mean(axis=0) / np.array([180.0, 255.0, 255.0])
        std = pixels.std(axis=0) / np.array([180.0, 255.0, 255.0])
        stats = np.concatenate([mean, std]).astype(np.float32)
        h = pixels[:, 0]
        s = pixels[:, 1]
        v = pixels[:, 2]
        valid = (s > 35) & (v > 30)
        hv = h[valid]
        total = max(1, hv.size)
        # coarse semantic color ratios; still generic, not fruit-specific.
        red = ((hv < 10) | (hv > 168)).sum() / total
        orange = ((hv >= 10) & (hv < 25)).sum() / total
        yellow = ((hv >= 25) & (hv < 38)).sum() / total
        green = ((hv >= 38) & (hv < 82)).sum() / total
        cyan = ((hv >= 82) & (hv < 100)).sum() / total
        blue = ((hv >= 100) & (hv < 125)).sum() / total
        purple = ((hv >= 125) & (hv < 155)).sum() / total
        pink = ((hv >= 155) & (hv <= 168)).sum() / total
        brown = ((h >= 5) & (h < 30) & (s > 35) & (v >= 45) & (v < 150)).sum() / max(1, pixels.shape[0])
        black = ((pixels[:, 2] < 65) & (pixels[:, 1] < 170)).sum() / max(1, pixels.shape[0])
        white = ((pixels[:, 2] > 205) & (pixels[:, 1] < 45)).sum() / max(1, pixels.shape[0])
        gray = ((pixels[:, 2] >= 65) & (pixels[:, 2] <= 205) & (pixels[:, 1] < 45)).sum() / max(1, pixels.shape[0])
        dark = ((pixels[:, 2] < 90) & (pixels[:, 1] > 30)).sum() / max(1, pixels.shape[0])
        bright = ((pixels[:, 2] > 180) & (pixels[:, 1] > 30)).sum() / max(1, pixels.shape[0])
        color_bins = np.array([
            red, orange, yellow, green, cyan, blue, purple, pink,
            brown, black, white, gray, dark, bright
        ], dtype=np.float32)
    return np.concatenate([hist, stats, color_bins]).astype(np.float32)


def size_feature(img, mask):
    h, w = mask.shape[:2]
    area_pixels = float(cv2.countNonZero(mask))
    area = area_pixels / max(1.0, float(h * w))
    x, y, bw, bh = cv2.boundingRect(mask)
    aspect = float(bw) / max(1.0, float(bh))
    extent = area_pixels / max(1.0, float(bw * bh))
    box_w = float(bw) / max(1.0, float(w))
    box_h = float(bh) / max(1.0, float(h))
    return np.array([area, aspect, extent, box_w, box_h], dtype=np.float32)


def contour_feature(img, mask):
    mask = _largest_component_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(15, dtype=np.float32)
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
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    complexity = min(len(approx) / 20.0, 1.0)

    # Split top/bottom width for rough pear-like shape signal.
    roi = mask[y:y+h, x:x+w]
    if roi.size == 0 or h < 4:
        top_width = bottom_width = 0.0
    else:
        top = roi[:max(1, h // 2), :]
        bottom = roi[max(1, h // 2):, :]
        top_cols = np.where(top.max(axis=0) > 0)[0]
        bottom_cols = np.where(bottom.max(axis=0) > 0)[0]
        top_width = (top_cols[-1] - top_cols[0] + 1) / max(1.0, float(w)) if top_cols.size else 0.0
        bottom_width = (bottom_cols[-1] - bottom_cols[0] + 1) / max(1.0, float(w)) if bottom_cols.size else 0.0
    pear_ratio = bottom_width / max(top_width, 1e-6)
    pear_ratio = min(pear_ratio / 3.0, 1.0)

    return np.array([
        circularity, extent, solidity, aspect, area_ratio, elongated,
        extent, complexity, hu[0], hu[1], hu[2], hu[3], top_width, bottom_width, pear_ratio
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


def fruit_color_feature(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]
    if pixels.size == 0:
        return np.zeros(19, dtype=np.float32)

    h = pixels[:, 0]
    s = pixels[:, 1]
    v = pixels[:, 2]
    valid = (s > 35) & (v > 30)
    hv = h[valid]
    total = max(1, hv.size)

    red = ((hv < 10) | (hv > 168)).sum() / total
    orange = ((hv >= 10) & (hv < 25)).sum() / total
    yellow = ((hv >= 25) & (hv < 38)).sum() / total
    green = ((hv >= 38) & (hv < 82)).sum() / total
    cyan = ((hv >= 82) & (hv < 100)).sum() / total
    blue = ((hv >= 100) & (hv < 125)).sum() / total
    purple = ((hv >= 125) & (hv < 155)).sum() / total
    pink = ((hv >= 155) & (hv <= 168)).sum() / total
    brown_dark = ((h >= 5) & (h < 28) & (s > 50) & (v < 130)).sum() / max(1, pixels.shape[0])
    black = ((v < 65) & (s < 170)).sum() / max(1, pixels.shape[0])
    white = ((v > 205) & (s < 45)).sum() / max(1, pixels.shape[0])
    gray = ((v >= 65) & (v <= 205) & (s < 45)).sum() / max(1, pixels.shape[0])
    bright_patch = ((v > 205) & (s > 25)).sum() / max(1, pixels.shape[0])
    high_sat = (s > 110).sum() / max(1, pixels.shape[0])
    color_bins = np.array([red, orange, yellow, green, cyan, blue, purple, pink, brown_dark, black, white, gray], dtype=np.float32)
    dominance = float(color_bins.max()) if color_bins.size else 0.0
    stats = np.array([
        float(s.mean()) / 255.0,
        float(v.mean()) / 255.0,
        min(float(s.std()) / 128.0, 1.0),
        min(float(v.std()) / 128.0, 1.0),
    ], dtype=np.float32)
    return np.concatenate([
        color_bins,
        np.array([bright_patch, high_sat, dominance], dtype=np.float32),
        stats,
    ]).astype(np.float32)


def fruit_shape_feature(img, mask):
    mask = _largest_component_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(11, dtype=np.float32)

    c = max(contours, key=cv2.contourArea)
    area = max(1.0, cv2.contourArea(c))
    peri = max(1.0, cv2.arcLength(c, True))
    x, y, w, h = cv2.boundingRect(c)
    rect_area = max(1.0, float(w * h))
    hull_area = max(1.0, cv2.contourArea(cv2.convexHull(c)))
    aspect = float(w) / max(1.0, float(h))
    circularity = 4 * np.pi * area / (peri * peri + 1e-6)
    elongated = max(aspect, 1.0 / max(aspect, 1e-6))
    elongated_score = min((elongated - 1.0) / 2.6, 1.0)
    solidity = area / hull_area
    extent = area / rect_area

    roi = mask[y:y+h, x:x+w]
    if roi.size == 0 or h < 4:
        top_width = bottom_width = symmetry = curve_signal = 0.0
    else:
        top = roi[:max(1, h // 2), :]
        bottom = roi[max(1, h // 2):, :]
        top_cols = np.where(top.max(axis=0) > 0)[0]
        bottom_cols = np.where(bottom.max(axis=0) > 0)[0]
        top_width = (top_cols[-1] - top_cols[0] + 1) / max(1.0, float(w)) if top_cols.size else 0.0
        bottom_width = (bottom_cols[-1] - bottom_cols[0] + 1) / max(1.0, float(w)) if bottom_cols.size else 0.0
        flipped = cv2.flip(roi, 1)
        overlap = cv2.countNonZero(cv2.bitwise_and(roi, flipped))
        union = cv2.countNonZero(cv2.bitwise_or(roi, flipped))
        symmetry = float(overlap) / max(1.0, float(union))

        centers_y = []
        centers_x = []
        for row in range(h):
            cols = np.where(roi[row, :] > 0)[0]
            if cols.size:
                centers_y.append(row / max(1.0, float(h - 1)))
                centers_x.append(float(cols.mean()) / max(1.0, float(w - 1)))
        if len(centers_y) >= 8:
            coef = np.polyfit(np.asarray(centers_y), np.asarray(centers_x), 2)
            curve_signal = min(abs(float(coef[0])) * 2.5, 1.0)
        else:
            curve_signal = 0.0

    pear_ratio = min((bottom_width / max(top_width, 1e-6)) / 3.0, 1.0)
    round_score = min(circularity, 1.0) * (1.0 - min(abs(np.log(max(aspect, 1e-6))) / np.log(3.0), 1.0))
    banana_like = min(0.6 * elongated_score + 0.4 * curve_signal, 1.0)

    return np.array([
        round_score,
        elongated_score,
        pear_ratio,
        min(aspect / 3.0, 1.0),
        min((1.0 / max(aspect, 1e-6)) / 3.0, 1.0),
        min(solidity, 1.0),
        min(extent, 1.0),
        top_width,
        bottom_width,
        symmetry,
        banana_like,
    ], dtype=np.float32)


def fruit_texture_feature(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    vals = gray[mask > 0]
    pixels = hsv[mask > 0]
    if vals.size == 0 or pixels.size == 0:
        return np.zeros(7, dtype=np.float32)

    masked_area = max(1.0, float(cv2.countNonZero(mask)))
    edges = cv2.Canny(gray, 45, 135)
    edge_density = float(cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))) / masked_area
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_vals = lap[mask > 0]
    lap_var = min(float(lap_vals.var()) / 1500.0, 1.0) if lap_vals.size else 0.0
    local_mean = cv2.blur(gray, (9, 9))
    local_delta = cv2.absdiff(gray, local_mean)
    local_contrast = min(float(local_delta[mask > 0].mean()) / 48.0, 1.0)
    hist = np.bincount(vals.astype(np.uint8), minlength=256).astype(np.float32)
    p = hist / (hist.sum() + 1e-8)
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum() / 8.0)

    h = pixels[:, 0]
    s = pixels[:, 1]
    v = pixels[:, 2]
    dark_spots = ((v < 95) & (s > 35)).sum() / max(1, pixels.shape[0])
    bright_spots = ((v > 205) & (s < 95)).sum() / max(1, pixels.shape[0])
    roughness = min(0.35 * edge_density * 2.5 + 0.35 * lap_var + 0.30 * local_contrast, 1.0)
    return np.array([
        min(edge_density * 2.5, 1.0),
        lap_var,
        local_contrast,
        min(entropy, 1.0),
        dark_spots,
        bright_spots,
        roughness,
    ], dtype=np.float32)


def fruit_structure_feature(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_area = max(1.0, float(cv2.countNonZero(mask)))
    edges = cv2.Canny(gray, 45, 135)
    edge_mask = cv2.bitwise_and(edges, edges, mask=mask)
    edge_density = float(cv2.countNonZero(edge_mask)) / masked_area

    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_round = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 8 or area > masked_area * 0.12:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 1:
            continue
        circularity = 4 * np.pi * area / (peri * peri + 1e-6)
        if circularity > 0.35:
            small_round += 1
    small_round_norm = min(small_round / 18.0, 1.0)

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    foreground_components = 0
    for idx in range(1, component_count):
        if stats[idx, cv2.CC_STAT_AREA] >= masked_area * 0.015:
            foreground_components += 1
    component_signal = min(foreground_components / 6.0, 1.0)

    contours_main, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_main:
        c = max(contours_main, key=cv2.contourArea)
        hull_area = max(1.0, cv2.contourArea(cv2.convexHull(c)))
        solidity = cv2.contourArea(c) / hull_area
    else:
        solidity = 0.0

    repeated_parts = min(0.55 * small_round_norm + 0.45 * min(edge_density * 2.8, 1.0), 1.0)
    cluster_like = min(0.45 * repeated_parts + 0.35 * component_signal + 0.20 * (1.0 - min(solidity, 1.0)), 1.0)
    single_object = max(0.0, 1.0 - cluster_like)
    return np.array([
        cluster_like,
        repeated_parts,
        small_round_norm,
        min(edge_density * 2.8, 1.0),
        single_object,
        min(solidity, 1.0),
    ], dtype=np.float32)


def surface_mark_feature(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    vals = gray[mask > 0]
    pixels = hsv[mask > 0]
    if vals.size == 0 or pixels.size == 0:
        return np.zeros(10, dtype=np.float32)

    masked_area = max(1.0, float(cv2.countNonZero(mask)))
    edges = cv2.Canny(gray, 35, 120)
    edge_density = float(cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))) / masked_area
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_vals = lap[mask > 0]
    lap_var = min(float(lap_vals.var()) / 1300.0, 1.0) if lap_vals.size else 0.0
    local_mean = cv2.blur(gray, (7, 7))
    local_delta = cv2.absdiff(gray, local_mean)
    local_contrast = min(float(local_delta[mask > 0].mean()) / 42.0, 1.0)

    h = pixels[:, 0]
    s = pixels[:, 1]
    v = pixels[:, 2]
    brown_hair = ((h >= 5) & (h < 35) & (s > 30) & (v > 35) & (v < 160)).sum() / max(1, pixels.shape[0])
    dark_hair = ((v < 100) & (s > 20)).sum() / max(1, pixels.shape[0])
    bright_patch = ((v > 210) & (s < 90)).sum() / max(1, pixels.shape[0])

    _, dark = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, dark, mask=mask)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    small_spots = 0
    spot_area = 0.0
    for idx in range(1, component_count):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if 3 <= area <= masked_area * 0.012:
            small_spots += 1
            spot_area += area
    small_spot_density = min(small_spots / 45.0, 1.0)
    spot_area_ratio = min(spot_area / max(masked_area * 0.10, 1.0), 1.0)

    rough_peel = min(0.35 * min(edge_density * 4.0, 1.0) + 0.35 * lap_var + 0.30 * local_contrast, 1.0)
    fuzzy_surface = min(0.30 * rough_peel + 0.25 * min(edge_density * 5.0, 1.0) + 0.25 * brown_hair * 2.0 + 0.20 * dark_hair, 1.0)
    speckled_surface = min(0.55 * small_spot_density + 0.30 * spot_area_ratio + 0.15 * local_contrast, 1.0)
    glossy_surface = min(0.65 * bright_patch * 3.0 + 0.20 * (1.0 - rough_peel) + 0.15 * (1.0 - min(edge_density * 6.0, 1.0)), 1.0)

    return np.array([
        fuzzy_surface,
        rough_peel,
        speckled_surface,
        glossy_surface,
        min(edge_density * 5.0, 1.0),
        lap_var,
        local_contrast,
        min(brown_hair * 2.0, 1.0),
        small_spot_density,
        min(bright_patch * 3.0, 1.0),
    ], dtype=np.float32)


def fruit_part_feature(img, mask):
    mask = _largest_component_mask(mask)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_area = max(1.0, float(cv2.countNonZero(mask)))
    pixels = hsv[mask > 0]
    if pixels.size == 0:
        return np.zeros(10, dtype=np.float32)

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.zeros(10, dtype=np.float32)
    cx = float(xs.mean())
    cy = float(ys.mean())
    dist = np.sqrt((xs.astype(np.float32) - cx) ** 2 + (ys.astype(np.float32) - cy) ** 2)
    max_dist = max(float(dist.max()), 1.0)
    norm = dist / max_dist

    ring_map = np.zeros_like(mask, dtype=np.float32)
    ring_map[ys, xs] = norm
    outer_mask = np.zeros_like(mask)
    mid_mask = np.zeros_like(mask)
    inner_mask = np.zeros_like(mask)
    outer_mask[(mask > 0) & (ring_map >= 0.72)] = 255
    mid_mask[(mask > 0) & (ring_map >= 0.42) & (ring_map < 0.72)] = 255
    inner_mask[(mask > 0) & (ring_map < 0.42)] = 255

    def _mean_hsv(region_mask):
        vals = hsv[region_mask > 0]
        if vals.size == 0:
            return np.zeros(3, dtype=np.float32)
        return vals.mean(axis=0).astype(np.float32)

    outer = _mean_hsv(outer_mask)
    inner = _mean_hsv(inner_mask)
    color_contrast = min(float(np.linalg.norm((outer - inner) / np.array([90.0, 128.0, 128.0], dtype=np.float32))) / 1.25, 1.0)
    inner_area = cv2.countNonZero(inner_mask) / masked_area
    outer_area = cv2.countNonZero(outer_mask) / masked_area

    edges = cv2.Canny(gray, 40, 130)
    inner_edges = float(cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=inner_mask))) / max(1.0, float(cv2.countNonZero(inner_mask)))
    all_edges = float(cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))) / masked_area
    inner_bright = ((inner_mask > 0) & (gray > 145)).sum() / max(1.0, float(cv2.countNonZero(inner_mask)))
    inner_sat = hsv[:, :, 1][inner_mask > 0].mean() / 255.0 if cv2.countNonZero(inner_mask) else 0.0
    outer_dark = hsv[:, :, 2][outer_mask > 0].mean() / 255.0 if cv2.countNonZero(outer_mask) else 0.0
    outer_dark = 1.0 - float(outer_dark)

    peel_h = hsv[:, :, 0][outer_mask > 0]
    peel_s = hsv[:, :, 1][outer_mask > 0]
    peel_v = hsv[:, :, 2][outer_mask > 0]
    if peel_h.size:
        outer_green_brown = (
            (((peel_h >= 5) & (peel_h < 35) & (peel_s > 35) & (peel_v < 175)).sum()
             + ((peel_h >= 35) & (peel_h < 85) & (peel_s > 35)).sum())
            / max(1, peel_h.size)
        )
    else:
        outer_green_brown = 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = max(1.0, cv2.contourArea(c))
        peri = max(1.0, cv2.arcLength(c, True))
        circularity = min(4 * np.pi * area / (peri * peri + 1e-6), 1.0)
    else:
        circularity = 0.0

    _, dark = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    dark_inner = cv2.bitwise_and(dark, dark, mask=cv2.bitwise_or(inner_mask, mid_mask))
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(dark_inner, 8)
    seed_count = 0
    seed_area = 0.0
    for idx in range(1, component_count):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if 3 <= area <= masked_area * 0.018:
            seed_count += 1
            seed_area += area
    seed_density = min(seed_count / 28.0, 1.0)
    seed_area_ratio = min(seed_area / max(masked_area * 0.08, 1.0), 1.0)

    center_mask = np.zeros_like(mask)
    center_mask[(mask > 0) & (ring_map < 0.22)] = 255
    center_vals = hsv[center_mask > 0]
    center_contrast = 0.0
    if center_vals.size and pixels.size:
        center_mean = center_vals.mean(axis=0).astype(np.float32)
        global_mean = pixels.mean(axis=0).astype(np.float32)
        center_contrast = min(float(np.linalg.norm((center_mean - global_mean) / np.array([90.0, 128.0, 128.0], dtype=np.float32))) / 1.1, 1.0)

    line_mask = cv2.bitwise_and(edges, edges, mask=cv2.bitwise_or(inner_mask, mid_mask))
    lines = cv2.HoughLinesP(line_mask, 1, np.pi / 180, threshold=18, minLineLength=20, maxLineGap=5)
    radial_signal = 0.0
    if lines is not None:
        radial = 0
        total = 0
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(x) for x in line]
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length < 1.0:
                continue
            mx = (x1 + x2) * 0.5
            my = (y1 + y2) * 0.5
            vx = mx - cx
            vy = my - cy
            radial_len = float(np.hypot(vx, vy))
            if radial_len < 1.0:
                continue
            cosine = abs((dx * vx + dy * vy) / (length * radial_len))
            total += 1
            if cosine > 0.72:
                radial += 1
        radial_signal = min((radial / max(1, total)) * min(total / 8.0, 1.0), 1.0)

    boundary_evidence = max(0.0, min(1.0, (color_contrast - 0.34) / 0.66))
    structure_evidence = max(
        seed_density,
        radial_signal,
        max(0.0, min(1.0, (center_contrast - 0.36) / 0.64)),
    )
    visible_inner = min(inner_area * 4.0, 1.0) * max(boundary_evidence, 0.65 * structure_evidence)

    peel_like = min(0.38 * color_contrast + 0.28 * outer_green_brown + 0.20 * outer_dark + 0.14 * min(outer_area * 2.2, 1.0), 1.0)
    raw_flesh = 0.55 * inner_bright + 0.45 * inner_sat
    flesh_like = min(raw_flesh * visible_inner * (0.35 + 0.65 * structure_evidence), 1.0)
    seed_like = min(0.58 * seed_density + 0.32 * seed_area_ratio + 0.10 * min(all_edges * 5.0, 1.0), 1.0)
    cut_surface = min(
        (0.50 * boundary_evidence + 0.32 * structure_evidence + 0.18 * circularity)
        * (0.30 + 0.70 * max(structure_evidence, seed_like)),
        1.0,
    )
    core_like = min(
        (0.70 * max(0.0, min(1.0, (center_contrast - 0.30) / 0.70)) + 0.30 * radial_signal)
        * (0.30 + 0.70 * boundary_evidence),
        1.0,
    )
    segment_like = min(radial_signal * (0.35 + 0.65 * max(boundary_evidence, cut_surface)), 1.0)
    rind_like = min((0.62 * peel_like + 0.38 * min(outer_area * 3.0, 1.0)) * (0.45 + 0.55 * boundary_evidence), 1.0)

    return np.array([
        peel_like,
        flesh_like,
        cut_surface,
        seed_like,
        core_like,
        segment_like,
        rind_like,
        color_contrast,
        seed_density,
        radial_signal,
    ], dtype=np.float32)


def animal_shape_feature(img, mask):
    mask = _largest_component_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(9, dtype=np.float32)

    c = max(contours, key=cv2.contourArea)
    area = max(1.0, cv2.contourArea(c))
    peri = max(1.0, cv2.arcLength(c, True))
    x, y, w, h = cv2.boundingRect(c)
    aspect = float(w) / max(1.0, float(h))
    circularity = min(4 * np.pi * area / (peri * peri + 1e-6), 1.0)
    solidity = area / max(1.0, cv2.contourArea(cv2.convexHull(c)))
    elongated = min((max(aspect, 1.0 / max(aspect, 1e-6)) - 1.0) / 2.5, 1.0)

    roi = mask[y:y+h, x:x+w]
    if roi.size == 0 or h < 6 or w < 6:
        top_protrusion = top_balance = head_body_ratio = symmetry = 0.0
        lower_bulk = upper_bulk = 0.0
    else:
        top = roi[:max(1, h // 3), :]
        mid = roi[max(1, h // 3):max(2, (2 * h) // 3), :]
        bottom = roi[max(2, (2 * h) // 3):, :]
        top_area = cv2.countNonZero(top) / max(1.0, float(top.size))
        mid_area = cv2.countNonZero(mid) / max(1.0, float(mid.size))
        bottom_area = cv2.countNonZero(bottom) / max(1.0, float(bottom.size))
        upper_bulk = min(top_area / max(mid_area, 1e-6), 1.0)
        lower_bulk = min(bottom_area / max(mid_area, 1e-6), 1.0)
        head_body_ratio = min((top_area + 0.5 * mid_area) / max(bottom_area + 0.5 * mid_area, 1e-6), 1.0)

        top_cols = np.where(top.max(axis=0) > 0)[0]
        mid_cols = np.where(mid.max(axis=0) > 0)[0]
        if top_cols.size and mid_cols.size:
            top_width = (top_cols[-1] - top_cols[0] + 1) / max(1.0, float(w))
            mid_width = (mid_cols[-1] - mid_cols[0] + 1) / max(1.0, float(w))
            left_top = (top[:, :w // 2].max(axis=0) > 0).sum() / max(1.0, float(w // 2))
            right_top = (top[:, w // 2:].max(axis=0) > 0).sum() / max(1.0, float(w - w // 2))
            top_balance = 1.0 - min(abs(left_top - right_top), 1.0)
            top_protrusion = min(max(mid_width - top_width, 0.0) * 2.4 + max(left_top, right_top) * 0.35, 1.0)
        else:
            top_protrusion = top_balance = 0.0

        flipped = cv2.flip(roi, 1)
        overlap = cv2.countNonZero(cv2.bitwise_and(roi, flipped))
        union = cv2.countNonZero(cv2.bitwise_or(roi, flipped))
        symmetry = float(overlap) / max(1.0, float(union))

    compact_body = min(0.50 * circularity + 0.30 * min(solidity, 1.0) + 0.20 * (1.0 - elongated), 1.0)
    pet_outline = min(0.34 * compact_body + 0.26 * top_protrusion + 0.22 * symmetry + 0.18 * head_body_ratio, 1.0)
    return np.array([
        pet_outline,
        top_protrusion,
        top_balance,
        head_body_ratio,
        symmetry,
        compact_body,
        elongated,
        upper_bulk,
        lower_bulk,
    ], dtype=np.float32)


def fur_texture_feature(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    vals = gray[mask > 0]
    pixels = hsv[mask > 0]
    if vals.size == 0 or pixels.size == 0:
        return np.zeros(8, dtype=np.float32)

    masked_area = max(1.0, float(cv2.countNonZero(mask)))
    edges = cv2.Canny(gray, 30, 110)
    edge_density = float(cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))) / masked_area
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = min(float(lap[mask > 0].var()) / 1200.0, 1.0)
    local_mean = cv2.blur(gray, (5, 5))
    local_delta = cv2.absdiff(gray, local_mean)
    fine_contrast = min(float(local_delta[mask > 0].mean()) / 34.0, 1.0)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobel_x, sobel_y)
    angle = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)
    selected = (mask > 0) & (mag > np.percentile(mag[mask > 0], 58))
    if selected.any():
        hist, _ = np.histogram(angle[selected] % 180.0, bins=6, range=(0.0, 180.0), weights=mag[selected])
        orientation_strength = float(hist.max() / (hist.sum() + 1e-8))
    else:
        orientation_strength = 0.0

    h = pixels[:, 0]
    s = pixels[:, 1]
    v = pixels[:, 2]
    dark_fur = ((v < 105) & (s < 190)).sum() / max(1, pixels.shape[0])
    light_fur = ((v > 175) & (s < 95)).sum() / max(1, pixels.shape[0])
    brown_fur = ((h >= 5) & (h < 35) & (s > 25) & (v > 35) & (v < 190)).sum() / max(1, pixels.shape[0])
    fur_like = min(
        0.30 * min(edge_density * 5.5, 1.0)
        + 0.25 * fine_contrast
        + 0.20 * lap_var
        + 0.15 * orientation_strength
        + 0.10 * min((dark_fur + light_fur + brown_fur) * 1.7, 1.0),
        1.0,
    )
    return np.array([
        fur_like,
        min(edge_density * 5.5, 1.0),
        lap_var,
        fine_contrast,
        orientation_strength,
        dark_fur,
        light_fur,
        brown_fur,
    ], dtype=np.float32)


def animal_face_feature(img, mask):
    mask = _largest_component_mask(mask)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return np.zeros(10, dtype=np.float32)

    x, y, w, h = cv2.boundingRect(mask)
    roi_mask = mask[y:y+h, x:x+w]
    roi_gray = gray[y:y+h, x:x+w]
    roi_hsv = hsv[y:y+h, x:x+w]
    if roi_mask.size == 0 or h < 12 or w < 12:
        return np.zeros(10, dtype=np.float32)

    flipped = cv2.flip(roi_mask, 1)
    overlap = cv2.countNonZero(cv2.bitwise_and(roi_mask, flipped))
    union = cv2.countNonZero(cv2.bitwise_or(roi_mask, flipped))
    symmetry = float(overlap) / max(1.0, float(union))

    upper_mask = np.zeros_like(roi_mask)
    upper_mask[:max(1, int(h * 0.58)), :] = roi_mask[:max(1, int(h * 0.58)), :]
    center_mask = np.zeros_like(roi_mask)
    x0, x1 = int(w * 0.30), int(w * 0.70)
    y0, y1 = int(h * 0.34), int(h * 0.74)
    center_mask[y0:y1, x0:x1] = roi_mask[y0:y1, x0:x1]

    _, dark = cv2.threshold(roi_gray, 88, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.bitwise_and(dark, dark, mask=upper_mask)
    component_count, _, stats, centers = cv2.connectedComponentsWithStats(dark, 8)
    eye_candidates = []
    masked_area = max(1.0, float(cv2.countNonZero(roi_mask)))
    for idx in range(1, component_count):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < 2 or area > masked_area * 0.035:
            continue
        cx, cy = centers[idx]
        if cy > h * 0.62:
            continue
        eye_candidates.append((float(cx), float(cy), area))

    eye_pair = 0.0
    eye_balance = 0.0
    for i, left in enumerate(eye_candidates):
        for right in eye_candidates[i + 1:]:
            x_left, y_left, a_left = left
            x_right, y_right, a_right = right
            dx = abs(x_right - x_left) / max(1.0, float(w))
            dy = abs(y_right - y_left) / max(1.0, float(h))
            if dx < 0.16 or dx > 0.62 or dy > 0.16:
                continue
            area_balance = 1.0 - min(abs(a_left - a_right) / max(a_left, a_right, 1.0), 1.0)
            center_balance = 1.0 - min(abs(((x_left + x_right) * 0.5 / max(1.0, float(w))) - 0.5) * 2.0, 1.0)
            candidate_score = min(0.45 * (dx / 0.62) + 0.30 * area_balance + 0.25 * center_balance, 1.0)
            if candidate_score > eye_pair:
                eye_pair = candidate_score
                eye_balance = min(0.55 * area_balance + 0.45 * center_balance, 1.0)

    center_vals = roi_hsv[center_mask > 0]
    all_vals = roi_hsv[roi_mask > 0]
    if center_vals.size and all_vals.size:
        center_mean = center_vals.mean(axis=0).astype(np.float32)
        all_mean = all_vals.mean(axis=0).astype(np.float32)
        muzzle_contrast = min(float(np.linalg.norm((center_mean - all_mean) / np.array([90.0, 128.0, 128.0], dtype=np.float32))) / 1.1, 1.0)
        center_v = center_vals[:, 2]
        center_s = center_vals[:, 1]
        muzzle_light = ((center_v > 150) & (center_s < 115)).sum() / max(1, center_vals.shape[0])
        nose_dark = (center_v < 75).sum() / max(1, center_vals.shape[0])
    else:
        muzzle_contrast = muzzle_light = nose_dark = 0.0
    muzzle_like = min(0.45 * muzzle_contrast + 0.30 * muzzle_light + 0.25 * nose_dark * 2.0, 1.0)

    edges = cv2.Canny(roi_gray, 35, 120)
    edge_center = cv2.bitwise_and(edges, edges, mask=center_mask)
    lines = cv2.HoughLinesP(edge_center, 1, np.pi / 180, threshold=12, minLineLength=max(8, w // 10), maxLineGap=4)
    horizontal_lines = 0
    total_lines = 0
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in line]
            angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
            total_lines += 1
            if angle < 22 or angle > 158:
                horizontal_lines += 1
    whisker_like = min((horizontal_lines / max(1, total_lines)) * min(total_lines / 6.0, 1.0), 1.0)

    top_mask = roi_mask[:max(1, int(h * 0.25)), :]
    mid_mask = roi_mask[max(1, int(h * 0.25)):max(2, int(h * 0.55)), :]
    top_cols = np.where(top_mask.max(axis=0) > 0)[0]
    mid_cols = np.where(mid_mask.max(axis=0) > 0)[0]
    if top_cols.size and mid_cols.size:
        top_width = (top_cols[-1] - top_cols[0] + 1) / max(1.0, float(w))
        mid_width = (mid_cols[-1] - mid_cols[0] + 1) / max(1.0, float(w))
        ear_like = min(max(mid_width - top_width, 0.0) * 2.8 + top_width * 0.25, 1.0)
    else:
        ear_like = 0.0

    face_like = min(
        0.25 * symmetry
        + 0.22 * eye_pair
        + 0.18 * muzzle_like
        + 0.14 * ear_like
        + 0.11 * whisker_like
        + 0.10 * eye_balance,
        1.0,
    )
    return np.array([
        face_like,
        symmetry,
        eye_pair,
        eye_balance,
        muzzle_like,
        muzzle_contrast,
        nose_dark,
        whisker_like,
        ear_like,
        min(len(eye_candidates) / 8.0, 1.0),
    ], dtype=np.float32)


def text_mark_feature(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_area = max(1.0, float(cv2.countNonZero(mask)))
    edges = cv2.Canny(gray, 55, 155)
    edge_mask = cv2.bitwise_and(edges, edges, mask=mask)
    edge_density = float(cv2.countNonZero(edge_mask)) / masked_area

    _, binary_dark = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY_INV)
    binary_dark = cv2.bitwise_and(binary_dark, binary_dark, mask=mask)
    kernel = np.ones((2, 2), np.uint8)
    binary_dark = cv2.morphologyEx(binary_dark, cv2.MORPH_OPEN, kernel)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary_dark, 8)
    small_components = 0
    medium_components = 0
    total_dark_area = 0.0
    for idx in range(1, component_count):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < 3 or area > masked_area * 0.18:
            continue
        total_dark_area += area
        if area <= masked_area * 0.018:
            small_components += 1
        else:
            medium_components += 1

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    horizontal = cv2.morphologyEx(binary_dark, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(binary_dark, cv2.MORPH_OPEN, v_kernel)
    horizontal_density = float(cv2.countNonZero(horizontal)) / masked_area
    vertical_density = float(cv2.countNonZero(vertical)) / masked_area

    ys, xs = np.where(binary_dark > 0)
    if xs.size:
        x_spread = min(float(xs.std()) / max(1.0, float(mask.shape[1]) * 0.22), 1.0)
        y_spread = min(float(ys.std()) / max(1.0, float(mask.shape[0]) * 0.18), 1.0)
    else:
        x_spread = y_spread = 0.0
    stroke_balance = 1.0 - min(abs(horizontal_density - vertical_density) / max(horizontal_density + vertical_density, 1e-6), 1.0)
    text_like = min(
        0.30 * min(edge_density * 5.0, 1.0)
        + 0.25 * min((small_components + medium_components) / 18.0, 1.0)
        + 0.20 * min(total_dark_area / max(masked_area * 0.18, 1.0), 1.0)
        + 0.15 * stroke_balance
        + 0.10 * min(x_spread + y_spread, 1.0),
        1.0,
    )
    return np.array([
        text_like,
        min(edge_density * 5.0, 1.0),
        min(small_components / 16.0, 1.0),
        min(medium_components / 8.0, 1.0),
        min(horizontal_density * 8.0, 1.0),
        min(vertical_density * 8.0, 1.0),
        x_spread,
        y_spread,
    ], dtype=np.float32)


def sign_symbol_feature(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_area = max(1.0, float(cv2.countNonZero(mask)))
    pixels = hsv[mask > 0]
    if pixels.size == 0:
        return np.zeros(10, dtype=np.float32)

    h = pixels[:, 0]
    s = pixels[:, 1]
    v = pixels[:, 2]
    red_ratio = (((h < 10) | (h > 165)) & (s > 70) & (v > 50)).sum() / max(1, pixels.shape[0])
    blue_ratio = ((h >= 85) & (h < 125) & (s > 70) & (v > 50)).sum() / max(1, pixels.shape[0])
    high_contrast = ((s > 65) & ((v < 90) | (v > 180))).sum() / max(1, pixels.shape[0])

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = max(1.0, cv2.contourArea(c))
        peri = max(1.0, cv2.arcLength(c, True))
        circularity = min(4 * np.pi * area / (peri * peri + 1e-6), 1.0)
        x, y, w, h_box = cv2.boundingRect(c)
        aspect = float(w) / max(1.0, float(h_box))
        aspect_balance = 1.0 - min(abs(np.log(max(aspect, 1e-6))) / np.log(3.0), 1.0)
        circle_like = min(circularity * aspect_balance, 1.0)
    else:
        circle_like = 0.0

    edges = cv2.Canny(gray, 45, 135)
    edge_mask = cv2.bitwise_and(edges, edges, mask=mask)
    lines = cv2.HoughLinesP(edge_mask, 1, np.pi / 180, threshold=22, minLineLength=18, maxLineGap=5)
    horizontal = vertical = diag_pos = diag_neg = 0.0
    total_len = 0.0
    if lines is not None:
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(x) for x in line]
            dx = x2 - x1
            dy = y2 - y1
            length = max(1.0, float(np.hypot(dx, dy)))
            angle = abs(float(np.degrees(np.arctan2(dy, dx))))
            total_len += length
            if angle < 18 or angle > 162:
                horizontal += length
            elif 72 < angle < 108:
                vertical += length
            elif dy * dx >= 0:
                diag_pos += length
            else:
                diag_neg += length
    if total_len > 0:
        horizontal /= total_len
        vertical /= total_len
        diag_pos /= total_len
        diag_neg /= total_len

    ys, xs = np.where(mask > 0)
    if xs.size:
        cx = float(xs.mean()) / max(1.0, float(mask.shape[1] - 1))
        cy = float(ys.mean()) / max(1.0, float(mask.shape[0] - 1))
        directional_bias = min((abs(cx - 0.5) + abs(cy - 0.5)) * 2.0, 1.0)
    else:
        directional_bias = 0.0

    diagonal_signal = max(diag_pos, diag_neg)
    arrow_like = min(0.45 * horizontal + 0.25 * vertical + 0.30 * directional_bias, 1.0)
    prohibition_like = min(0.45 * red_ratio * 3.0 + 0.35 * circle_like + 0.20 * diagonal_signal, 1.0)
    sign_like = min(0.30 * high_contrast + 0.25 * max(red_ratio, blue_ratio) * 2.5 + 0.25 * circle_like + 0.20 * max(arrow_like, prohibition_like), 1.0)

    return np.array([
        sign_like,
        arrow_like,
        prohibition_like,
        min(red_ratio * 3.0, 1.0),
        min(blue_ratio * 3.0, 1.0),
        circle_like,
        horizontal,
        vertical,
        diag_pos,
        diag_neg,
    ], dtype=np.float32)


def extract_features_from_image(img):
    img = cv2.resize(img, (256, 256))
    mask = _main_mask(img)
    mask = _largest_component_mask(mask)
    return {
        'color': color_feature(img, mask),
        'size': size_feature(img, mask),
        'contour': contour_feature(img, mask),
        'texture': texture_feature(img, mask),
        'quality': quality_feature(img, mask),
        'fruit_color': fruit_color_feature(img, mask),
        'fruit_shape': fruit_shape_feature(img, mask),
        'fruit_texture': fruit_texture_feature(img, mask),
        'fruit_structure': fruit_structure_feature(img, mask),
        'surface_mark': surface_mark_feature(img, mask),
        'fruit_part': fruit_part_feature(img, mask),
        'animal_shape': animal_shape_feature(img, mask),
        'fur_texture': fur_texture_feature(img, mask),
        'animal_face': animal_face_feature(img, mask),
        'text_mark': text_mark_feature(img, mask),
        'sign_symbol': sign_symbol_feature(img, mask),
    }


def extract_features(path):
    img = _safe_read(path)
    return extract_features_from_image(img)
