import cv2
import numpy as np

def _safe_read(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f'Cannot read image: {path}')
    return img

def _main_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def color_feature(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [16, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)

def size_feature(img, mask):
    h, w = mask.shape[:2]
    area = float(cv2.countNonZero(mask)) / max(1.0, float(h * w))
    x, y, bw, bh = cv2.boundingRect(mask)
    aspect = float(bw) / max(1.0, float(bh))
    extent = float(cv2.countNonZero(mask)) / max(1.0, float(bw * bh))
    return np.array([area, aspect, extent], dtype=np.float32)

def contour_feature(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(7, dtype=np.float32)
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    circularity = 4 * np.pi * area / (peri * peri + 1e-6)
    rect_area = max(1.0, float(w * h))
    extent = area / rect_area
    hull = cv2.convexHull(c)
    hull_area = max(1.0, cv2.contourArea(hull))
    solidity = area / hull_area
    moments = cv2.HuMoments(cv2.moments(c)).flatten()
    hu = -np.sign(moments[:2]) * np.log10(np.abs(moments[:2]) + 1e-12)
    return np.array([circularity, extent, solidity, w / max(1.0, h), area / (mask.shape[0] * mask.shape[1]), hu[0], hu[1]], dtype=np.float32)

def texture_feature(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))) / max(1.0, float(cv2.countNonZero(mask)))
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0
    lap_var = min(lap_var, 10.0) / 10.0
    return np.array([edge_density, lap_var], dtype=np.float32)

def extract_features(path):
    img = _safe_read(path)
    img = cv2.resize(img, (256, 256))
    mask = _main_mask(img)
    return {
        'color': color_feature(img, mask),
        'size': size_feature(img, mask),
        'contour': contour_feature(img, mask),
        'texture': texture_feature(img, mask),
    }
