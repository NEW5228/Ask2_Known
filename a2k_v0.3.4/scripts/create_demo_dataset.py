from pathlib import Path
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'datasets' / 'fruit_demo'


def save(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def apple(path, seed=0, radius=60, color=(20, 30, 210)):
    rng = np.random.default_rng(seed)
    img = np.full((256, 256, 3), 255, dtype=np.uint8)
    center = (128 + int(rng.normal(0, 5)), 130 + int(rng.normal(0, 5)))
    cv2.circle(img, center, radius + int(rng.normal(0, 3)), color, -1)
    cv2.ellipse(img, (center[0]+20, center[1]-70), (18, 8), -30, 0, 360, (40, 130, 40), -1)
    cv2.rectangle(img, (center[0]-5, center[1]-75), (center[0]+5, center[1]-50), (30, 80, 30), -1)
    save(path, img)


def strawberry(path, seed=0, color=(30, 30, 210)):
    rng = np.random.default_rng(seed)
    img = np.full((256, 256, 3), 255, dtype=np.uint8)
    pts = np.array([[128, 55], [65, 120], [95, 210], [128, 230], [160, 210], [190, 120]], np.int32)
    pts = pts + rng.integers(-5, 6, size=pts.shape)
    cv2.fillPoly(img, [pts], color)
    for _ in range(35):
        x = int(rng.integers(85, 175))
        y = int(rng.integers(95, 210))
        if cv2.pointPolygonTest(pts, (x, y), False) >= 0:
            cv2.circle(img, (x, y), 2, (180, 210, 230), -1)
    leaf = np.array([[100, 60], [118, 80], [128, 55], [138, 80], [158, 60], [140, 92], [116, 92]], np.int32)
    cv2.fillPoly(img, [leaf], (40, 150, 50))
    save(path, img)


def main():
    write_json(DATA / 'objects.json', {
        'objects': [
            {
                'object_id': 'F001',
                'name': 'apple',
                'display_name': '苹果',
                'description': '通常较圆，表面较平滑，可能是红色、绿色或黄色'
            },
            {
                'object_id': 'F002',
                'name': 'strawberry',
                'display_name': '草莓',
                'description': '通常轮廓偏尖，表面纹理或颗粒感更明显'
            }
        ]
    })

    write_json(DATA / 'concepts.json', {
        'concepts': [
            {
                'object_a': 'apple',
                'object_b': 'strawberry',
                'hint': '苹果和草莓都可能偏红，颜色不是唯一依据；轮廓和纹理通常更有区分价值。',
                'important_features': ['contour', 'texture'],
                'weak_features': ['size']
            }
        ]
    })

    for i in range(5):
        apple(DATA / 'train' / 'apple' / f'apple_{i+1:03d}.jpg', seed=i, radius=58+i%3)
        strawberry(DATA / 'train' / 'strawberry' / f'strawberry_{i+1:03d}.jpg', seed=i)

    apple(DATA / 'unlabeled' / 'unknown_apple_red.jpg', seed=20, radius=62)
    apple(DATA / 'unlabeled' / 'unknown_apple_small.jpg', seed=21, radius=50, color=(35, 60, 200))
    strawberry(DATA / 'unlabeled' / 'unknown_strawberry_red.jpg', seed=22)
    strawberry(DATA / 'unlabeled' / 'unknown_strawberry_big.jpg', seed=23, color=(35, 20, 190))

    print('Demo dataset created:', DATA)
    print('Run: python run_demo.py --config configs/fruit_demo.yaml')


if __name__ == '__main__':
    main()
