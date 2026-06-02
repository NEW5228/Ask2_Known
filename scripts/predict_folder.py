import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ask2know.deployment import load_deployment_bundle, predict_with_loaded_bundle
from ask2know.utils.io_utils import save_json


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def image_paths(input_path, recursive=False):
    input_path = Path(input_path).expanduser().resolve()
    if input_path.is_file():
        return [input_path]
    pattern = '**/*' if recursive else '*'
    return sorted(
        path for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def flatten_row(result, top_k):
    predictions = result.get('predictions') or []
    row = {
        'image': result.get('image'),
        'predicted_label': result.get('predicted_label'),
        'score': result.get('score'),
        'confidence': result.get('confidence'),
        'top2_margin': result.get('top2_margin'),
    }
    for idx in range(max(1, int(top_k))):
        pred = predictions[idx] if idx < len(predictions) else {}
        rank = idx + 1
        row[f'top{rank}_label'] = pred.get('label')
        row[f'top{rank}_score'] = pred.get('score')
    return row


def write_csv(path, rows, top_k):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['image', 'predicted_label', 'score', 'confidence', 'top2_margin']
    for idx in range(1, max(1, int(top_k)) + 1):
        fieldnames.extend([f'top{idx}_label', f'top{idx}_score'])
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Predict every image in a folder with an exported Ask2Know model bundle.')
    parser.add_argument('--model', required=True, help='Path to .a2kmodel.json')
    parser.add_argument('--input', required=True, help='Image file or folder to classify')
    parser.add_argument('--output', required=True, help='Output .csv or .json path')
    parser.add_argument('--top-k', type=int, default=5, help='Number of predictions to keep per image')
    parser.add_argument('--recursive', action='store_true', help='Scan input folder recursively')
    parser.add_argument('--cache-dir', help='Optional CLIP feature cache directory for deployment runtime')
    parser.add_argument('--json', action='store_true', help='Write full JSON instead of flat CSV')
    args = parser.parse_args()

    paths = image_paths(args.input, recursive=args.recursive)
    if not paths:
        print('No images found.')
        return 1

    model, weights, bundle = load_deployment_bundle(args.model, cache_dir=args.cache_dir)
    results = []
    flat_rows = []
    for index, path in enumerate(paths, 1):
        result = predict_with_loaded_bundle(model, weights, bundle, args.model, path, top_k=args.top_k)
        results.append(result)
        flat_rows.append(flatten_row(result, args.top_k))
        print(f'[{index}/{len(paths)}] {path.name}: {result.get("predicted_label")} {float(result.get("score", 0.0)):.6f}')

    output_path = Path(args.output).expanduser().resolve()
    if args.json or output_path.suffix.lower() == '.json':
        save_json(output_path, {
            'model': str(Path(args.model).expanduser().resolve()),
            'input': str(Path(args.input).expanduser().resolve()),
            'count': len(results),
            'results': results,
        })
    else:
        write_csv(output_path, flat_rows, args.top_k)
    print('Saved predictions:', output_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
