import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ask2know.deployment import predict_with_bundle


def main():
    parser = argparse.ArgumentParser(description='Predict an image with an exported Ask2Know deployment bundle.')
    parser.add_argument('--model', required=True, help='Path to .a2kmodel.json')
    parser.add_argument('--image', required=True, help='Image path to classify')
    parser.add_argument('--top-k', type=int, default=5, help='Number of predictions to print')
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON')
    parser.add_argument('--cache-dir', help='Optional CLIP feature cache directory for deployment runtime')
    args = parser.parse_args()

    result = predict_with_bundle(args.model, args.image, top_k=args.top_k, cache_dir=args.cache_dir)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print('Image:', result['image'])
    predictions = result.get('predictions') or []
    if not predictions:
        print('No predictions.')
        return 1
    print('Top predictions:')
    for index, row in enumerate(predictions, 1):
        print(f'{index}. {row["label"]}: {row["score"]:.6f}')
        sources = row.get('sources') or {}
        if sources:
            print('   sources:', ', '.join(f'{key}={value:.6f}' for key, value in sources.items()))
        nearest = row.get('nearest_samples') or []
        if nearest:
            print(f'   nearest: {nearest[0]["score"]:.6f} {nearest[0]["path"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
