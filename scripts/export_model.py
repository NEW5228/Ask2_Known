import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ask2know.deployment import build_deployment_bundle, build_deployment_bundle_from_model_cache


def main():
    parser = argparse.ArgumentParser(description='Export a trained Ask2Know project as a deployable model bundle.')
    parser.add_argument('--config', required=True, help='Project task_config.yaml')
    parser.add_argument('--output', help='Output .a2kmodel.json path. Defaults to outputs/<task>.a2kmodel.json.')
    parser.add_argument('--model-cache', help='Use an existing fitted prototype_model_cache.json instead of fitting from train data.')
    parser.add_argument(
        '--no-sample-features',
        action='store_true',
        help='Do not embed training sample features in the bundle. This makes the file smaller but disables kNN/crop evidence.',
    )
    args = parser.parse_args()

    if args.model_cache:
        output_path, bundle = build_deployment_bundle_from_model_cache(
            args.config,
            args.model_cache,
            output_path=args.output,
            include_sample_features=not args.no_sample_features,
        )
    else:
        output_path, bundle = build_deployment_bundle(
            args.config,
            output_path=args.output,
            include_sample_features=not args.no_sample_features,
        )
    model = bundle.get('model') or {}
    sample_count = sum(len(paths) for paths in (model.get('sample_index') or {}).values())
    print('Exported model:', output_path)
    print('Schema:', bundle.get('schema_version'))
    print('Classes:', ', '.join(item.get('name', '') for item in bundle.get('classes', [])))
    print('Training samples:', sample_count)
    validation = bundle.get('validation_status')
    if validation:
        passed = bool(validation.get('passed', False))
        print('Validation passed' if passed else 'Validation failed')
    print('Predict one image: python scripts\\predict_model.py --model', output_path, '--image path\\to\\image.jpg')
    print('Predict a folder: python scripts\\predict_folder.py --model', output_path, '--input path\\to\\images --output predictions.csv')
    print('Runtime dependency: Python environment with OpenCV, NumPy, torch, and open_clip_torch.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
