import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ask2know import __version__
from ask2know.utils.io_utils import load_json, save_json


PACKAGE_SCRIPTS = [
    'predict_model.py',
    'predict_folder.py',
]


def ignore_runtime_files(directory, names):
    ignored = set()
    for name in names:
        if name == '__pycache__' or name.endswith('.pyc') or name.endswith('.pyo'):
            ignored.add(name)
    return ignored


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_requirements(output_dir):
    source = ROOT / 'requirements.txt'
    target = output_dir / 'requirements.txt'
    if source.exists():
        copy_file(source, target)
        return
    target.write_text(
        '\n'.join([
            'opencv-python',
            'numpy',
            'pyyaml',
            'torch',
            'torchvision',
            'pillow',
            'open_clip_torch',
            'rich>=13.8,<14',
            '',
        ]),
        encoding='utf-8',
    )


def write_readme(output_dir, model_filename, include_server=False):
    server_section = ''
    if include_server:
        server_section = f'''
## Optional HTTP service

```powershell
python scripts\\serve_model.py --model {model_filename} --port 8000
```

Health check:

```text
GET http://127.0.0.1:8000/health
```

Predict:

```text
POST http://127.0.0.1:8000/predict
{{"image": "path\\to\\image.jpg", "top_k": 5}}
```
'''

    text = f'''# Ask2Know Offline Model Package

This directory is a self-contained Python deployment package for an exported Ask2Know model.

## Contents

- `{model_filename}`: exported Ask2Know model bundle.
- `ask2know/`: runtime package used by prediction scripts.
- `scripts/predict_model.py`: predict one image.
- `scripts/predict_folder.py`: predict a folder of images.
- `requirements.txt`: Python runtime dependencies.

## Install dependencies

Use Python 3.9+.

```powershell
pip install -r requirements.txt
```

## Predict one image

```powershell
python scripts\\predict_model.py --model {model_filename} --image path\\to\\image.jpg --top-k 5
```

Machine-readable JSON:

```powershell
python scripts\\predict_model.py --model {model_filename} --image path\\to\\image.jpg --top-k 5 --json
```

## Predict a folder

```powershell
python scripts\\predict_folder.py --model {model_filename} --input path\\to\\images --output predictions.csv --top-k 5
```

Recursive folder scan:

```powershell
python scripts\\predict_folder.py --model {model_filename} --input path\\to\\images --output predictions.csv --top-k 5 --recursive
```
{server_section}
## Output fields

Predictions include:

- `predicted_label`: final top-1 class.
- `confidence`: final Ask2Know score.
- `top2_margin`: score gap between top-1 and top-2.
- `predictions`: top-k candidates with evidence scores.
'''
    (output_dir / 'README.md').write_text(text, encoding='utf-8')


def build_package(model_path, output_dir, include_server=False, model_filename=None):
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f'Model file not found: {model_path}')
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f'Output directory already exists: {output_dir}')

    bundle = load_json(model_path)
    output_dir.mkdir(parents=True)
    model_filename = model_filename or model_path.name
    target_model = output_dir / model_filename
    copy_file(model_path, target_model)

    shutil.copytree(ROOT / 'ask2know', output_dir / 'ask2know', ignore=ignore_runtime_files)
    scripts_dir = output_dir / 'scripts'
    for script in PACKAGE_SCRIPTS:
        copy_file(ROOT / 'scripts' / script, scripts_dir / script)
    if include_server:
        copy_file(ROOT / 'scripts' / 'serve_model.py', scripts_dir / 'serve_model.py')

    write_requirements(output_dir)
    write_readme(output_dir, model_filename, include_server=include_server)
    save_json(output_dir / 'package_manifest.json', {
        'schema_version': 'ask2know_python_offline_package_v1',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'ask2know_version': __version__,
        'model_file': model_filename,
        'source_model': str(model_path),
        'model_schema_version': bundle.get('schema_version'),
        'task': bundle.get('task', {}),
        'class_count': len(bundle.get('classes') or []),
        'include_server': bool(include_server),
        'entrypoints': {
            'single_image': f'python scripts\\predict_model.py --model {model_filename} --image path\\to\\image.jpg',
            'folder': f'python scripts\\predict_folder.py --model {model_filename} --input path\\to\\images --output predictions.csv',
            'server': None if not include_server else f'python scripts\\serve_model.py --model {model_filename} --port 8000',
        },
    })
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Build a Python offline deployment package from an Ask2Know .a2kmodel.json bundle.')
    parser.add_argument('--model', required=True, help='Path to exported .a2kmodel.json')
    parser.add_argument('--output-dir', required=True, help='New package directory to create. Must not already exist.')
    parser.add_argument('--model-filename', help='Model filename inside the package. Defaults to the source filename.')
    parser.add_argument('--include-server', action='store_true', help='Also include scripts/serve_model.py for optional local HTTP serving.')
    args = parser.parse_args()

    output_dir = build_package(
        args.model,
        args.output_dir,
        include_server=args.include_server,
        model_filename=args.model_filename,
    )
    manifest = load_json(output_dir / 'package_manifest.json')
    print('Built offline package:', output_dir)
    print('Model:', output_dir / manifest['model_file'])
    print('Classes:', manifest.get('class_count'))
    print('Predict one image:')
    print(' ', manifest['entrypoints']['single_image'])
    print('Predict a folder:')
    print(' ', manifest['entrypoints']['folder'])
    if manifest['entrypoints'].get('server'):
        print('Optional server:')
        print(' ', manifest['entrypoints']['server'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
