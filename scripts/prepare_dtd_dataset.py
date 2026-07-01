import argparse
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ask2know.runtime.project import _write_task_config
from ask2know.utils.io_utils import ensure_dir, save_json

DTD_URL = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'
VERSION = '0.5.0'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _safe_label(name):
    return str(name).strip().lower().replace(' ', '_').replace('-', '_')


def _download(url, path):
    ensure_dir(path.parent)
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f'Downloading {url}')
    print(f'To: {path}')
    urllib.request.urlretrieve(url, path)
    return path


def _extract(archive_path, extract_root):
    marker = extract_root / 'dtd' / 'labels' / 'train1.txt'
    if marker.exists():
        return extract_root / 'dtd'
    ensure_dir(extract_root)
    print(f'Extracting: {archive_path}')
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(extract_root)
    return extract_root / 'dtd'


def _read_split(dtd_root, name):
    path = dtd_root / 'labels' / f'{name}1.txt'
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        rel = Path(line)
        label = _safe_label(rel.parts[0])
        src = dtd_root / 'images' / rel
        if src.suffix.lower() in IMAGE_EXTS:
            rows.append({'label': label, 'src': src})
    return rows


def _copy_rows(rows, dst_root, limit_per_class=None):
    counts = {}
    copied = []
    for row in rows:
        label = row['label']
        counts[label] = counts.get(label, 0)
        if limit_per_class is not None and counts[label] >= limit_per_class:
            continue
        counts[label] += 1
        dst_dir = dst_root / label
        ensure_dir(dst_dir)
        suffix = row['src'].suffix.lower()
        dst = dst_dir / f'{label}_{counts[label]:03d}{suffix}'
        if not dst.exists():
            shutil.copy2(row['src'], dst)
        copied.append({'label': label, 'path': str(dst)})
    return copied, counts


def main():
    parser = argparse.ArgumentParser(description='Prepare DTD as an Ask2Know train/unlabeled task.')
    parser.add_argument('--output-root', default='data', help='Parent directory for the prepared task.')
    parser.add_argument('--name', default='dtd_train80_eval40', help='Task folder name.')
    parser.add_argument('--archive', default='data/dtd-r1.0.1.tar.gz', help='Local archive path.')
    parser.add_argument('--source-root', default='data/source_dtd', help='Extraction directory.')
    parser.add_argument('--download', action='store_true', help='Download the official DTD archive if missing.')
    parser.add_argument('--train-per-class', type=int, default=None, help='Limit train+val images per class. Default: all 80.')
    parser.add_argument('--eval-per-class', type=int, default=None, help='Limit test images per class. Default: all 40.')
    args = parser.parse_args()

    archive_path = Path(args.archive)
    if args.download:
        _download(DTD_URL, archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f'DTD archive not found: {archive_path}. Use --download to fetch it.')
    dtd_root = _extract(archive_path, Path(args.source_root))

    train_rows = _read_split(dtd_root, 'train') + _read_split(dtd_root, 'val')
    eval_rows = _read_split(dtd_root, 'test')
    labels = sorted({row['label'] for row in train_rows + eval_rows})

    project_root = Path(args.output_root) / args.name
    dataset_dir = project_root / 'datasets'
    output_dir = project_root / 'outputs'
    config_dir = project_root / 'configs'
    metadata_dir = project_root / 'metadata'
    for path in (dataset_dir / 'train', dataset_dir / 'unlabeled', dataset_dir / 'unknown', output_dir, config_dir, metadata_dir):
        ensure_dir(path)

    copied_train, train_counts = _copy_rows(train_rows, dataset_dir / 'train', args.train_per_class)
    copied_eval, eval_counts = _copy_rows(eval_rows, dataset_dir / 'unlabeled', args.eval_per_class)

    objects = [
        {
            'object_id': f'C{idx:03d}',
            'name': label,
            'display_name': label.replace('_', ' '),
            'description': 'DTD texture class',
        }
        for idx, label in enumerate(labels, 1)
    ]
    save_json(dataset_dir / 'objects.json', {'objects': objects})
    save_json(dataset_dir / 'concepts.json', {'concepts': []})
    save_json(metadata_dir / 'dataset_index.json', {
        'schema_version': VERSION,
        'project': args.name,
        'source': 'DTD r1.0.1 split 1',
        'train_counts': train_counts,
        'eval_counts': eval_counts,
    })

    config_path = config_dir / 'task_config.yaml'
    enabled_features = ['color', 'shape', 'texture', 'surface', 'part']
    _write_task_config(
        config_path,
        args.name,
        str(dataset_dir).replace('\\', '/'),
        str(output_dir).replace('\\', '/'),
        str(project_root).replace('\\', '/'),
        labels,
        'texture',
        enabled_features,
    )

    readme = project_root / 'README_task.md'
    readme.write_text(
        f'# {args.name}\n\n'
        f'Source: DTD r1.0.1 split 1\n\n'
        f'Train samples: {len(copied_train)}\n\n'
        f'Evaluation samples: {len(copied_eval)}\n\n'
        f'Config: {config_path}\n',
        encoding='utf-8',
    )
    print('Project:', project_root)
    print('Config:', config_path)
    print('Classes:', len(labels))
    print('Train samples:', len(copied_train))
    print('Eval samples:', len(copied_eval))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
