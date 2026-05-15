import argparse
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ask2know.sample_pool.manager import SamplePoolManager
from ask2know.data.dataset_loader import DatasetLoader


def main():
    parser = argparse.ArgumentParser(description='Normalize train/unlabeled filenames in an existing a2k project.')
    parser.add_argument('--config', required=True, help='项目 task_config.yaml')
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    dataset_dir = cfg['paths']['dataset_dir']
    output_dir = cfg['paths']['output_dir']
    project_root = cfg.get('paths', {}).get('project_root')
    loader = DatasetLoader(dataset_dir)
    objects = loader.load_objects()
    labels = [o['name'] for o in objects]
    pool = SamplePoolManager(project_root=project_root, output_dir=output_dir, dataset_dir=dataset_dir, version='0.3.7.1')
    pool.ensure_for_classes(labels)
    changed_train = pool.normalize_train_images(labels)
    changed_unlabeled = pool.normalize_unlabeled()
    print('训练集重命名:', len(changed_train))
    print('未标注集重命名:', len(changed_unlabeled))
    print('metadata:', pool.metadata_dir)


if __name__ == '__main__':
    main()
