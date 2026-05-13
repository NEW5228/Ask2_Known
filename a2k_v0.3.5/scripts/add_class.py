import argparse
import json
from pathlib import Path
import yaml


def load_json(path, default):
    path = Path(path)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Add a class to an existing Ask2Know project without losing old data.')
    parser.add_argument('--project', required=True, help='已有项目目录，例如 D:\\a2k_test\\fruit_test3')
    parser.add_argument('--class', dest='class_name', required=True, help='新类别英文名，例如 cherry')
    args = parser.parse_args()

    project = Path(args.project).expanduser().resolve()
    cls = args.class_name.strip()
    if not cls:
        raise SystemExit('类别名不能为空')

    dataset_dir = project / 'datasets'
    train_dir = dataset_dir / 'train'
    config_path = project / 'configs' / 'task_config.yaml'
    objects_path = dataset_dir / 'objects.json'
    concepts_path = dataset_dir / 'concepts.json'
    metadata_dir = project / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)

    (train_dir / cls).mkdir(parents=True, exist_ok=True)

    objects_data = load_json(objects_path, {'objects': []})
    objects = objects_data.setdefault('objects', [])
    names = [o.get('name') for o in objects]
    if cls not in names:
        objects.append({
            'object_id': f'C{len(objects) + 1:03d}',
            'name': cls,
            'display_name': cls,
            'description': 'added by a2k_v0.3.5 add_class'
        })
        save_json(objects_path, objects_data)

    if not concepts_path.exists():
        save_json(concepts_path, {'concepts': []})

    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        classes = cfg.setdefault('classes', [])
        if cls not in classes:
            classes.append(cls)
        cfg.setdefault('paths', {})['project_root'] = str(project).replace('\\', '/')
        cfg['paths']['dataset_dir'] = str(dataset_dir).replace('\\', '/')
        cfg['paths']['output_dir'] = str((project / 'outputs')).replace('\\', '/')
        cfg.setdefault('train_import', {})['auto_rename'] = True
        cfg.setdefault('unlabeled_import', {})['auto_rename'] = True
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    else:
        print('未找到 task_config.yaml。请先确认这是 a2k 项目目录。')

    meta = load_json(metadata_dir / 'project_meta.json', {})
    meta['last_used_by'] = 'a2k_v0.3.5'
    meta['schema_version'] = '0.3.5'
    meta['classes'] = sorted(set((meta.get('classes') or []) + [cls]))
    save_json(metadata_dir / 'project_meta.json', meta)

    print('已在旧项目中添加类别，不会删除旧数据。')
    print('项目:', project)
    print('新类别:', cls)
    print('请把该类别训练图片放入:', train_dir / cls)
    print('运行时使用:', config_path)


if __name__ == '__main__':
    main()
