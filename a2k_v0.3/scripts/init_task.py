import argparse
import json
from pathlib import Path


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_yaml_like(path, task_name, dataset_dir, output_dir):
    text = f'''task:
  name: {task_name}
  type: image_object_recognition
  description: 用户自定义 Ask2Know 任务

paths:
  dataset_dir: {dataset_dir}
  output_dir: {output_dir}

features:
  color: true
  size: true
  contour: true
  texture: true

learning:
  initial_weights:
    color: 0.25
    size: 0.15
    contour: 0.30
    texture: 0.30
  update_step: 0.07
  min_weight: 0.05
  max_weight: 0.70

confidence:
  auto_accept_threshold: 0.80
  ask_user_threshold: 0.12

question:
  max_questions_per_sample: 1
  enable_question_reward: true

teaching:
  allow_region_seed: false
  max_seed_annotations_per_class: 2
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Create a new Ask2Know task folder.')
    parser.add_argument('--name', required=True, help='任务名，例如 catdog_demo 或 vehicle_brand_demo')
    parser.add_argument('--classes', nargs='+', required=True, help='类别名，例如 cat dog 或 bmw audi benz')
    parser.add_argument('--dataset-root', default='datasets', help='数据集根目录，默认 datasets')
    parser.add_argument('--config-dir', default='configs', help='配置文件目录，默认 configs')
    parser.add_argument('--output-root', default='outputs', help='输出根目录，默认 outputs')
    args = parser.parse_args()

    task_name = args.name
    dataset_dir = Path(args.dataset_root) / task_name
    output_dir = Path(args.output_root) / task_name
    config_path = Path(args.config_dir) / f'{task_name}.yaml'

    for cls in args.classes:
        (dataset_dir / 'train' / cls).mkdir(parents=True, exist_ok=True)
    (dataset_dir / 'unlabeled').mkdir(parents=True, exist_ok=True)

    objects = []
    for idx, cls in enumerate(args.classes, 1):
        objects.append({
            'object_id': f'C{idx:03d}',
            'name': cls,
            'display_name': cls,
            'description': ''
        })

    write_json(dataset_dir / 'objects.json', {'objects': objects})
    write_json(dataset_dir / 'concepts.json', {'concepts': []})
    write_yaml_like(config_path, task_name, str(dataset_dir).replace('\\', '/'), str(output_dir).replace('\\', '/'))

    print('Ask2Know task created.')
    print('Dataset:', dataset_dir)
    print('Config :', config_path)
    print('\nNext steps:')
    print('1. Put known samples into:')
    for cls in args.classes:
        print(f'   {dataset_dir / "train" / cls}')
    print('2. Put unknown samples into:')
    print(f'   {dataset_dir / "unlabeled"}')
    print('3. Run:')
    print(f'   python run_demo.py --config {config_path}')


if __name__ == '__main__':
    main()
