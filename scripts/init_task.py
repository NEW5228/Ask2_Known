import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ask2know.features.feature_config import (
    DEFAULT_GROUP_WEIGHTS,
    USER_FEATURE_GROUPS,
    infer_feature_preset,
    resolve_feature_preset,
)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_yaml_like(path, task_name, dataset_dir, output_dir, project_root, classes, feature_preset, enabled_features):
    class_lines = '\n'.join([f'  - {c}' for c in classes])
    group_lines = '\n'.join([f'    {name}: {str(name in enabled_features).lower()}' for name in USER_FEATURE_GROUPS])
    weight_lines = '\n'.join([
        f'    {name}: {DEFAULT_GROUP_WEIGHTS[name]:.2f}'
        for name in USER_FEATURE_GROUPS
        if name in enabled_features
    ])
    text = f'''task:
  name: {task_name}
  type: image_object_recognition
  description: 用户自定义 Ask2Know 任务

paths:
  project_root: {project_root}
  dataset_dir: {dataset_dir}
  output_dir: {output_dir}

classes:
{class_lines}

features:
  preset: {feature_preset}
  groups:
{group_lines}
  system:
    quality: true

concepts:
  enable: true
  score_weight: 0.25

learning:
  initial_weights:
{weight_lines}
  default_feature_weight: 0.08
  update_step: 0.07
  min_weight: 0.05
  max_weight: 0.70

confidence:
  auto_accept_threshold: 0.88
  ask_user_threshold: 0.12
  global_uncertainty_spread: 0.08
  global_uncertainty_top_n: 5
  saturation_ratio_threshold: 0.65

question:
  max_questions_per_sample: 1
  enable_question_reward: true

sample_pool:
  enable: true
  require_confirm_before_learning: true
  move_unlabeled_after_decision: true

train_import:
  auto_rename: true

unlabeled_import:
  auto_rename: true

augmentation:
  enable: true
  brightness: true
  rotation: true
  crop: true
  blur: false

future_modules:
  crawler_external_candidates: planned
  visual_concept_layer: planned
  deep_feature_adapter: planned

teaching:
  allow_region_seed: false
  max_seed_annotations_per_class: 2
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Create a new Ask2Know task folder.')
    parser.add_argument('--name', required=True, help='任务名，例如 fruit_task 或 vehicle_brand_demo')
    parser.add_argument('--classes', nargs='+', required=True, help='类别名，例如 apple banana pear 或 bmw audi benz')
    parser.add_argument('--output', default='.', help='任务输出根目录，例如 D:\\a2k_test。默认当前目录')
    parser.add_argument('--feature-preset', choices=['auto', 'general', 'fruit'], default='auto', help='特征预设。auto 会根据类别名判断是否使用 fruit。')
    parser.add_argument('--features', nargs='+', choices=list(USER_FEATURE_GROUPS), default=list(USER_FEATURE_GROUPS), help='用户可选特征大类。quality 是系统质量检查，不在这里选择。')
    args = parser.parse_args()

    task_name = args.name
    feature_preset = resolve_feature_preset(args.feature_preset, args.classes)
    if args.feature_preset == 'auto':
        feature_preset = infer_feature_preset(args.classes)
    enabled_features = list(dict.fromkeys(args.features))
    project_root = Path(args.output).expanduser().resolve() / task_name
    dataset_dir = project_root / 'datasets'
    output_dir = project_root / 'outputs'
    config_path = project_root / 'configs' / 'task_config.yaml'

    dirs = [
        project_root / 'configs',
        dataset_dir / 'train',
        dataset_dir / 'unlabeled',
        output_dir,
        project_root / 'experience',
        project_root / 'logs',
        project_root / 'sample_pools' / 'candidate',
        project_root / 'sample_pools' / 'rejected',
        project_root / 'sample_pools' / 'unknown',
        project_root / 'metadata',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    for cls in args.classes:
        (dataset_dir / 'train' / cls).mkdir(parents=True, exist_ok=True)

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
    write_json(project_root / 'metadata' / 'project_meta.json', {
        'project_name': task_name,
        'created_by': 'a2k_v0.3.7',
        'schema_version': '0.3.7',
        'classes': args.classes
    })
    write_json(project_root / 'metadata' / 'dataset_index.json', {
        'schema_version': '0.3.7',
        'classes': {cls: {'next_id': 1, 'count': 0} for cls in args.classes}
    })
    write_yaml_like(
        config_path,
        task_name,
        str(dataset_dir).replace('\\', '/'),
        str(output_dir).replace('\\', '/'),
        str(project_root).replace('\\', '/'),
        args.classes,
        feature_preset,
        enabled_features,
    )

    readme = project_root / 'README_task.md'
    readme.write_text(f'''# {task_name}\n\n这个任务由 Ask2Know v0.3.7 自动创建。\n\n## 放图片\n\n已知样本放入：\n\n```text\n{dataset_dir / 'train'}\n```\n\n未知样本放入：\n\n```text\n{dataset_dir / 'unlabeled'}\n```\n\n## 运行\n\n在 Ask2Know 框架目录执行：\n\n```bat\npython run_demo.py --config {config_path}\n```\n\nv0.3.7 默认不弹图，避免 Windows 图片查看器占用文件。需要预览时：\n\n```bat\npython run_demo.py --config {config_path} --preview\n```\n''', encoding='utf-8')

    print('Ask2Know task created.')
    print('Project:', project_root)
    print('Dataset:', dataset_dir)
    print('Config :', config_path)
    print('Feature preset:', feature_preset)
    print('User features:', ', '.join(enabled_features))
    print('\nNext steps:')
    print('1. Put known samples into:')
    for cls in args.classes:
        print(f'   {dataset_dir / "train" / cls}')
    print('2. Put unknown samples into:')
    print(f'   {dataset_dir / "unlabeled"}')
    print('3. Run from the a2k framework folder:')
    print(f'   python run_demo.py --config {config_path}')


if __name__ == '__main__':
    main()
