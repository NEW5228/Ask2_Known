from pathlib import Path

from ask2know.features.feature_config import (
    DEFAULT_GROUP_WEIGHTS,
    PRESET_DEFAULT_GROUPS,
    USER_FEATURE_GROUPS,
    infer_feature_preset,
    resolve_feature_preset,
)
from ask2know.utils.io_utils import save_json

VERSION = '0.4.63.1'


def prompt_templates_for_preset(feature_preset):
    if feature_preset == 'car':
        return [
            'a photo of a {label} car',
            'a photo of a {label} vehicle',
            'the front grille and badge of a {label} car',
            'the logo or wordmark on a {label} vehicle',
        ]
    return [
        'a photo of a {label}',
        'a close-up photo of a {label}',
    ]


def crop_names_for_preset(feature_preset):
    crops = ['full', 'center', 'five_crop', 'object', 'head']
    if feature_preset == 'car':
        crops.extend(['car_front', 'grille', 'tail_lights', 'side', 'wheels'])
    return crops


def hierarchy_yaml_for_preset(feature_preset):
    if feature_preset != 'car':
        return '''  hierarchy:
    enable: false
    parser: auto
    max_candidate_classes: 12
    min_group_size: 1
    score_weight: 0.08
    max_score_margin: 0.060
    min_gap: 0.0
    level_weights:
      level_1: 0.40
      level_2: 0.35
      level_3: 0.25'''
    return '''  hierarchy:
    enable: false
    parser: car
    max_candidate_classes: 6
    min_group_size: 1
    score_weight: 0.03
    max_score_margin: 0.012
    min_gap: 0.003
    level_weights:
      brand: 0.35
      model: 0.45
      year: 0.20'''


def _write_task_config(path, task_name, dataset_dir, output_dir, project_root, classes, feature_preset, enabled_features):
    class_lines = '\n'.join([f'  - {c}' for c in classes])
    group_lines = '\n'.join([f'    {name}: {str(name in enabled_features).lower()}' for name in USER_FEATURE_GROUPS])
    weight_groups = list(enabled_features)
    if 'embedding' not in weight_groups:
        weight_groups.append('embedding')
    weight_lines = '\n'.join([
        f'    {name}: {DEFAULT_GROUP_WEIGHTS[name]:.2f}'
        for name in weight_groups
        if name in DEFAULT_GROUP_WEIGHTS
    ])
    prompt_lines = '\n'.join([
        f'      - "{template}"'
        for template in prompt_templates_for_preset(feature_preset)
    ])
    crop_lines = '\n'.join([f'      - {name}' for name in crop_names_for_preset(feature_preset)])
    hierarchy_lines = hierarchy_yaml_for_preset(feature_preset)
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
  score_weight: 0.05

deep_features:
  enable: true
  provider: open_clip
  model_name: ViT-B-32
  pretrained: laion2b_s34b_b79k
  device: auto
  feature_name: image_embedding
  cache: true
  fallback_to_opencv: false
  include_augmented: false
  multi_crop:
    enable: true
    crops:
{crop_lines}
    center_ratio: 0.86
    corner_ratio: 0.72

similarity:
  mode: hybrid
  knn:
    enable: true
    k: 3
    score_weight: 0.20
  sub_prototypes:
    enable: true
    max_centers: 3
    min_samples_per_center: 8
    score_weight: 0.06
    mode: conservative
    min_gain_over_prototype: 0.015
    min_top_gap: 0.0
    allow_rank_flip: true
    max_base_margin_for_flip: 0.010
    rank_flip_prototype_veto_margin: 0.003
  text_semantic:
    enable: true
    score_weight: 0.08
    prompt_templates:
{prompt_lines}
  pairwise_rerank:
    enable: true
    local_k: 5
    score_weight: 0.25
    max_score_margin: 0.018
    min_pair_similarity: 0.90
    min_local_gap: 0.008
  crop_rerank:
    enable: true
    max_candidate_classes: 3
    local_k: 5
    score_weight: 0.18
    max_score_margin: 0.018
    min_pair_similarity: 0.94
    min_local_gap: 0.006
    use_full_crop: false
    trigger_mode: margin_and_pair_similarity
  late_fusion:
    enable: true
    max_candidate_classes: 3
    weights:
      base_score: 1.0
      knn_score: 0.8
      text_semantic_score: 0.8
      crop_rerank_score: 0.4
{hierarchy_lines}
  robust_prototype:
    enable: true
    deep_only: true
    min_samples: 24
    trim_fraction: 0.08
    report_margin: 0.015
    top_outliers_per_class: 5
  concept_gate:
    enable: true
    min_top_gap: 0.035
    weak_score_weight: 0.00

diagnostics:
  low_margin_threshold: 0.015
  weak_signal_threshold: 0.005

validation:
  pass_accuracy_threshold: 0.85

learning:
  initial_weights:
{weight_lines}
  default_feature_weight: 0.08
  update_step: 0.07
  min_weight: 0.05
  max_weight: 0.95

confidence:
  auto_accept_threshold: 0.88
  ask_user_threshold: 0.03
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

unknown_import:
  auto_rename: true

unlabeled_import:
  auto_rename: false

augmentation:
  enable: true
  brightness: true
  rotation: true
  crop: true
  blur: false

future_modules:
  crawler_external_candidates: planned
  visual_concept_layer: planned
  deep_feature_adapter: enabled

teaching:
  allow_region_seed: false
  max_seed_annotations_per_class: 2
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def create_task_project(name, classes, output='.', feature_preset='auto', features=None):
    task_name = str(name).strip()
    if not task_name:
        raise ValueError('项目名不能为空。')
    clean_classes = [str(item).strip() for item in classes or [] if str(item).strip()]
    if not clean_classes:
        raise ValueError('至少需要一个类别。')

    resolved_preset = resolve_feature_preset(feature_preset, clean_classes)
    if feature_preset == 'auto':
        resolved_preset = infer_feature_preset(clean_classes)
    if features is None:
        enabled_features = list(PRESET_DEFAULT_GROUPS.get(resolved_preset, PRESET_DEFAULT_GROUPS['general']))
    else:
        enabled_features = [name for name in dict.fromkeys(features) if name in USER_FEATURE_GROUPS]
        if not enabled_features:
            raise ValueError('至少需要选择一个用户可见特征。')

    project_root = Path(output).expanduser().resolve() / task_name
    dataset_dir = project_root / 'datasets'
    output_dir = project_root / 'outputs'
    config_path = project_root / 'configs' / 'task_config.yaml'

    dirs = [
        project_root / 'configs',
        dataset_dir / 'train',
        dataset_dir / 'unknown',
        dataset_dir / 'unlabeled',
        output_dir,
        project_root / 'experience',
        project_root / 'logs',
        project_root / 'sample_pools' / 'candidate',
        project_root / 'sample_pools' / 'rejected',
        project_root / 'sample_pools' / 'unknown',
        project_root / 'metadata',
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

    for cls in clean_classes:
        (dataset_dir / 'train' / cls).mkdir(parents=True, exist_ok=True)

    objects = [
        {
            'object_id': f'C{idx:03d}',
            'name': cls,
            'display_name': cls,
            'description': '',
        }
        for idx, cls in enumerate(clean_classes, 1)
    ]
    save_json(dataset_dir / 'objects.json', {'objects': objects})
    save_json(dataset_dir / 'concepts.json', {'concepts': []})
    save_json(project_root / 'metadata' / 'project_meta.json', {
        'project_name': task_name,
        'created_by': f'a2k_v{VERSION}',
        'schema_version': VERSION,
        'classes': clean_classes,
    })
    save_json(project_root / 'metadata' / 'dataset_index.json', {
        'schema_version': VERSION,
        'classes': {cls: {'next_id': 1, 'count': 0} for cls in clean_classes},
    })
    _write_task_config(
        config_path,
        task_name,
        str(dataset_dir).replace('\\', '/'),
        str(output_dir).replace('\\', '/'),
        str(project_root).replace('\\', '/'),
        clean_classes,
        resolved_preset,
        enabled_features,
    )
    readme = project_root / 'README_task.md'
    readme.write_text(f'''# {task_name}

这个任务由 Ask2Know v{VERSION} 自动创建。

## 放图片

已确认训练样本放入：

```text
{dataset_dir / 'train'}
```

待学习未知样本放入：

```text
{dataset_dir / 'unknown'}
```

''', encoding='utf-8')

    return {
        'project_root': str(project_root),
        'dataset_dir': str(dataset_dir),
        'output_dir': str(output_dir),
        'config_path': str(config_path),
        'feature_preset': resolved_preset,
        'features': enabled_features,
        'classes': clean_classes,
    }
