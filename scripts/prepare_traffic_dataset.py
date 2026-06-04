import argparse
import csv
import random
import sys
import zipfile
from collections import defaultdict
from io import TextIOWrapper
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ask2know.utils.io_utils import ensure_dir, save_json
from ask2know.utils.io_utils import load_yaml


GTSRB_CLASS_NAMES = {
    0: 'speed_limit_20',
    1: 'speed_limit_30',
    2: 'speed_limit_50',
    3: 'speed_limit_60',
    4: 'speed_limit_70',
    5: 'speed_limit_80',
    6: 'end_speed_limit_80',
    7: 'speed_limit_100',
    8: 'speed_limit_120',
    9: 'no_passing',
    10: 'no_passing_trucks',
    11: 'right_of_way_next_intersection',
    12: 'priority_road',
    13: 'yield',
    14: 'stop',
    15: 'no_vehicles',
    16: 'vehicles_over_3_5_tons_prohibited',
    17: 'no_entry',
    18: 'general_caution',
    19: 'dangerous_curve_left',
    20: 'dangerous_curve_right',
    21: 'double_curve',
    22: 'bumpy_road',
    23: 'slippery_road',
    24: 'road_narrows_right',
    25: 'road_work',
    26: 'traffic_signals',
    27: 'pedestrians',
    28: 'children_crossing',
    29: 'bicycles_crossing',
    30: 'beware_ice_snow',
    31: 'wild_animals_crossing',
    32: 'end_all_speed_passing_limits',
    33: 'turn_right_ahead',
    34: 'turn_left_ahead',
    35: 'ahead_only',
    36: 'go_straight_or_right',
    37: 'go_straight_or_left',
    38: 'keep_right',
    39: 'keep_left',
    40: 'roundabout_mandatory',
    41: 'end_no_passing',
    42: 'end_no_passing_trucks',
}

GTSRB_SYMBOL_VALUES = {
    0: ('number', 'number_20'),
    1: ('number', 'number_30'),
    2: ('number', 'number_50'),
    3: ('number', 'number_60'),
    4: ('number', 'number_70'),
    5: ('number', 'number_80'),
    6: ('number_end', 'number_80_end'),
    7: ('number', 'number_100'),
    8: ('number', 'number_120'),
    9: ('vehicle_overtaking', 'no_passing_cars'),
    10: ('vehicle_overtaking', 'no_passing_trucks'),
    11: ('intersection_priority', 'priority_next_intersection'),
    12: ('priority_symbol', 'priority_road_diamond'),
    13: ('priority_symbol', 'yield_triangle'),
    14: ('priority_symbol', 'stop_octagon'),
    15: ('vehicle_prohibition', 'no_vehicles_all'),
    16: ('vehicle_prohibition', 'no_heavy_trucks'),
    17: ('entry_symbol', 'no_entry_bar'),
    18: ('warning_symbol', 'general_caution_mark'),
    19: ('curve_symbol', 'curve_left'),
    20: ('curve_symbol', 'curve_right'),
    21: ('curve_symbol', 'double_curve'),
    22: ('road_surface', 'bumpy_road_symbol'),
    23: ('road_surface', 'slippery_road_symbol'),
    24: ('road_shape', 'road_narrows_right_symbol'),
    25: ('work_symbol', 'road_work_symbol'),
    26: ('traffic_control', 'traffic_lights_symbol'),
    27: ('person_symbol', 'pedestrian_symbol'),
    28: ('person_symbol', 'children_symbol'),
    29: ('vehicle_symbol', 'bicycle_symbol'),
    30: ('weather_symbol', 'ice_snow_symbol'),
    31: ('animal_symbol', 'wild_animals_symbol'),
    32: ('end_symbol', 'end_all_limits'),
    33: ('arrow', 'arrow_right'),
    34: ('arrow', 'arrow_left'),
    35: ('arrow', 'arrow_straight'),
    36: ('arrow', 'arrow_straight_or_right'),
    37: ('arrow', 'arrow_straight_or_left'),
    38: ('arrow', 'arrow_keep_right'),
    39: ('arrow', 'arrow_keep_left'),
    40: ('arrow', 'roundabout_arrow'),
    41: ('end_symbol', 'end_no_passing_cars'),
    42: ('end_symbol', 'end_no_passing_trucks'),
}


def _family_for_class(class_id):
    if class_id in {0, 1, 2, 3, 4, 5, 7, 8}:
        return 'speed_limit'
    if class_id == 6:
        return 'end_speed_limit'
    if class_id in {9, 10, 15, 16, 17}:
        return 'prohibition'
    if class_id in {11, 12, 13, 14}:
        return 'priority'
    if 18 <= class_id <= 31:
        return 'warning'
    if class_id in {32, 41, 42}:
        return 'end_restriction'
    if 33 <= class_id <= 40:
        return 'mandatory_direction'
    return 'other_traffic_sign'


def _shape_for_class(class_id):
    if class_id == 12:
        return 'diamond'
    if class_id == 13:
        return 'inverted_triangle'
    if class_id == 14:
        return 'octagon'
    if class_id in set(range(18, 32)) | {11}:
        return 'triangle'
    return 'circle'


def _color_for_class(class_id):
    if class_id in set(range(33, 41)):
        return 'blue_white'
    if class_id in {12}:
        return 'yellow_white'
    if class_id in {32, 41, 42, 6}:
        return 'gray_white'
    if class_id in set(range(18, 32)) | {11, 13}:
        return 'red_white_warning'
    if class_id == 14:
        return 'red_stop'
    return 'red_white'


def _safe_token(value):
    return str(value).strip().replace('.', '_').replace('-', '_').replace(' ', '_')


def _read_csv_from_zip(zf, name):
    with zf.open(name) as handle:
        reader = csv.DictReader(TextIOWrapper(handle, encoding='utf-8', newline=''))
        return [dict(row) for row in reader]


def _group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row['ClassId'])].append(row)
    return grouped


def _write_zip_member(zf, member_name, output_path):
    if output_path.exists():
        return False
    ensure_dir(output_path.parent)
    data = zf.read(member_name)
    output_path.write_bytes(data)
    return True


def _choose_rows(rows, count, rng):
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:min(len(rows), int(count))]


def _taxonomy_config(meta_rows, class_ids):
    meta_by_class = {int(row['ClassId']): row for row in meta_rows}
    label_paths = {}
    for class_id in class_ids:
        label = GTSRB_CLASS_NAMES[class_id]
        meta = meta_by_class.get(class_id, {})
        family = _family_for_class(class_id)
        shape = 'shape_' + _safe_token(meta.get('ShapeId', 'unknown'))
        color = 'color_' + _safe_token(meta.get('ColorId', 'unknown'))
        sign = 'sign_' + _safe_token(meta.get('SignId', 'unknown'))
        label_paths[label] = ['traffic_sign', family, shape, color, sign, label]
    return {
        'enable': True,
        'root': 'traffic_sign',
        'levels': ['root', 'family', 'shape', 'color', 'sign_id', 'leaf'],
        'label_paths': label_paths,
        'level_weights': {
            'root': 0.15,
            'family': 0.30,
            'shape': 0.15,
            'color': 0.15,
            'sign_id': 0.15,
            'leaf': 0.10,
        },
        'top_k_paths': 5,
        'score_weight': 0.08,
        'min_gap': 0.0,
        'max_score_margin': 0.06,
        'apply_to_score': True,
    }


def _semantic_taxonomy_config(class_ids):
    label_paths = {}
    for class_id in class_ids:
        label = GTSRB_CLASS_NAMES[class_id]
        family = _family_for_class(class_id)
        shape = _shape_for_class(class_id)
        color = _color_for_class(class_id)
        symbol_type, symbol_value = GTSRB_SYMBOL_VALUES.get(class_id, ('symbol', label + '_symbol'))
        label_paths[label] = [
            'traffic_sign',
            family,
            shape,
            color,
            symbol_type,
            symbol_value,
            label,
        ]
    return {
        'enable': True,
        'root': 'traffic_sign',
        'levels': ['root', 'family', 'shape', 'color', 'symbol_type', 'symbol_value', 'leaf'],
        'label_paths': label_paths,
        'level_weights': {
            'root': 0.08,
            'family': 0.18,
            'shape': 0.08,
            'color': 0.08,
            'symbol_type': 0.18,
            'symbol_value': 0.25,
            'leaf': 0.15,
        },
        'top_k_paths': 5,
        'score_weight': 0.16,
        'min_gap': 0.0,
        'max_score_margin': 0.20,
        'apply_to_score': True,
    }


def _load_taxonomy_override(path):
    data = load_yaml(path)
    if data.get('taxonomy'):
        return data['taxonomy']
    if data.get('similarity', {}).get('taxonomy'):
        return data['similarity']['taxonomy']
    return data


def _yaml_scalar(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return str(value)
    return '"' + str(value).replace('\\', '/').replace('"', '\\"') + '"'


def _write_yaml(path, data, indent=0):
    lines = []

    def emit(value, level):
        prefix = ' ' * level
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    lines.append(f'{prefix}{key}:')
                    emit(item, level + 2)
                else:
                    lines.append(f'{prefix}{key}: {_yaml_scalar(item)}')
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f'{prefix}-')
                    emit(item, level + 2)
                else:
                    lines.append(f'{prefix}- {_yaml_scalar(item)}')

    emit(data, indent)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _task_config(
    project_root,
    labels,
    taxonomy,
    include_reference_icons=False,
    recognition_mode='multilayer',
    enable_fine_grained=False,
    enable_field_clip=False,
    enable_field_shape=False,
    enable_local_leaf=False,
):
    dataset_dir = project_root / 'datasets'
    output_dir = project_root / 'outputs'
    recognition_mode = 'flat' if str(recognition_mode).strip().lower() == 'flat' else 'multilayer'
    taxonomy = dict(taxonomy or {})
    if recognition_mode == 'flat':
        taxonomy['enable'] = False
    return {
        'task': {
            'name': project_root.name,
            'type': 'image_object_recognition',
            'description': 'GTSRB traffic sign multilayer recognition task',
        },
        'paths': {
            'project_root': str(project_root),
            'dataset_dir': str(dataset_dir),
            'output_dir': str(output_dir),
        },
        'classes': labels,
        'features': {
            'preset': 'traffic_sign',
            'groups': {
                'color': True,
                'shape': True,
                'texture': False,
                'surface': False,
                'part': False,
                'size': False,
                'text': True,
                'sign': True,
            },
            'system': {'quality': True},
        },
        'concepts': {'enable': True, 'score_weight': 0.05},
        'deep_features': {
            'enable': True,
            'provider': 'open_clip',
            'model_name': 'ViT-B-32',
            'pretrained': 'laion2b_s34b_b79k',
            'device': 'auto',
            'feature_name': 'image_embedding',
            'cache': True,
            'fallback_to_opencv': False,
            'include_augmented': False,
            'multi_crop': {
                'enable': True,
                'crops': ['full', 'center', 'five_crop', 'object', 'traffic_inner', 'traffic_symbol'],
                'center_ratio': 0.86,
                'corner_ratio': 0.72,
            },
        },
        'similarity': {
            'mode': 'hybrid',
            'recognition_mode': recognition_mode,
            'knn': {'enable': True, 'k': 3, 'score_weight': 0.20},
            'sub_prototypes': {
                'enable': True,
                'max_centers': 3,
                'min_samples_per_center': 8,
                'score_weight': 0.06,
                'mode': 'conservative',
                'min_gain_over_prototype': 0.015,
                'min_top_gap': 0.0,
                'allow_rank_flip': True,
                'max_base_margin_for_flip': 0.010,
                'rank_flip_prototype_veto_margin': 0.003,
            },
            'text_semantic': {
                'enable': True,
                'score_weight': 0.08,
                'prompt_templates': [
                    'a photo of a {label} traffic sign',
                    'a close-up road sign showing {label}',
                ],
            },
            'pairwise_rerank': {
                'enable': True,
                'local_k': 5,
                'score_weight': 0.25,
                'max_score_margin': 0.040,
                'min_pair_similarity': 0.88,
                'min_local_gap': 0.004,
            },
            'crop_rerank': {
                'enable': True,
                'max_candidate_classes': 5,
                'local_k': 7,
                'score_weight': 0.24,
                'max_score_margin': 0.050,
                'min_pair_similarity': 0.88,
                'min_local_gap': 0.004,
                'use_full_crop': False,
                'trigger_mode': 'margin_and_pair_similarity',
            },
            'late_fusion': {
                'enable': True,
                'max_candidate_classes': 5,
                'weights': {
                    'base_score': 1.0,
                    'knn_score': 0.8,
                    'text_semantic_score': 0.8,
                    'pairwise_score': 0.7,
                    'crop_rerank_score': 0.6,
                    'taxonomy_score': 0.7,
                },
            },
            'hierarchy': {'enable': False},
            'taxonomy': taxonomy,
            'reference_icon_rerank': {
                'enable': recognition_mode != 'flat' and bool(include_reference_icons),
                'score_weight': 0.14,
                'max_score_margin': 0.18,
            },
            'fine_grained_rerank': {
                'enable': recognition_mode != 'flat' and bool(enable_fine_grained),
                'score_weight': 0.18,
                'max_candidate_classes': 5,
                'max_score_margin': 0.20,
                'group_level': 'symbol_type',
                'min_group_size': 2,
            },
            'field_clip_rerank': {
                'enable': recognition_mode != 'flat' and bool(enable_field_clip),
                'score_weight': 0.12,
                'max_candidate_classes': 5,
                'max_score_margin': 0.08,
                'group_level': 'symbol_type',
                'min_group_size': 2,
                'crop_ids': ['traffic_inner', 'traffic_symbol'],
            },
            'field_shape_rerank': {
                'enable': recognition_mode != 'flat' and bool(enable_field_shape),
                'score_weight': 0.10,
                'max_candidate_classes': 5,
                'max_score_margin': 0.08,
                'group_level': 'symbol_type',
                'min_group_size': 2,
            },
            'local_leaf_rerank': {
                'enable': recognition_mode != 'flat' and bool(enable_local_leaf),
                'score_weight': 0.26,
                'parent_level': 'symbol_type',
                'min_group_size': 2,
                'min_parent_margin': 0.0,
                'max_parent_score_margin': 0.20,
                'min_local_gap': 0.0,
                'use_crop': True,
                'use_field_clip': True,
                'crop_ids': ['traffic_inner', 'traffic_symbol'],
                'component_weights': {
                    'prototype_score': 0.12,
                    'knn_score': 0.34,
                    'field_crop_score': 0.38,
                    'field_clip_score': 0.16,
                },
            },
            'robust_prototype': {
                'enable': True,
                'deep_only': True,
                'min_samples': 24,
                'trim_fraction': 0.08,
                'report_margin': 0.015,
                'top_outliers_per_class': 5,
            },
            'concept_gate': {'enable': True, 'min_top_gap': 0.035, 'weak_score_weight': 0.0},
        },
        'diagnostics': {'low_margin_threshold': 0.015, 'weak_signal_threshold': 0.005},
        'validation': {'pass_accuracy_threshold': 0.85},
        'learning': {
            'initial_weights': {
                'color': 0.08,
                'shape': 0.08,
                'text': 0.08,
                'sign': 0.08,
                'embedding': 5.00,
            },
            'default_feature_weight': 0.08,
            'update_step': 0.07,
            'min_weight': 0.05,
            'max_weight': 0.95,
        },
        'question': {
            'max_questions_per_sample': 2,
            'enable_taxonomy_ask': True,
            'ask_candidate_top_k': 10,
            'max_taxonomy_options': 8,
            'enable_question_reward': True,
        },
        'sample_pool': {
            'enable': True,
            'require_confirm_before_learning': True,
            'move_unlabeled_after_decision': True,
        },
        'train_import': {'auto_rename': False},
        'unknown_import': {'auto_rename': False},
        'augmentation': {'enable': True, 'brightness': True, 'rotation': True, 'crop': True, 'blur': False},
    }


def prepare(args):
    project_root = Path(args.output).expanduser().resolve() / args.name
    if project_root.exists() and not args.reuse:
        raise SystemExit(f'Project already exists: {project_root}. Use --reuse to add missing files without deleting anything.')

    rng = random.Random(args.seed)
    with zipfile.ZipFile(args.zip) as zf:
        train_rows = _group_rows(_read_csv_from_zip(zf, 'Train.csv'))
        test_rows = _group_rows(_read_csv_from_zip(zf, 'Test.csv'))
        meta_rows = _read_csv_from_zip(zf, 'Meta.csv')
        class_ids = sorted(set(train_rows.keys()) & set(test_rows.keys()) & set(GTSRB_CLASS_NAMES.keys()))
        if args.limit_classes:
            class_ids = class_ids[:int(args.limit_classes)]
        labels = [GTSRB_CLASS_NAMES[class_id] for class_id in class_ids]
        if args.taxonomy_file:
            taxonomy = _load_taxonomy_override(args.taxonomy_file)
        elif args.taxonomy_mode == 'semantic':
            taxonomy = _semantic_taxonomy_config(class_ids)
        else:
            taxonomy = _taxonomy_config(meta_rows, class_ids)

        dataset_dir = project_root / 'datasets'
        train_dir = dataset_dir / 'train'
        eval_dir = dataset_dir / 'unlabeled'
        config_dir = project_root / 'configs'
        metadata_dir = project_root / 'metadata'
        output_dir = project_root / 'outputs'
        for directory in [train_dir, eval_dir, config_dir, metadata_dir, output_dir]:
            ensure_dir(directory)

        copied_train = 0
        copied_eval = 0
        manifest = {'train': [], 'eval': []}
        for class_id in class_ids:
            label = GTSRB_CLASS_NAMES[class_id]
            if args.include_meta_icons:
                meta_src = f'Meta/{class_id}.png'
                if meta_src in zf.namelist():
                    dst = train_dir / label / f'meta_{class_id}.png'
                    copied = _write_zip_member(zf, meta_src, dst)
                    copied_train += 1 if copied else 0
                    manifest['train'].append({'class_id': class_id, 'label': label, 'source': meta_src, 'path': str(dst), 'role': 'meta_icon'})
            for row in _choose_rows(train_rows[class_id], args.train_per_class, rng):
                src = row['Path']
                dst = train_dir / label / Path(src).name
                copied = _write_zip_member(zf, src, dst)
                copied_train += 1 if copied else 0
                manifest['train'].append({'class_id': class_id, 'label': label, 'source': src, 'path': str(dst)})
            for row in _choose_rows(test_rows[class_id], args.eval_per_class, rng):
                src = row['Path']
                dst = eval_dir / label / Path(src).name
                copied = _write_zip_member(zf, src, dst)
                copied_eval += 1 if copied else 0
                manifest['eval'].append({'class_id': class_id, 'label': label, 'source': src, 'path': str(dst)})

    objects = [
        {
            'object_id': f'C{idx + 1:03d}',
            'name': label,
            'display_name': label.replace('_', ' '),
            'description': 'GTSRB traffic sign class',
        }
        for idx, label in enumerate(labels)
    ]
    save_json(dataset_dir / 'objects.json', {'objects': objects})
    save_json(dataset_dir / 'concepts.json', {'concepts': []})
    save_json(metadata_dir / 'traffic_split_manifest.json', manifest)
    save_json(metadata_dir / 'project_meta.json', {'project_name': args.name, 'classes': labels})
    save_json(metadata_dir / 'taxonomy.json', taxonomy)
    _write_yaml(
        config_dir / 'task_config.yaml',
        _task_config(
            project_root,
            labels,
            taxonomy,
            include_reference_icons=args.include_meta_icons,
            recognition_mode=args.recognition_mode,
            enable_fine_grained=args.enable_fine_grained,
            enable_field_clip=args.enable_field_clip,
            enable_field_shape=args.enable_field_shape,
            enable_local_leaf=args.enable_local_leaf,
        ),
    )
    return project_root, copied_train, copied_eval, len(labels)


def main():
    parser = argparse.ArgumentParser(description='Prepare a GTSRB traffic sign project for Ask2Know multilayer recognition.')
    parser.add_argument('--zip', default='data/traffic.zip', help='Path to traffic.zip.')
    parser.add_argument('--output', default='data', help='Directory where the task project will be created.')
    parser.add_argument('--name', default='traffic_multilayer', help='Task project name.')
    parser.add_argument('--train-per-class', type=int, default=20, help='Training images copied per class.')
    parser.add_argument('--eval-per-class', type=int, default=20, help='Evaluation images copied per class.')
    parser.add_argument('--limit-classes', type=int, default=0, help='Optional first-N class limit for smoke tests.')
    parser.add_argument('--taxonomy-mode', choices=['semantic', 'meta'], default='semantic', help='semantic uses editable GTSRB meaning paths; meta uses ShapeId/ColorId/SignId from Meta.csv.')
    parser.add_argument('--taxonomy-file', default='', help='Optional YAML file containing taxonomy or similarity.taxonomy.')
    parser.add_argument('--include-meta-icons', action='store_true', help='Add Meta/<ClassId>.png standard icons as clean reference training samples.')
    parser.add_argument('--recognition-mode', choices=['flat', 'multilayer'], default='multilayer', help='flat keeps the old single-layer classifier; multilayer enables taxonomy/path output.')
    parser.add_argument('--enable-fine-grained', action='store_true', help='Experimental: enable CLIP sibling-group fine-grained rerank inside top-k candidates.')
    parser.add_argument('--enable-field-clip', action='store_true', help='Experimental: enable crop-level CLIP rerank for digits/arrows/symbol fields inside sibling groups.')
    parser.add_argument('--enable-field-shape', action='store_true', help='Experimental: enable OpenCV field-shape rerank for digits/arrows/symbol fields inside sibling groups.')
    parser.add_argument('--enable-local-leaf', action='store_true', help='Experimental: once a taxonomy parent is selected, rerank all leaf classes under that parent.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reuse', action='store_true', help='Reuse an existing project and only add missing files.')
    args = parser.parse_args()
    project_root, copied_train, copied_eval, class_count = prepare(args)
    print('Project:', project_root)
    print('Classes:', class_count)
    print('Copied train:', copied_train)
    print('Copied eval:', copied_eval)
    print('Config:', project_root / 'configs' / 'task_config.yaml')


if __name__ == '__main__':
    main()
