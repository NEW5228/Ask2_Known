USER_FEATURE_GROUPS = ('color', 'shape', 'texture', 'surface', 'part', 'size', 'text', 'sign')
DEFAULT_USER_FEATURE_GROUPS = ('color', 'shape', 'texture', 'surface', 'size')
SYSTEM_FEATURES = ('quality',)
SCORING_ONLY_FEATURE_GROUPS = ('embedding',)

DEFAULT_DEEP_FEATURE_CONFIG = {
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
        'crops': ['full', 'center', 'five_crop', 'object', 'head'],
        'center_ratio': 0.86,
        'corner_ratio': 0.72,
    },
}

FRUIT_CLASS_NAMES = {
    'apple', 'banana', 'pear', 'grape', 'orange', 'peach', 'cherry',
    'strawberry', 'lemon', 'lime', 'mango', 'watermelon', 'kiwi',
    'pineapple', 'plum', 'apricot', 'blueberry', 'raspberry',
}

TRAFFIC_SIGN_CLASS_NAMES = {
    'stop', 'no_entry', 'yield', 'speed_limit', 'turn_left', 'turn_right',
    'straight', 'u_turn', 'no_turn', 'no_parking', 'prohibit', 'forbidden',
    'warning', 'arrow', 'left_arrow', 'right_arrow', 'traffic_sign',
}

PET_CLASS_NAMES = {
    'cat', 'dog', 'kitten', 'puppy', 'feline', 'canine', 'pet', 'pets',
    'cats', 'dogs', 'cat_dog', 'animal',
    '猫', '狗', '小猫', '小狗', '宠物', '动物',
}

CAR_BRAND_CLASS_NAMES = {
    'am_general', 'acura', 'alfa_romeo', 'aston_martin', 'audi', 'bmw',
    'bentley', 'bugatti', 'buick', 'cadillac', 'chevrolet', 'chrysler',
    'daewoo', 'dodge', 'eagle', 'fiat', 'ferrari', 'fisker', 'ford',
    'gmc', 'geo', 'honda', 'hummer', 'hyundai', 'infiniti', 'isuzu',
    'jaguar', 'jeep', 'lamborghini', 'land_rover', 'lincoln', 'mini',
    'mazda', 'mclaren', 'mercedes_benz', 'mitsubishi', 'nissan',
    'plymouth', 'porsche', 'ram', 'rolls_royce', 'scion', 'spyker',
    'suzuki', 'tesla', 'toyota', 'volkswagen', 'volvo', 'smart',
}

PRESET_FEATURES = {
    'general': {
        'color': ['color'],
        'shape': ['contour'],
        'texture': ['texture'],
        'surface': ['surface_mark'],
        'part': ['fruit_part'],
        'size': ['size'],
        'text': ['text_mark'],
        'sign': ['sign_symbol'],
    },
    'fruit': {
        'color': ['color', 'fruit_color'],
        'shape': ['contour', 'fruit_shape', 'fruit_structure'],
        'texture': ['texture', 'fruit_texture'],
        'surface': ['surface_mark'],
        'part': ['fruit_part'],
        'size': ['size'],
        'text': ['text_mark'],
        'sign': ['sign_symbol'],
    },
    'pet': {
        'color': ['color'],
        'shape': ['contour', 'animal_shape'],
        'texture': ['texture', 'fur_texture'],
        'surface': ['surface_mark'],
        'part': ['animal_face'],
        'size': ['size'],
        'text': ['text_mark'],
        'sign': ['sign_symbol'],
    },
    'car': {
        'color': ['color'],
        'shape': ['contour', 'car_shape'],
        'texture': ['texture'],
        'surface': ['surface_mark'],
        'part': ['car_part'],
        'size': ['size'],
        'text': ['text_mark'],
        'sign': ['sign_symbol'],
    },
    'traffic_sign': {
        'color': ['color'],
        'shape': ['contour'],
        'texture': ['texture'],
        'surface': ['surface_mark'],
        'part': ['fruit_part'],
        'size': ['size'],
        'text': ['text_mark'],
        'sign': ['sign_symbol'],
    },
}

DEFAULT_GROUP_WEIGHTS = {
    'color': 0.03,
    'shape': 0.03,
    'texture': 0.03,
    'surface': 0.03,
    'part': 0.03,
    'size': 0.02,
    'text': 0.03,
    'sign': 0.03,
    'embedding': 5.00,
}

PRESET_DEFAULT_GROUPS = {
    'general': ('color', 'shape', 'texture', 'surface', 'size'),
    'fruit': ('color', 'shape', 'texture', 'surface', 'part', 'size'),
    'pet': ('color', 'shape', 'texture', 'surface', 'part', 'size'),
    'car': ('color', 'shape', 'texture', 'surface', 'part', 'size', 'text', 'sign'),
    'traffic_sign': ('color', 'shape', 'text', 'sign'),
}


def infer_feature_preset(classes):
    names = {str(name).strip().lower() for name in (classes or [])}
    if names & TRAFFIC_SIGN_CLASS_NAMES:
        return 'traffic_sign'
    if names & CAR_BRAND_CLASS_NAMES:
        return 'car'
    if names & PET_CLASS_NAMES:
        return 'pet'
    return 'fruit' if names & FRUIT_CLASS_NAMES else 'general'


def resolve_feature_preset(preset, classes=None):
    preset = str(preset or 'auto').strip().lower()
    if preset == 'auto':
        return infer_feature_preset(classes)
    if preset not in PRESET_FEATURES:
        raise ValueError(f'Unsupported feature preset: {preset}. Use auto, general, fruit, pet, car, or traffic_sign.')
    return preset


def resolve_deep_feature_config(cfg):
    raw = cfg.get('deep_features')
    merged = dict(DEFAULT_DEEP_FEATURE_CONFIG)
    if isinstance(raw, dict):
        merged.update(raw)
    if not bool(merged.get('enable', True)):
        raise ValueError('Ask2Know v0.4.63.1 requires deep_features.enable: true with provider: open_clip.')
    provider = str(merged.get('provider', 'open_clip')).strip().lower()
    if provider not in ('clip', 'open_clip'):
        raise ValueError('Ask2Know v0.4.63.1 requires deep_features.provider: open_clip.')
    merged['provider'] = provider
    merged['fallback_to_opencv'] = False
    return merged


def parse_feature_config(cfg, classes=None):
    raw = cfg.get('features')
    if not isinstance(raw, dict) or 'groups' not in raw:
        raise ValueError(
            'Unsupported feature config. Use the new format: '
            'features: {preset: fruit, groups: {color: true, shape: true, texture: true, surface: true, part: true, size: true, text: false, sign: false}, '
            'system: {quality: true}}.'
        )

    preset = resolve_feature_preset(raw.get('preset', 'auto'), classes)
    groups_raw = raw.get('groups') or {}
    system_raw = raw.get('system') or {}
    default_groups = set(PRESET_DEFAULT_GROUPS.get(preset, DEFAULT_USER_FEATURE_GROUPS))

    groups = {
        name: bool(groups_raw.get(name, name in default_groups))
        for name in USER_FEATURE_GROUPS
    }
    system = {name: bool(system_raw.get(name, True)) for name in SYSTEM_FEATURES}

    group_features = {}
    scoring_features = []
    for group in USER_FEATURE_GROUPS:
        if not groups[group]:
            continue
        internal = list(PRESET_FEATURES[preset].get(group, []))
        group_features[group] = internal
        for name in internal:
            if name not in scoring_features:
                scoring_features.append(name)

    system_features = [name for name in SYSTEM_FEATURES if system.get(name, True)]
    deep_raw = resolve_deep_feature_config(cfg)
    deep_enabled = bool(deep_raw.get('enable', False))
    if deep_enabled:
        deep_feature_name = str(deep_raw.get('feature_name', 'image_embedding'))
        group_features['embedding'] = [deep_feature_name]
        if deep_feature_name not in scoring_features:
            scoring_features.append(deep_feature_name)

    all_features = list(scoring_features)
    for name in system_features:
        if name not in all_features:
            all_features.append(name)

    feature_to_group = {}
    for group, names in group_features.items():
        for name in names:
            feature_to_group[name] = group

    return {
        'preset': preset,
        'groups': groups,
        'system': system,
        'display_features': [name for name in USER_FEATURE_GROUPS if groups.get(name)],
        'scoring_features': scoring_features,
        'system_features': system_features,
        'all_features': all_features,
        'group_features': group_features,
        'feature_to_group': feature_to_group,
        'scoring_only_groups': [name for name in SCORING_ONLY_FEATURE_GROUPS if name in group_features],
    }


def expand_feature_keys(keys, feature_spec):
    expanded = []
    group_features = feature_spec.get('group_features', {})
    scoring = set(feature_spec.get('scoring_features', []))
    system = set(feature_spec.get('system_features', []))
    for key in keys or []:
        if key in group_features:
            candidates = group_features[key]
        elif key in scoring or key in system:
            candidates = [key]
        else:
            candidates = []
        for name in candidates:
            if name not in expanded:
                expanded.append(name)
    return expanded


def expand_feature_adjustments(keys, feature_spec):
    adjustments = {}
    group_features = feature_spec.get('group_features', {})
    scoring = set(feature_spec.get('scoring_features', []))
    system = set(feature_spec.get('system_features', []))
    for key in keys or []:
        if key in group_features:
            candidates = group_features[key]
            factor = 1.0 / max(1, len(candidates))
        elif key in scoring or key in system:
            candidates = [key]
            factor = 1.0
        else:
            candidates = []
            factor = 0.0
        for name in candidates:
            adjustments[name] = adjustments.get(name, 0.0) + factor
    return adjustments


def initial_feature_weights(cfg, feature_spec):
    learning = cfg.get('learning', {})
    configured = dict(learning.get('initial_weights', {}))
    default_weight = float(learning.get('default_feature_weight', 0.08))
    weights = {}
    for group, names in feature_spec.get('group_features', {}).items():
        group_weight = float(configured.get(group, DEFAULT_GROUP_WEIGHTS.get(group, default_weight)))
        split_weight = group_weight / max(1, len(names))
        for name in names:
            weights[name] = split_weight
    return weights


def summarize_group_weights(weights, feature_spec):
    summary = {}
    for group, names in feature_spec.get('group_features', {}).items():
        vals = [float(weights[name]) for name in names if name in weights]
        if vals:
            summary[group] = sum(vals)
    return summary
