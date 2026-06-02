from datetime import datetime, timezone
from pathlib import Path

from ask2know.data.dataset_loader import DatasetLoader
from ask2know.features.feature_config import (
    expand_feature_adjustments,
    initial_feature_weights,
    parse_feature_config,
    resolve_deep_feature_config,
    summarize_group_weights,
)
from ask2know.inference.prototype_model import PrototypeModel
from ask2know.learning.weights import AdaptiveWeights
from ask2know.utils.io_utils import ensure_dir, load_json, load_yaml, save_json
from ask2know import __version__ as VERSION


def class_names(objects):
    return [item.get('name') for item in objects if item.get('name')]


def expand_concept_feature_hints(concepts, feature_spec):
    expanded = []
    for item in concepts or []:
        copied = dict(item)
        copied['important_features'] = expand_feature_adjustments(item.get('important_features', []), feature_spec)
        copied['weak_features'] = expand_feature_adjustments(item.get('weak_features', []), feature_spec)
        expanded.append(copied)
    return expanded


def build_deployment_bundle(config_path, output_path=None, include_sample_features=True):
    config_path = Path(config_path).expanduser().resolve()
    cfg = load_yaml(config_path)
    dataset_dir = cfg['paths']['dataset_dir']
    output_dir = Path(cfg['paths']['output_dir'])
    ensure_dir(output_dir)

    loader = DatasetLoader(dataset_dir)
    objects = loader.load_objects()
    labels = class_names(objects)
    train_samples = loader.load_train_samples()
    if not labels:
        raise RuntimeError('No classes found. Create train folders or objects.json first.')
    if not train_samples:
        raise RuntimeError(f'No train samples found in {Path(dataset_dir) / "train"}')

    concepts = loader.load_concepts()
    deep_feature_config = resolve_deep_feature_config(cfg)
    feature_spec = parse_feature_config(cfg, classes=labels or cfg.get('classes', []))
    adaptive_weights = AdaptiveWeights(
        initial_feature_weights(cfg, feature_spec),
        cfg['learning'].get('update_step', 0.07),
        cfg['learning'].get('min_weight', 0.05),
        cfg['learning'].get('max_weight', 0.70),
    )
    adaptive_weights.apply_concepts(expand_concept_feature_hints(concepts, feature_spec))
    weights = adaptive_weights.export()

    model = PrototypeModel(
        feature_spec['scoring_features'],
        augmentation_config=cfg.get('augmentation', {}),
        concept_config=cfg.get('concepts', {'enable': True, 'score_weight': 0.25}),
        system_feature_names=feature_spec['system_features'],
        feature_groups=feature_spec['group_features'],
        similarity_config=cfg.get('similarity', {}),
        deep_feature_config=deep_feature_config,
        deep_cache_dir=output_dir / '.cache' / 'deep_features',
    ).fit(train_samples)

    task_name = cfg.get('task', {}).get('name', 'ask2know_model')
    if output_path is None:
        output_path = output_dir / f'{task_name}.a2kmodel.json'
    output_path = Path(output_path).expanduser().resolve()

    bundle = {
        'schema_version': 'ask2know_deployment_bundle_v1',
        'framework': 'Ask2Know',
        'version': VERSION,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_config': str(config_path),
        'task': cfg.get('task', {}),
        'classes': objects,
        'feature_spec': feature_spec,
        'weights': weights,
        'visible_weights': summarize_group_weights(weights, feature_spec),
        'deep_feature_config': deep_feature_config,
        'model': model.export(include_sample_features=include_sample_features),
    }
    save_json(output_path, bundle)
    return output_path, bundle


def build_deployment_bundle_from_model_cache(
    config_path,
    model_cache_path,
    output_path=None,
    include_sample_features=True,
):
    config_path = Path(config_path).expanduser().resolve()
    model_cache_path = Path(model_cache_path).expanduser().resolve()
    cfg = load_yaml(config_path)
    dataset_dir = cfg['paths']['dataset_dir']
    output_dir = Path(cfg['paths']['output_dir'])
    ensure_dir(output_dir)

    loader = DatasetLoader(dataset_dir)
    objects = loader.load_objects()
    labels = class_names(objects)
    concepts = loader.load_concepts()
    deep_feature_config = resolve_deep_feature_config(cfg)
    feature_spec = parse_feature_config(cfg, classes=labels or cfg.get('classes', []))
    adaptive_weights = AdaptiveWeights(
        initial_feature_weights(cfg, feature_spec),
        cfg['learning'].get('update_step', 0.07),
        cfg['learning'].get('min_weight', 0.05),
        cfg['learning'].get('max_weight', 0.70),
    )
    adaptive_weights.apply_concepts(expand_concept_feature_hints(concepts, feature_spec))
    weights = adaptive_weights.export()

    cached = load_json(model_cache_path)
    model_data = cached.get('model') if isinstance(cached, dict) and 'model' in cached else cached
    model = PrototypeModel.from_export(
        model_data,
        deep_feature_config=deep_feature_config,
        deep_cache_dir=output_dir / '.cache' / 'deep_features',
    )

    task_name = cfg.get('task', {}).get('name', 'ask2know_model')
    if output_path is None:
        output_path = output_dir / f'{task_name}.a2kmodel.json'
    output_path = Path(output_path).expanduser().resolve()

    bundle = {
        'schema_version': 'ask2know_deployment_bundle_v1',
        'framework': 'Ask2Know',
        'version': VERSION,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_config': str(config_path),
        'source_model_cache': str(model_cache_path),
        'task': cfg.get('task', {}),
        'classes': objects,
        'feature_spec': feature_spec,
        'weights': weights,
        'visible_weights': summarize_group_weights(weights, feature_spec),
        'deep_feature_config': deep_feature_config,
        'model': model.export(include_sample_features=include_sample_features),
    }
    save_json(output_path, bundle)
    return output_path, bundle


def load_deployment_bundle(model_path, cache_dir=None):
    model_path = Path(model_path).expanduser().resolve()
    bundle = load_json(model_path)
    model_data = bundle.get('model') or bundle
    deep_feature_config = bundle.get('deep_feature_config') or model_data.get('deep_feature_config') or {}
    if cache_dir is None:
        cache_dir = model_path.parent / '.cache' / 'deep_features'
    model = PrototypeModel.from_export(
        model_data,
        deep_feature_config=deep_feature_config,
        deep_cache_dir=cache_dir,
    )
    weights = bundle.get('weights') or bundle.get('internal_feature_weights') or {}
    if not weights:
        raise RuntimeError('Deployment bundle does not contain feature weights.')
    return model, weights, bundle


def compact_prediction(row):
    sources = {}
    for key in (
        'prototype_score',
        'subprototype_score',
        'knn_score',
        'text_semantic_score',
        'hierarchy_score',
        'pairwise_score',
        'crop_rerank_score',
        'pair_confusion_score',
        'late_fusion_score',
        'concept_score',
    ):
        if row.get(key) is not None:
            sources[key] = round(float(row[key]), 6)
    if row.get('text_tiebreak_gate_reason') is not None:
        sources['text_tiebreak_gate_reason'] = row.get('text_tiebreak_gate_reason')
    if row.get('text_tiebreak_gap') is not None:
        sources['text_tiebreak_gap'] = round(float(row['text_tiebreak_gap']), 6)
    return {
        'label': row.get('label'),
        'score': round(float(row.get('score', 0.0)), 6),
        'sources': sources,
        'nearest_samples': [
            {
                'score': round(float(item.get('score', 0.0)), 6),
                'path': item.get('path'),
            }
            for item in (row.get('nearest_samples') or [])[:3]
        ],
    }


def _prediction_response(model, weights, bundle, model_path, image_path, top_k=5):
    results = model.predict(Path(image_path).expanduser().resolve(), weights)
    predictions = [compact_prediction(row) for row in results[:max(1, int(top_k))]]
    top = predictions[0] if predictions else {}
    second = predictions[1] if len(predictions) > 1 else {}
    top_score = float(top.get('score', 0.0)) if top else 0.0
    second_score = float(second.get('score', 0.0)) if second else None
    return {
        'model': str(Path(model_path).expanduser().resolve()),
        'image': str(Path(image_path).expanduser().resolve()),
        'task': bundle.get('task', {}),
        'predicted_label': top.get('label'),
        'score': round(top_score, 6),
        'confidence': round(top_score, 6),
        'top2_margin': None if second_score is None else round(top_score - second_score, 6),
        'predictions': predictions,
    }


def predict_with_loaded_bundle(model, weights, bundle, model_path, image_path, top_k=5):
    return _prediction_response(model, weights, bundle, model_path, image_path, top_k=top_k)


def predict_with_bundle(model_path, image_path, top_k=5, cache_dir=None):
    model, weights, bundle = load_deployment_bundle(model_path, cache_dir=cache_dir)
    return _prediction_response(model, weights, bundle, model_path, image_path, top_k=top_k)
