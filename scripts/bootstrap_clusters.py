import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ask2know.data.dataset_loader import DatasetLoader
from ask2know.features.deep_adapter import DeepFeatureAdapter
from ask2know.features.feature_config import resolve_deep_feature_config
from ask2know.sample_pool.manager import SamplePoolManager, _safe_name
from ask2know.utils.io_utils import load_yaml, save_json

VERSION = '0.4.2.1'


def _normalize_rows(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms <= 1e-8] = 1.0
    return arr / norms


def _farthest_first_init(x, k):
    centers = [0]
    min_dist = np.sum((x - x[0]) ** 2, axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(min_dist))
        centers.append(idx)
        dist = np.sum((x - x[idx]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, dist)
    return x[centers].copy()


def _kmeans(x, k, max_iter=80):
    if k <= 1:
        return np.zeros(len(x), dtype=np.int32), np.mean(x, axis=0, keepdims=True)
    centers = _farthest_first_init(x, k)
    labels = np.zeros(len(x), dtype=np.int32)
    for _ in range(max_iter):
        dists = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            members = x[labels == cluster_id]
            if len(members):
                centers[cluster_id] = np.mean(members, axis=0)
        centers = _normalize_rows(centers)
    return labels, centers


def _silhouette_score(x, labels):
    unique = sorted(set(int(v) for v in labels))
    if len(unique) <= 1 or len(unique) >= len(x):
        return 0.0
    d = np.sqrt(np.maximum(0.0, np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)))
    scores = []
    for i, label in enumerate(labels):
        same = labels == label
        same[i] = False
        a = float(np.mean(d[i, same])) if np.any(same) else 0.0
        b_vals = []
        for other in unique:
            if other == int(label):
                continue
            mask = labels == other
            if np.any(mask):
                b_vals.append(float(np.mean(d[i, mask])))
        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b)
        scores.append(0.0 if denom <= 1e-8 else (b - a) / denom)
    return float(np.mean(scores))


def _choose_auto_k(x, max_clusters, min_cluster_size):
    n = len(x)
    if n <= 2:
        return 1, [{'k': 1, 'score': 0.0, 'silhouette': 0.0, 'min_size': n}]
    upper = max(2, min(max_clusters, n - 1))
    candidates = []
    for k in range(2, upper + 1):
        labels, _ = _kmeans(x, k)
        counts = [int(np.sum(labels == i)) for i in range(k)]
        min_size = min(counts) if counts else 0
        silhouette = _silhouette_score(x, labels)
        small_penalty = 0.18 if min_size < min_cluster_size else 0.0
        score = silhouette - small_penalty - 0.015 * (k - 2)
        candidates.append({
            'k': k,
            'score': float(score),
            'silhouette': float(silhouette),
            'min_size': int(min_size),
        })
    candidates.sort(key=lambda item: item['score'], reverse=True)
    return int(candidates[0]['k']), candidates[:3]


def _extract_embeddings(samples, adapter):
    paths = [Path(item['path']) for item in samples]
    vectors = []
    kept_paths = []
    for path in paths:
        feats = adapter.extract_path(path)
        vec = feats.get(adapter.feature_name)
        if vec is None and feats:
            vec = next(iter(feats.values()))
        if vec is None:
            print('Skip unreadable image:', path)
            continue
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size:
            vectors.append(arr)
            kept_paths.append(path)
    if not vectors:
        return kept_paths, np.zeros((0, 0), dtype=np.float32)
    return kept_paths, _normalize_rows(np.stack(vectors).astype(np.float32))


def _cluster_summary(paths, x, labels, centers, preview_count):
    summaries = []
    for cluster_id in sorted(set(int(v) for v in labels)):
        idxs = np.where(labels == cluster_id)[0]
        center = centers[cluster_id]
        sims = x[idxs] @ center
        order = idxs[np.argsort(-sims)]
        summaries.append({
            'cluster_id': int(cluster_id),
            'count': int(len(idxs)),
            'representative': str(paths[int(order[0])]),
            'mean_similarity': float(np.mean(sims)),
            'preview': [str(paths[int(i)]) for i in order[:preview_count]],
            'all_paths': [str(paths[int(i)]) for i in order],
        })
    summaries.sort(key=lambda item: item['representative'])
    return summaries


def _load_objects(dataset_dir):
    path = Path(dataset_dir) / 'objects.json'
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding='utf-8')).get('objects', [])
    except Exception:
        return []


def _save_objects(dataset_dir, labels):
    existing = _load_objects(dataset_dir)
    by_name = {item.get('name'): dict(item) for item in existing if item.get('name')}
    for label in labels:
        safe = _safe_name(label)
        if safe not in by_name:
            by_name[safe] = {
                'object_id': f'C{len(by_name) + 1:03d}',
                'name': safe,
                'display_name': label,
                'description': f'added by bootstrap_clusters v{VERSION}',
            }
    objects = list(by_name.values())
    save_json(Path(dataset_dir) / 'objects.json', {'objects': objects})
    return objects


def _ask_mapping(summaries, names):
    assignments = {}
    remaining = list(names or [])
    for summary in summaries:
        print('\nCluster', summary['cluster_id'], f'({summary["count"]} images)')
        print('Representative:', summary['representative'])
        print('Mean similarity:', f'{summary["mean_similarity"]:.3f}')
        for path in summary['preview']:
            print('  ', path)
        if remaining:
            print('Available names:')
            for idx, name in enumerate(remaining, 1):
                print(f'  {idx}. {name}')
            ans = input('Map this cluster to name/number, or Enter to skip: ').strip()
            if ans.isdigit() and 1 <= int(ans) <= len(remaining):
                label = remaining.pop(int(ans) - 1)
            elif ans in remaining:
                label = ans
                remaining.remove(ans)
            else:
                label = ans
        else:
            label = input('Name this cluster, or Enter to skip: ').strip()
        if label:
            assignments[summary['cluster_id']] = label
    return assignments


def main():
    parser = argparse.ArgumentParser(description='Bootstrap train folders from datasets/unknown image clusters.')
    parser.add_argument('--config', required=True, help='Project task_config.yaml')
    parser.add_argument('--names', nargs='*', default=None, help='Known class names. If provided, k=len(names).')
    parser.add_argument('--max-clusters', type=int, default=12, help='Max k to try in auto mode.')
    parser.add_argument('--min-cluster-size', type=int, default=2, help='Penalty threshold for tiny auto clusters.')
    parser.add_argument('--preview-count', type=int, default=5, help='How many paths to show per cluster.')
    parser.add_argument('--no-copy', action='store_true', help='Only show clusters; do not copy into train.')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset_dir = cfg['paths']['dataset_dir']
    output_dir = Path(cfg['paths']['output_dir'])
    project_root = cfg.get('paths', {}).get('project_root')

    loader = DatasetLoader(dataset_dir)
    samples = loader.load_unknown_samples()
    if not samples:
        print(f'No learning samples found in {Path(dataset_dir) / "unknown"}')
        return 1

    deep_cfg = resolve_deep_feature_config(cfg)
    adapter = DeepFeatureAdapter(deep_cfg, cache_dir=output_dir / '.cache' / 'deep_features')
    paths, x = _extract_embeddings(samples, adapter)
    if len(paths) < 1:
        print('No readable images found.')
        return 1

    if args.names:
        k = len(args.names)
        candidates = [{'k': k, 'score': None, 'silhouette': None, 'min_size': None}]
    else:
        k, candidates = _choose_auto_k(x, args.max_clusters, args.min_cluster_size)

    labels, centers = _kmeans(x, k)
    summaries = _cluster_summary(paths, x, labels, centers, args.preview_count)

    print(f'Clustered {len(paths)} images from {Path(dataset_dir) / "unknown"} into {k} groups.')
    print('Candidate k scores:')
    for item in candidates:
        print(' ', item)

    assignments = _ask_mapping(summaries, args.names)
    if not assignments:
        print('No clusters confirmed. Nothing copied.')
        return 0

    if args.no_copy:
        print('Confirmed mapping, but --no-copy was set. Nothing copied.')
        return 0

    pool = SamplePoolManager(project_root=project_root, output_dir=output_dir, dataset_dir=dataset_dir, version=VERSION)
    objects = _save_objects(dataset_dir, assignments.values())
    pool.ensure_for_classes(assignments.values())
    pool.update_project_meta(classes=[item['name'] for item in objects if item.get('name')])

    copied = []
    for summary in summaries:
        label = assignments.get(summary['cluster_id'])
        if not label:
            continue
        safe = _safe_name(label)
        for path in summary['all_paths']:
            saved = pool.copy_confirmed(path, safe, source=f'bootstrap_cluster_{summary["cluster_id"]}')
            copied.append({'source': path, 'cluster_id': summary['cluster_id'], 'label': safe, 'saved_as': saved})

    log_path = pool.metadata_dir / 'bootstrap_cluster_map.jsonl'
    for item in copied:
        pool._append_jsonl(log_path, item)

    print(f'Copied {len(copied)} images into train folders.')
    print('Mapping log:', log_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
