import argparse
import json
import sys
from datetime import datetime
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

VERSION = '0.4.6.2a'


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


def _cluster_summary(paths, x, labels, centers, preview_count, outlier_count=3):
    summaries = []
    for cluster_id in sorted(set(int(v) for v in labels)):
        idxs = np.where(labels == cluster_id)[0]
        center = centers[cluster_id]
        sims = x[idxs] @ center
        mean_similarity = float(np.mean(sims))
        similarity_std = float(np.std(sims))
        order = idxs[np.argsort(-sims)]
        risk_order = idxs[np.argsort(sims)]
        sim_by_idx = {int(idx): float(sim) for idx, sim in zip(idxs, sims)}
        threshold = mean_similarity - max(similarity_std, 0.03)
        review_idxs = [
            int(i) for i in risk_order
            if len(idxs) > 1 and sim_by_idx[int(i)] <= threshold
        ][:outlier_count]
        if not review_idxs and len(idxs) > 2 and outlier_count > 0:
            review_idxs = [int(i) for i in risk_order[:min(outlier_count, len(risk_order))]]
        review_idx_set = set(review_idxs)
        items = []
        for rank, idx in enumerate(order, 1):
            idx = int(idx)
            items.append({
                'rank': rank,
                'path': str(paths[idx]),
                'similarity': sim_by_idx[idx],
                'review_candidate': idx in review_idx_set,
            })
        summaries.append({
            'cluster_id': int(cluster_id),
            'count': int(len(idxs)),
            'representative': str(paths[int(order[0])]),
            'mean_similarity': mean_similarity,
            'similarity_std': similarity_std,
            'preview': [str(paths[int(i)]) for i in order[:preview_count]],
            'review_candidates': [
                {
                    'path': str(paths[int(i)]),
                    'similarity': sim_by_idx[int(i)],
                }
                for i in review_idxs
            ],
            'all_paths': [str(paths[int(i)]) for i in order],
            'items': items,
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
        print('Mean similarity:', f'{summary["mean_similarity"]:.3f}', 'std:', f'{summary["similarity_std"]:.3f}')
        for path in summary['preview']:
            print('  ', path)
        if summary.get('review_candidates'):
            print('Potential outliers / review first:')
            for item in summary['review_candidates']:
                print('  ', f'{item["similarity"]:.3f}', item['path'])
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


def _parse_skip_answer(answer, shown_items):
    text = str(answer or '').strip()
    if not text:
        return set(), False
    if text.lower() in {'a', 'all', '*'}:
        return set(), True

    skipped = set()
    by_index = {str(idx): item['path'] for idx, item in enumerate(shown_items, 1)}
    tokens = [part.strip() for part in text.replace(',', ' ').split() if part.strip()]
    for token in tokens:
        if '-' in token:
            left, right = token.split('-', 1)
            if left.isdigit() and right.isdigit():
                start, end = int(left), int(right)
                if start > end:
                    start, end = end, start
                for idx in range(start, end + 1):
                    path = by_index.get(str(idx))
                    if path:
                        skipped.add(path)
                continue
        path = by_index.get(token)
        if path:
            skipped.add(path)
    return skipped, False


def _ask_image_skips(summaries, assignments, review_count, initial_skips=None):
    skipped = set(str(Path(p)) for p in (initial_skips or []))
    for summary in summaries:
        label = assignments.get(summary['cluster_id'])
        if not label:
            continue
        items = sorted(summary.get('items', []), key=lambda item: (not item.get('review_candidate'), item['similarity']))
        if not items:
            continue
        print('\nReview before copy:', f'cluster {summary["cluster_id"]} -> {_safe_name(label)}')
        print('Enter numbers/ranges to skip single images, "all" to skip this cluster, or Enter to keep.')
        shown = items if review_count <= 0 else items[:review_count]
        if len(items) > len(shown):
            print(f'Showing {len(shown)} review-priority images out of {len(items)}. Full list is written to the review report.')
        for idx, item in enumerate(shown, 1):
            marker = 'outlier' if item.get('review_candidate') else 'sample'
            print(f'  {idx}. [{marker}] {item["similarity"]:.3f} {item["path"]}')
        answer = input('Skip image numbers/ranges: ').strip()
        selected, skip_all = _parse_skip_answer(answer, shown)
        if skip_all:
            skipped.update(item['path'] for item in summary.get('items', []))
        else:
            skipped.update(selected)
    return skipped


def _build_review_rows(summaries, assignments, skipped_paths=None):
    skipped = set(str(Path(p)) for p in (skipped_paths or []))
    rows = []
    for summary in summaries:
        label = assignments.get(summary['cluster_id'])
        safe = _safe_name(label) if label else ''
        for item in summary.get('items', []):
            path = str(Path(item['path']))
            if not label:
                action = 'unassigned'
            elif path in skipped:
                action = 'skip'
            else:
                action = 'copy'
            rows.append({
                'cluster_id': summary['cluster_id'],
                'label': safe,
                'display_label': label or '',
                'action': action,
                'path': path,
                'similarity': float(item['similarity']),
                'review_candidate': bool(item.get('review_candidate')),
                'cluster_mean_similarity': float(summary['mean_similarity']),
                'cluster_similarity_std': float(summary['similarity_std']),
            })
    return rows


def _metadata_dir(project_root, output_dir):
    if project_root:
        return Path(project_root) / 'metadata'
    return Path(output_dir) / 'metadata'


def _write_review_report(metadata_dir, rows, candidates, report_prefix='bootstrap_review'):
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    jsonl_path = metadata_dir / f'{report_prefix}_{stamp}.jsonl'
    md_path = metadata_dir / f'{report_prefix}_{stamp}.md'

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    lines = [
        f'# Bootstrap review {stamp}',
        '',
        '## Candidate k scores',
    ]
    for item in candidates:
        lines.append(f'- {item}')
    lines.extend(['', '## Copy plan', ''])
    for idx, row in enumerate(rows, 1):
        marker = 'review' if row['review_candidate'] else 'ok'
        lines.append(
            f'{idx}. [{row["action"]}] cluster={row["cluster_id"]} '
            f'label={row["label"] or "-"} sim={row["similarity"]:.3f} '
            f'{marker} path={row["path"]}'
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return jsonl_path, md_path


def main():
    parser = argparse.ArgumentParser(description='Bootstrap train folders from datasets/unknown image clusters.')
    parser.add_argument('--config', required=True, help='Project task_config.yaml')
    parser.add_argument('--names', nargs='*', default=None, help='Known class names. If provided, k=len(names).')
    parser.add_argument('--max-clusters', type=int, default=12, help='Max k to try in auto mode.')
    parser.add_argument('--min-cluster-size', type=int, default=2, help='Penalty threshold for tiny auto clusters.')
    parser.add_argument('--preview-count', type=int, default=5, help='How many paths to show per cluster.')
    parser.add_argument('--outlier-count', type=int, default=3, help='How many low-similarity samples to flag per cluster.')
    parser.add_argument('--review-count', type=int, default=30, help='How many review-priority images to print before copying. Use 0 to print all.')
    parser.add_argument('--skip-path', nargs='*', default=None, help='Exact image paths to skip during copy.')
    parser.add_argument('--no-review', action='store_true', help='Do not prompt for per-image skips before copy.')
    parser.add_argument('--report-only', '--dry-run', action='store_true', help='Write review reports and do not copy into train.')
    parser.add_argument('--no-copy', action='store_true', help='Deprecated alias for --report-only.')
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
    summaries = _cluster_summary(paths, x, labels, centers, args.preview_count, args.outlier_count)

    print(f'Clustered {len(paths)} images from {Path(dataset_dir) / "unknown"} into {k} groups.')
    print('Candidate k scores:')
    for item in candidates:
        print(' ', item)

    assignments = _ask_mapping(summaries, args.names)
    report_only = args.report_only or args.no_copy
    if not assignments and not report_only:
        print('No clusters confirmed. Nothing copied.')
        return 0

    skipped_paths = set(str(Path(p)) for p in (args.skip_path or []))
    if assignments and not args.no_review:
        skipped_paths = _ask_image_skips(summaries, assignments, args.review_count, skipped_paths)

    review_rows = _build_review_rows(summaries, assignments, skipped_paths)
    report_dir = _metadata_dir(project_root, output_dir)
    jsonl_path, md_path = _write_review_report(report_dir, review_rows, candidates)
    print('Review report:', md_path)
    print('Review JSONL:', jsonl_path)

    if report_only:
        print('Report-only mode. Nothing copied into train folders.')
        return 0

    copy_labels = list(dict.fromkeys(row['display_label'] for row in review_rows if row['action'] == 'copy'))
    if not copy_labels:
        print('No images selected for copy after review. Nothing copied.')
        return 0

    pool = SamplePoolManager(project_root=project_root, output_dir=output_dir, dataset_dir=dataset_dir, version=VERSION)
    objects = _save_objects(dataset_dir, copy_labels)
    pool.ensure_for_classes(copy_labels)
    pool.update_project_meta(classes=[item['name'] for item in objects if item.get('name')])

    copied = []
    for row in review_rows:
        if row['action'] != 'copy':
            continue
        saved = pool.copy_confirmed(row['path'], row['label'], source=f'bootstrap_cluster_{row["cluster_id"]}')
        copied.append({
            'source': row['path'],
            'cluster_id': row['cluster_id'],
            'label': row['label'],
            'similarity': row['similarity'],
            'review_candidate': row['review_candidate'],
            'saved_as': saved,
        })

    log_path = pool.metadata_dir / 'bootstrap_cluster_map.jsonl'
    for item in copied:
        pool._append_jsonl(log_path, item)

    print(f'Copied {len(copied)} images into train folders.')
    print('Mapping log:', log_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
