import argparse
import json
from collections import Counter
from pathlib import Path


def load_report(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Compare two evaluation_report.json files.')
    parser.add_argument('--old', required=True)
    parser.add_argument('--new', required=True)
    parser.add_argument('--examples', type=int, default=12)
    args = parser.parse_args()

    old = load_report(Path(args.old))
    new = load_report(Path(args.new))
    old_map = {sample['path']: sample for sample in old.get('samples', [])}

    fixed = []
    regressed = []
    changed = []
    crop_paths = set()
    crop_rows = 0
    crop_reasons = Counter()
    for sample in new.get('samples', []):
        old_sample = old_map.get(sample.get('path'))
        if old_sample:
            if old_sample.get('predicted_label') != sample.get('predicted_label'):
                changed.append(sample)
            if not old_sample.get('correct') and sample.get('correct'):
                fixed.append(sample)
            if old_sample.get('correct') and not sample.get('correct'):
                regressed.append(sample)
        for pred in sample.get('top_predictions', []):
            reason = pred.get('crop_rerank_gate_reason')
            if reason:
                crop_reasons[reason] += 1
            if float(pred.get('crop_rerank_score_weight_used') or 0.0) > 0.0:
                crop_paths.add(sample.get('path'))
                crop_rows += 1

    def fmt_accuracy(report):
        return f"{report.get('correct_count')}/{report.get('eval_sample_count')} = {float(report.get('accuracy', 0.0)):.4f}"

    print('old_accuracy:', fmt_accuracy(old), 'schema:', old.get('schema_version'))
    print('new_accuracy:', fmt_accuracy(new), 'schema:', new.get('schema_version'))
    print('changed_predictions:', len(changed))
    print('fixed:', len(fixed))
    print('regressed:', len(regressed))
    print('crop_samples_weighted:', len(crop_paths))
    print('crop_prediction_rows_weighted:', crop_rows)
    print('crop_gate_reasons:', crop_reasons.most_common())
    print('fixed_by_class:', Counter(sample.get('true_label') for sample in fixed).most_common())
    print('regressed_by_class:', Counter(sample.get('true_label') for sample in regressed).most_common())
    old_per_class = old.get('per_class', {})
    new_per_class = new.get('per_class', {})
    deltas = []
    for label, item in new_per_class.items():
        old_item = old_per_class.get(label, {})
        deltas.append((
            int(item.get('correct', 0)) - int(old_item.get('correct', 0)),
            label,
            int(old_item.get('correct', 0)),
            int(item.get('correct', 0)),
        ))
    print('per_class_gains:', [row for row in sorted(deltas, reverse=True) if row[0] > 0])
    print('per_class_losses:', [row for row in sorted(deltas) if row[0] < 0])

    print('fixed_examples:')
    for sample in fixed[:args.examples]:
        old_sample = old_map[sample['path']]
        print(' ', sample['true_label'], old_sample.get('predicted_label'), '->', sample.get('predicted_label'), sample['path'])

    print('regressed_examples:')
    for sample in regressed[:args.examples]:
        old_sample = old_map[sample['path']]
        print(' ', sample['true_label'], old_sample.get('predicted_label'), '->', sample.get('predicted_label'), sample['path'])


if __name__ == '__main__':
    raise SystemExit(main())
