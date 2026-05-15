import argparse
import os
import platform
import subprocess
from pathlib import Path

from ask2know.utils.io_utils import load_yaml, save_json, ensure_dir
from ask2know.data.dataset_loader import DatasetLoader
from ask2know.inference.prototype_model import PrototypeModel
from ask2know.inference.uncertainty import is_globally_uncertain, top_gap, score_spread, saturated_feature_ratio
from ask2know.questions.question_selector import QuestionSelector
from ask2know.questions.question_generator import generate_natural_question, generate_question_context
from ask2know.learning.weights import AdaptiveWeights
from ask2know.learning.feedback_updater import apply_answer_to_weights, update_question_reward
from ask2know.sample_pool.manager import SamplePoolManager
from ask2know.experience.pairwise import PairwiseExperienceManager
from ask2know.concepts.basic_concepts import DISPLAY_NAMES, summarize_concepts
from ask2know.features.feature_config import (
    expand_feature_adjustments,
    initial_feature_weights,
    parse_feature_config,
    summarize_group_weights,
)

VERSION = '0.3.7.2'


def open_image_file(image_path):
    try:
        system = platform.system()
        image_path = str(image_path)
        if system == 'Windows':
            os.startfile(image_path)
        elif system == 'Darwin':
            subprocess.Popen(['open', image_path])
        else:
            subprocess.Popen(['xdg-open', image_path])
    except Exception:
        pass


def display_results(results, max_items=5):
    for i, r in enumerate(results[:max_items], 1):
        visible_detail = r.get('group_detail') or r.get('detail', {})
        detail = ', '.join(f'{k}:{v:.2f}' for k, v in visible_detail.items())
        print(f'{i}. {r["label"]}: {r["score"]:.3f}  ({detail})')


def pretty_weights(weights):
    return {k: round(v, 3) for k, v in weights.items()}


def print_header(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def class_names(objects):
    return [o['name'] for o in objects]


def save_objects_file(dataset_dir, objects):
    save_json(Path(dataset_dir) / 'objects.json', {'objects': objects})


def pretty_group_weights(weights, feature_spec):
    return pretty_weights(summarize_group_weights(weights, feature_spec))


def expand_concept_feature_hints(concepts, feature_spec):
    expanded = []
    for item in concepts or []:
        copied = dict(item)
        copied['important_features'] = expand_feature_adjustments(item.get('important_features', []), feature_spec)
        copied['weak_features'] = expand_feature_adjustments(item.get('weak_features', []), feature_spec)
        expanded.append(copied)
    return expanded


def correction_option_enabled(reason_id, feature_spec):
    enabled = set(feature_spec.get('display_features', []))
    if reason_id in ('background', 'other', 'quality'):
        return True
    if reason_id == 'cluster':
        return bool(enabled & {'shape', 'texture'})
    return reason_id in enabled


def ask_true_label(objects, allow_new=True, allow_reject=True):
    names = class_names(objects)
    print('\n请选择这张图的真实类别：')
    for i, name in enumerate(names, 1):
        print(f'{i}. {name}')
    if allow_new:
        print('N. 新类别 / 当前类别列表里没有')
    if allow_reject:
        print('R. 不适合学习 / 太糊 / 主体不明显')
    print('S. 跳过')

    ans = input('请输入编号/类别名/N/R/S: ').strip()
    if not ans:
        return 'skip', None
    low = ans.lower()
    if low in ('s', 'skip'):
        return 'skip', None
    if low in ('r', 'reject', 'rejected'):
        return 'reject', None
    if low in ('n', 'new'):
        new_name = input('请输入新类别英文名，例如 peach: ').strip()
        return ('new', new_name) if new_name else ('skip', None)
    if ans.isdigit():
        idx = int(ans)
        if 1 <= idx <= len(names):
            return 'class', names[idx - 1]
    if ans in names:
        return 'class', ans
    print('输入无法识别，本次跳过。')
    return 'skip', None


def confirm_prediction(pred_label, objects):
    print('\n确认本次识别结果：')
    print(f'Y. 正确，加入 confirmed 样本池：{pred_label}')
    print('N. 错误，我选择正确类别')
    print('C. 暂时放入 candidate，不进入正式学习')
    print('R. 不适合学习 / 拒绝样本')
    print('S. 跳过')
    ans = input('请输入 Y/N/C/R/S: ').strip().lower()
    if ans == 'y':
        return 'confirmed', pred_label
    if ans == 'n':
        mode, label = ask_true_label(objects, allow_new=True, allow_reject=True)
        if mode in ('class', 'new') and label:
            return mode, label
        if mode == 'reject':
            return 'reject', None
        return 'skip', None
    if ans == 'c':
        return 'candidate', pred_label
    if ans == 'r':
        return 'reject', None
    return 'skip', None


def add_new_object_if_needed(objects, label):
    if not label or label in class_names(objects):
        return objects
    objects.append({
        'object_id': f'C{len(objects) + 1:03d}',
        'name': label,
        'display_name': label,
        'description': f'added during v{VERSION} interactive learning'
    })
    return objects


def handle_sample_decision(decision, label, sample_path, model, pool, objects, dataset_dir, pool_enabled=True, move_after_decision=True):
    if decision in ('confirmed', 'class', 'new') and label:
        objects = add_new_object_if_needed(objects, label)
        save_objects_file(dataset_dir, objects)
        pool.ensure_for_classes(class_names(objects))
        pool.update_project_meta(classes=class_names(objects))
        if not pool_enabled or not move_after_decision:
            model.add_confirmed_sample(label, sample_path)
            print('已在本轮运行中学习该 confirmed 样本，但按配置不移动文件到长期训练库。')
            return {'decision': 'confirmed', 'label': label, 'saved_to': None, 'persisted': False}
        saved_path = pool.add_confirmed(sample_path, label)
        model.add_confirmed_sample(label, saved_path)
        print('已加入长期训练库 confirmed/train:', label)
        print('保存为:', saved_path)
        return {'decision': 'confirmed', 'label': label, 'saved_to': saved_path, 'persisted': True}

    if decision == 'candidate' and label:
        if not pool_enabled or not move_after_decision:
            print('已记录为 candidate，但按配置不移动文件。')
            return {'decision': 'candidate', 'label': label, 'saved_to': None, 'persisted': False}
        saved_path = pool.add_candidate(sample_path, label)
        print('已加入 candidate，不进入正式学习:', label)
        print('保存为:', saved_path)
        return {'decision': 'candidate', 'label': label, 'saved_to': saved_path, 'persisted': True}

    if decision == 'reject':
        if not pool_enabled or not move_after_decision:
            print('已记录为 rejected，但按配置不移动文件。')
            return {'decision': 'rejected', 'label': None, 'saved_to': None, 'persisted': False}
        saved_path = pool.add_rejected(sample_path, 'rejected')
        print('已加入 rejected。')
        print('保存为:', saved_path)
        return {'decision': 'rejected', 'label': None, 'saved_to': saved_path, 'persisted': True}

    if decision == 'unknown':
        if not pool_enabled or not move_after_decision:
            print('已记录为 unknown，但按配置不移动文件。')
            return {'decision': 'unknown', 'label': None, 'saved_to': None, 'persisted': False}
        saved_path = pool.add_unknown(sample_path)
        print('已加入 unknown。')
        print('保存为:', saved_path)
        return {'decision': 'unknown', 'label': None, 'saved_to': saved_path, 'persisted': True}

    print('已跳过，文件仍保留在 unlabeled。')
    return {'decision': 'skip', 'label': None, 'saved_to': None}




def _parse_multi_choice(text, valid_keys):
    raw = (text or '').strip().upper().replace('，', ',').replace('、', ',').replace(' ', ',')
    if not raw:
        return []
    if ',' in raw:
        parts = [x.strip() for x in raw.split(',') if x.strip()]
    else:
        parts = list(raw)
    out = []
    for x in parts:
        if x in valid_keys and x not in out:
            out.append(x)
    return out


def ask_correction_reason(predicted_label, true_label, pairwise_manager, adaptive_weights, feature_spec, sample_path=None):
    """Ask why a wrong prediction happened and store pairwise experience.

    v0.3.7.2 supports multi-select answers because real differences often involve
    color + shape + texture together.
    """
    if not predicted_label or not true_label or predicted_label == true_label:
        return None

    print('\n错误后追问：')
    print(generate_question_context({'label': predicted_label}, {'label': true_label}, sample_path=sample_path, true_label=true_label, phase='post_error'))
    print('\n系统刚才识别错了。为了避免下次再犯同类错误，请选择主要原因，可以多选。')
    options = [
        item for item in pairwise_manager.correction_options(predicted_label, true_label)
        if correction_option_enabled(item[1], feature_spec)
    ]
    option_map = {}
    for key, reason_id, text, action in options:
        option_map[key.upper()] = (reason_id, text, action)
        print(f'{key}. {text}')
    ans = input('请输入选项，可多选，例如 A,B 或 ABC；直接回车表示不确定: ').strip().upper()
    selected_keys = _parse_multi_choice(ans, set(option_map.keys()))
    if not selected_keys:
        selected_keys = [
            key for key, (reason_id, _, _) in option_map.items()
            if reason_id == 'other'
        ] or list(option_map.keys())[-1:]

    selected_items = []
    inc = []
    dec = []
    for key in selected_keys:
        reason_id, reason_text, action = option_map[key]
        selected_items.append((reason_id, reason_text))
        inc.extend(action.get('increase', []))
        dec.extend(action.get('decrease', []))

    before, after = adaptive_weights.update(
        expand_feature_adjustments(inc, feature_spec),
        expand_feature_adjustments(dec, feature_spec),
    )
    pair = pairwise_manager.record_corrections(predicted_label, true_label, selected_items)

    print('已记录类别对经验:', f'{predicted_label} vs {true_label}')
    print('错因:', '；'.join([x[1] for x in selected_items]))
    print('权重更新前:', pretty_group_weights(before, feature_spec))
    print('权重更新后:', pretty_group_weights(after, feature_spec))

    return {
        'predicted': predicted_label,
        'true_label': true_label,
        'reasons': [{'reason_id': x[0], 'reason_text': x[1]} for x in selected_items],
        'weights_before': before,
        'weights_after': after,
        'pairwise': pair,
    }


def make_experience_report(cfg, logs, final_weights, question_state, objects, feature_spec, pairwise_state=None):
    task = cfg.get('task', {})
    report = {
        'framework': 'Ask2Know',
        'version': VERSION,
        'task': task,
        'objects': objects,
        'summary': {
            'total_samples': len(logs),
            'asked_count': sum(1 for x in logs if x.get('asked')),
            'global_uncertain_count': sum(1 for x in logs if x.get('global_uncertain')),
            'confirmed_count': sum(1 for x in logs if x.get('pool', {}).get('decision') == 'confirmed'),
            'candidate_count': sum(1 for x in logs if x.get('pool', {}).get('decision') == 'candidate'),
            'rejected_count': sum(1 for x in logs if x.get('pool', {}).get('decision') == 'rejected'),
            'skipped_count': sum(1 for x in logs if x.get('pool', {}).get('decision') == 'skip'),
        },
        'final_feature_weights': summarize_group_weights(final_weights, feature_spec),
        'internal_feature_weights': final_weights,
        'question_state': question_state,
        'pairwise_state': pairwise_state or {},
        'learned_notes': [],
        'next_suggestions': []
    }

    for item in logs:
        if item.get('global_uncertain'):
            report['learned_notes'].append({
                'sample': item.get('sample'),
                'type': 'global_uncertainty',
                'reason': item.get('uncertainty_reason'),
                'decision': item.get('pool', {}).get('decision'),
                'label': item.get('pool', {}).get('label')
            })
        elif item.get('correction_reason'):
            report['learned_notes'].append({
                'sample': item.get('sample'),
                'type': 'correction_reason',
                'predicted': item.get('correction_reason', {}).get('predicted'),
                'true_label': item.get('correction_reason', {}).get('true_label'),
                'reasons': item.get('correction_reason', {}).get('reasons', []),
            })
        elif item.get('asked') and item.get('answer_text'):
            report['learned_notes'].append({
                'sample': item.get('sample'),
                'type': 'question_feedback',
                'question': item.get('question'),
                'answer_text': item.get('answer_text'),
                'helpful': item.get('helpful'),
                'weights_after': item.get('weights_after'),
                'concept_evidence': item.get('generated_question', {}).get('concept_evidence', '')
            })

    visible_weights = summarize_group_weights(final_weights, feature_spec)
    low_weight = [k for k, v in visible_weights.items() if v < 0.12]
    high_weight = [k for k, v in visible_weights.items() if v > 0.30]
    if low_weight:
        report['next_suggestions'].append(f'这些特征当前权重较低：{", ".join(low_weight)}。后续可检查它们是否在当前任务中不可靠。')
    if high_weight:
        report['next_suggestions'].append(f'这些特征当前权重较高：{", ".join(high_weight)}。后续可围绕它们收集更清晰、更典型的样本。')
    if report['summary']['global_uncertain_count'] > 0:
        report['next_suggestions'].append('存在整体不确定样本。建议优先检查背景复杂、主体不明显或类别样本太少的问题。')
    if report['summary']['candidate_count'] > 0:
        report['next_suggestions'].append('candidate 样本需要后续人工确认，确认前不要作为正式训练知识。')
    correction_count = sum(1 for x in logs if x.get('correction_reason'))
    if correction_count > 0:
        report['next_suggestions'].append('本轮存在识别错误后的错因追问。下一轮会优先参考 pairwise_experience.json 中的类别对经验。')
    return report



def make_experience_summary(pairwise_state, objects):
    """Generate a compact self-summary from pairwise experience.

    This is a weak summary, not final truth. Its goal is to let a2k start
    explaining what it has learned so far.
    """
    labels = [o.get('name') for o in objects]
    summary = {
        'schema_version': VERSION,
        'summary_type': 'weak_experience_summary',
        'notice': '这是系统基于纠错记录自动生成的弱总结，不代表最终真理。',
        'classes': {
            name: {
                'known_confusions': [],
                'possible_useful_features': {},
                'possible_useful_concepts': {},
            }
            for name in labels
        },
        'pairs': {}
    }
    pairs = (pairwise_state or {}).get('pairs', {})
    for key, pair in pairs.items():
        classes = pair.get('classes', [])
        useful = pair.get('useful_features', {})
        useful_concepts = pair.get('useful_concepts', {})
        reasons = pair.get('reason_counts', {})
        item = {
            'classes': classes,
            'confused_count': pair.get('confused_count', 0),
            'correction_count': pair.get('correction_count', 0),
            'useful_features': useful,
            'useful_concepts': pair.get('useful_concepts', {}),
            'reason_counts': reasons,
            'short_text': ''
        }
        if useful_concepts:
            ranked = sorted(useful_concepts.items(), key=lambda x: x[1], reverse=True)
            item['short_text'] = '、'.join([x[0] for x in ranked[:4]]) + ' 可能是这组类别的重要基础视觉概念。'
        elif useful:
            ranked = sorted(useful.items(), key=lambda x: x[1], reverse=True)
            item['short_text'] = '、'.join([x[0] for x in ranked[:3]]) + ' 可能是这组类别的重要区分点。'
        else:
            item['short_text'] = '暂时没有稳定区分经验。'
        summary['pairs'][key] = item
        for cls in classes:
            if cls not in summary['classes']:
                summary['classes'][cls] = {
                    'known_confusions': [],
                    'possible_useful_features': {},
                    'possible_useful_concepts': {},
                }
            summary['classes'][cls]['known_confusions'].append(key)
            for feat, count in useful.items():
                cur = summary['classes'][cls]['possible_useful_features'].get(feat, 0)
                summary['classes'][cls]['possible_useful_features'][feat] = cur + int(count)
            for concept, count in useful_concepts.items():
                cur = summary['classes'][cls]['possible_useful_concepts'].get(concept, 0)
                summary['classes'][cls]['possible_useful_concepts'][concept] = cur + int(count)
    return summary


def make_class_understanding_summary(model, objects, pairwise_state=None):
    """Summarize what the current concept prototypes say about each class."""
    diagnostic_concepts = {'clear_foreground', 'background_interference'}
    generic_summary_concepts = {'color_family', 'rectangular_like'}
    summary_priority = {
        'red': 10, 'orange': 10, 'yellow': 10, 'green': 10, 'cyan': 10,
        'blue': 10, 'purple': 10, 'pink': 10, 'brown': 10, 'black': 10,
        'white': 10, 'gray': 10, 'dark': 9, 'bright': 9,
        'round': 8, 'elongated': 8, 'pear_like': 8,
        'fuzzy_surface': 7, 'rough_peel': 7, 'speckled_surface': 7, 'glossy_surface': 7,
        'texture_rich': 6, 'smooth_surface': 6, 'edge_rich': 6,
        'cluster_like': 5, 'repeated_parts': 5, 'single_object': 5,
    }
    labels = [o.get('name') for o in objects]
    concept_prototypes = getattr(model, 'concept_prototypes', {}) or {}
    concept_counts = getattr(model, 'concept_counts', {}) or {}
    pairs = (pairwise_state or {}).get('pairs', {})
    summary = {
        'schema_version': VERSION,
        'summary_type': 'class_understanding_summary',
        'notice': '这是系统根据当前训练样本和已确认样本自动生成的类别理解草稿，供用户检查，不代表最终真理。',
        'classes': {}
    }

    for label in labels:
        concepts = concept_prototypes.get(label, {}) or {}
        raw_strong = summarize_concepts(concepts, top_n=12, min_score=0.35)
        strong = [item for item in raw_strong if item['id'] not in diagnostic_concepts][:8]
        weak = [
            {
                'id': name,
                'name': DISPLAY_NAMES.get(name, name),
                'score': round(float(score), 3),
            }
            for name, score in sorted(concepts.items(), key=lambda x: float(x[1]), reverse=True)
            if name not in diagnostic_concepts and 0.22 <= float(score) < 0.35
        ][:6]
        confusion_keys = [
            key for key, pair in pairs.items()
            if label in [str(x) for x in pair.get('classes', [])]
        ]
        if strong:
            summary_items = [
                item for item in strong
                if item['id'] not in generic_summary_concepts or float(item['score']) >= 0.82
            ]
            summary_items.sort(
                key=lambda item: (summary_priority.get(item['id'], 0), float(item['score'])),
                reverse=True,
            )
            if not summary_items:
                summary_items = strong
            concept_text = '、'.join([item['name'] for item in summary_items[:5]])
            summary_text = f'系统目前认为 {label} 更像：{concept_text}。'
        else:
            summary_text = f'系统目前对 {label} 的可解释概念证据不足，需要更多清晰样本或用户纠正。'
        needs_check = []
        if not strong:
            needs_check.append('这个类别的样本可能太少，或当前浅层特征无法形成稳定概念。')
        if float(concepts.get('background_interference', 0.0)) >= 0.35:
            needs_check.append('请检查该类别是否被背景、光线、遮挡或主体清晰度影响。')
        if weak:
            needs_check.append('低置信概念需要用户复核：' + '、'.join([item['name'] for item in weak[:4]]) + '。')
        if confusion_keys:
            needs_check.append('该类别存在已记录混淆，可结合类别对经验一起检查。')

        summary['classes'][label] = {
            'summary_text': summary_text,
            'strong_concepts': strong,
            'weak_or_uncertain_concepts': weak,
            'concept_sample_counts': concept_counts.get(label, {}),
            'known_confusions': confusion_keys,
            'needs_user_check': needs_check,
        }
    return summary


def render_class_understanding_markdown(summary):
    lines = [
        '# Ask2Know 类别理解总结',
        '',
        summary.get('notice', ''),
        '',
    ]
    for label, item in (summary.get('classes') or {}).items():
        lines.append(f'## {label}')
        lines.append('')
        lines.append(item.get('summary_text', ''))
        lines.append('')
        strong = item.get('strong_concepts') or []
        if strong:
            lines.append('主要概念：')
            for concept in strong:
                lines.append(f'- {concept["name"]}: {concept["score"]:.3f}')
        else:
            lines.append('主要概念：暂无稳定概念。')
        checks = item.get('needs_user_check') or []
        if checks:
            lines.append('')
            lines.append('建议用户检查：')
            for text in checks:
                lines.append(f'- {text}')
        confusions = item.get('known_confusions') or []
        if confusions:
            lines.append('')
            lines.append('已记录混淆：' + '、'.join(confusions))
        lines.append('')
    return '\n'.join(lines).rstrip() + '\n'


def main():
    parser = argparse.ArgumentParser(description='Ask2Know low-sample active teaching demo')
    parser.add_argument('--config', default='configs/fruit_demo.yaml')
    parser.add_argument('--preview', action='store_true', help='手动开启图片预览。默认关闭，避免 Windows 图片查看器占用文件导致卡死')
    parser.add_argument('--no-preview', action='store_true', help='兼容旧参数；v0.3.7.2 默认就是不预览')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    task = cfg.get('task', {})
    dataset_dir = cfg['paths']['dataset_dir']
    output_dir = Path(cfg['paths']['output_dir'])
    project_root = cfg.get('paths', {}).get('project_root')
    ensure_dir(output_dir)
    ensure_dir(output_dir / 'logs')

    loader = DatasetLoader(dataset_dir)
    objects = loader.load_objects()
    concepts = loader.load_concepts()
    feature_spec = parse_feature_config(cfg, classes=class_names(objects) or cfg.get('classes', []))
    feature_names = feature_spec['scoring_features']
    display_features = feature_spec['display_features']
    system_feature_names = feature_spec['system_features']

    pool = SamplePoolManager(project_root=project_root, output_dir=output_dir, dataset_dir=dataset_dir, version=VERSION)
    pairwise = PairwiseExperienceManager(metadata_dir=pool.metadata_dir, version=VERSION)
    pool.ensure_for_classes(class_names(objects))
    pool.update_project_meta(project_name=task.get('name', 'unknown'), classes=class_names(objects))

    if cfg.get('train_import', {}).get('auto_rename', True):
        train_renamed = pool.normalize_train_images(class_names(objects))
        if train_renamed:
            print(f'已自动规范化 train 图片命名：{len(train_renamed)} 张。')

    if cfg.get('unlabeled_import', {}).get('auto_rename', True):
        renamed = pool.normalize_unlabeled()
        if renamed:
            print(f'已自动规范化 unlabeled 图片命名：{len(renamed)} 张。')

    train_samples = loader.load_train_samples()
    unlabeled = loader.load_unlabeled_samples()

    if not objects:
        print('没有找到对象类别。请先运行 init_task 自动创建任务。')
        print('示例：python scripts/init_task.py --name fruit_task --classes apple banana pear --output D:\\a2k_test')
        return
    if not train_samples:
        print('没有找到训练样本。')
        print(f'请把每个类别的已知图片放到：{Path(dataset_dir) / "train" / "类别名"}')
        return
    if not unlabeled:
        print('没有找到待识别样本。')
        print(f'请把未知图片放到：{Path(dataset_dir) / "unlabeled"}')
        return


    print_header(f'Ask2Know v{VERSION} - 持续学习与数据累积版')
    print('任务名:', task.get('name', 'unknown'))
    print('任务类型:', task.get('type', 'image_object_recognition'))
    if task.get('description'):
        print('任务说明:', task.get('description'))
    print('数据目录:', dataset_dir)
    print('输出目录:', output_dir)
    if project_root:
        print('项目目录:', project_root)
    print('对象类别:', ', '.join(class_names(objects)))
    print(f'训练样本数: {len(train_samples)}，待识别样本数: {len(unlabeled)}')
    print('启用特征:', ', '.join(display_features))
    if system_feature_names:
        print('系统质量检查:', ', '.join(system_feature_names))

    initial_weights = initial_feature_weights(cfg, feature_spec)
    aw = AdaptiveWeights(
        initial_weights,
        cfg['learning'].get('update_step', 0.07),
        cfg['learning'].get('min_weight', 0.05),
        cfg['learning'].get('max_weight', 0.70)
    )
    aw.apply_concepts(expand_concept_feature_hints(concepts, feature_spec))

    model = PrototypeModel(
        feature_names,
        augmentation_config=cfg.get('augmentation', {}),
        concept_config=cfg.get('concepts', {'enable': True, 'score_weight': 0.25}),
        system_feature_names=system_feature_names,
        feature_groups=feature_spec['group_features'],
    ).fit(train_samples)
    q_selector = QuestionSelector(pairwise_manager=pairwise, enabled_features=display_features)
    logs = []
    confidence_cfg = cfg.get('confidence', {})
    question_cfg = cfg.get('question', {})
    sample_pool_cfg = cfg.get('sample_pool', {})
    ask_threshold = confidence_cfg.get('ask_user_threshold', 0.12)
    auto_accept_threshold = confidence_cfg.get('auto_accept_threshold', 0.88)
    enable_question_reward = question_cfg.get('enable_question_reward', True)
    require_confirm = sample_pool_cfg.get('require_confirm_before_learning', True)
    pool_enabled = sample_pool_cfg.get('enable', True)
    move_after_decision = sample_pool_cfg.get('move_unlabeled_after_decision', True)

    for idx, sample in enumerate(unlabeled, 1):
        sample_path = sample['path']
        print_header(f'正在识别第 {idx}/{len(unlabeled)} 张未知样本')
        print('图片路径:', sample_path)
        print('提示：如果系统整体分不清，会先让你直接选择真实类别，避免乱问、乱学。')
        if args.preview and not args.no_preview:
            open_image_file(sample_path)

        print('\n当前特征权重:', pretty_group_weights(aw.export(), feature_spec))
        print('\n初始识别结果:')
        results = model.predict(sample_path, aw.export())
        display_results(results)

        if len(results) < 2:
            print('当前类别不足两个，无法进行混淆对比。')
            continue

        gap = top_gap(results)
        spread = score_spread(results)
        sat = saturated_feature_ratio(results)
        print(f'\n第一名和第二名分数差距: {gap:.3f}')
        print(f'Top 类别分数跨度: {spread:.3f}，特征饱和比例: {sat:.2f}')

        global_uncertain, reason = is_globally_uncertain(results, cfg)
        if global_uncertain:
            print('\n系统判断：整体不确定。')
            print('原因:', reason)
            print('现在不继续问 top1/top2 的差异问题，先请你确认真实类别，避免把错误问题建立在错误候选上。')
            mode, label = ask_true_label(objects, allow_new=True, allow_reject=True)
            pool_info = handle_sample_decision(mode, label, sample_path, model, pool, objects, dataset_dir, pool_enabled, move_after_decision)
            logs.append({
                'sample': sample_path,
                'before': results,
                'global_uncertain': True,
                'uncertainty_reason': reason,
                'asked': False,
                'pool': pool_info,
                'weights_after': aw.export()
            })
            continue

        if gap > ask_threshold:
            print('\n候选差距较大，系统暂不主动提问。')
            predicted_label = results[0]['label']
            if not require_confirm and results[0]['score'] >= auto_accept_threshold:
                print(f'已达到自动确认阈值 {auto_accept_threshold:.2f}，按配置自动加入 confirmed。')
                decision, label = 'confirmed', predicted_label
            else:
                decision, label = confirm_prediction(predicted_label, objects)
            pool_info = handle_sample_decision(decision, label, sample_path, model, pool, objects, dataset_dir, pool_enabled, move_after_decision)
            correction_info = None
            if decision in ('class', 'new') and label and label != predicted_label:
                correction_sample_path = pool_info.get('saved_to') or sample_path
                correction_info = ask_correction_reason(predicted_label, label, pairwise, aw, feature_spec, sample_path=correction_sample_path)
            logs.append({
                'sample': sample_path,
                'before': results,
                'global_uncertain': False,
                'asked': False,
                'pool': pool_info,
                'correction_reason': correction_info,
                'weights_after': aw.export()
            })
            continue

        print('\n候选差距较小，系统不确定，进入主动询问。')
        q = q_selector.select(results[0], results[1], weights=summarize_group_weights(aw.export(), feature_spec))
        generated = generate_natural_question(results[0], results[1], q, summarize_group_weights(aw.export(), feature_spec), sample_path, pairwise_manager=pairwise)

        print('\n系统分析:')
        print(generated['evidence'])
        if generated.get('concept_evidence'):
            print(generated['concept_evidence'])
        print('\n问题:')
        print(generated['question'])
        for key, opt_text, _ in q['options']:
            print(f'{key}. ' + opt_text.format(a=results[0]['label'], b=results[1]['label']))

        ans = input('请输入选项，直接回车表示不确定: ').strip().upper()
        if not ans:
            ans = q['options'][-1][0]

        answer_text, before, after = apply_answer_to_weights(
            aw,
            q,
            ans,
            feature_expander=lambda keys: expand_feature_adjustments(keys, feature_spec),
        )
        if answer_text is None:
            print('无效选项，跳过本次问题更新。')
            logs.append({'sample': sample_path, 'before': results, 'asked': True, 'question': q['id'], 'answer': ans, 'valid_answer': False})
            continue

        print('\n用户回答:', answer_text.format(a=results[0]['label'], b=results[1]['label']))
        print('权重更新前:', pretty_group_weights(before, feature_spec))
        print('权重更新后:', pretty_group_weights(after, feature_spec))

        print('\n重新识别结果:')
        new_results = model.predict(sample_path, aw.export())
        display_results(new_results)

        predicted_label = new_results[0]['label']
        decision, label = confirm_prediction(predicted_label, objects)
        pool_info = handle_sample_decision(decision, label, sample_path, model, pool, objects, dataset_dir, pool_enabled, move_after_decision)
        correction_info = None
        if decision in ('class', 'new') and label and label != predicted_label:
            correction_sample_path = pool_info.get('saved_to') or sample_path
            correction_info = ask_correction_reason(predicted_label, label, pairwise, aw, feature_spec, sample_path=correction_sample_path)
        helpful = False
        if pool_info.get('decision') == 'confirmed':
            old_gap = results[0]['score'] - results[1]['score']
            new_gap = new_results[0]['score'] - new_results[1]['score']
            helpful = new_gap >= old_gap and pool_info.get('label') == predicted_label

        if enable_question_reward:
            old_qw, new_qw = update_question_reward(q_selector.question_weights, q['id'], helpful)
        else:
            old_qw = new_qw = q_selector.question_weights.get(q['id'], 1.0)
        pairwise.record_question_result(results[0]['label'], results[1]['label'], q['id'], helpful)
        if enable_question_reward:
            print(f'\n问题权重更新: {q["id"]}: {old_qw:.2f} -> {new_qw:.2f}')
        else:
            print(f'\n问题权重保持不变: {q["id"]}: {old_qw:.2f}')

        logs.append({
            'sample': sample_path,
            'before': results,
            'gap_before': gap,
            'global_uncertain': False,
            'asked': True,
            'question': q['id'],
            'generated_question': generated,
            'answer': ans,
            'answer_text': answer_text.format(a=results[0]['label'], b=results[1]['label']),
            'weights_before': before,
            'weights_after': aw.export(),
            'after': new_results,
            'pool': pool_info,
            'correction_reason': correction_info,
            'helpful': helpful,
            'question_state': q_selector.export(),
        })

    final_weights = aw.export()
    question_state = q_selector.export()
    pairwise_state = pairwise.export()
    report = make_experience_report(cfg, logs, final_weights, question_state, objects, feature_spec, pairwise_state=pairwise_state)
    experience_summary = make_experience_summary(pairwise_state, objects)
    class_understanding_summary = make_class_understanding_summary(model, objects, pairwise_state=pairwise_state)
    class_understanding_markdown = render_class_understanding_markdown(class_understanding_summary)

    save_json(output_dir / 'feature_weights.json', summarize_group_weights(final_weights, feature_spec))
    save_json(output_dir / 'internal_feature_weights.json', final_weights)
    save_json(output_dir / 'question_weights.json', question_state)
    save_json(output_dir / 'prototype_model.json', model.export())
    save_json(output_dir / 'logs' / 'demo_log.json', logs)
    save_json(output_dir / 'experience_report.json', report)
    save_json(output_dir / 'objects_runtime.json', {'objects': objects})
    save_json(output_dir / 'pairwise_experience_runtime.json', pairwise_state)
    save_json(output_dir / 'experience_summary.json', experience_summary)
    save_json(pool.metadata_dir / 'experience_summary.json', experience_summary)
    save_json(output_dir / 'class_understanding_summary.json', class_understanding_summary)
    (output_dir / 'class_understanding_summary.md').write_text(class_understanding_markdown, encoding='utf-8')
    save_json(pool.metadata_dir / 'class_understanding_summary.json', class_understanding_summary)

    print_header('演示结束')
    print('结果已保存到:', output_dir)
    print('特征权重:', output_dir / 'feature_weights.json')
    print('内部特征权重:', output_dir / 'internal_feature_weights.json')
    print('问题权重:', output_dir / 'question_weights.json')
    print('学习日志:', output_dir / 'logs' / 'demo_log.json')
    print('经验报告:', output_dir / 'experience_report.json')
    print('运行时类别:', output_dir / 'objects_runtime.json')
    print('类别对经验:', output_dir / 'pairwise_experience_runtime.json')
    print('自我总结:', output_dir / 'experience_summary.json')
    print('类别理解总结:', output_dir / 'class_understanding_summary.json')
    print('类别理解文本:', output_dir / 'class_understanding_summary.md')
    if project_root:
        print('样本池:', Path(project_root) / 'sample_pools')
        print('元数据:', Path(project_root) / 'metadata')


if __name__ == '__main__':
    main()
