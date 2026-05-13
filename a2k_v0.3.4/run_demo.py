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
from ask2know.questions.question_generator import generate_natural_question
from ask2know.learning.weights import AdaptiveWeights
from ask2know.learning.feedback_updater import apply_answer_to_weights, update_question_reward
from ask2know.sample_pool.manager import SamplePoolManager
from ask2know.experience.pairwise import PairwiseExperienceManager
from ask2know.experience.summary import ExperienceSummarizer

VERSION = '0.3.4'


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
        detail = ', '.join(f'{k}:{v:.2f}' for k, v in r['detail'].items())
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
        'description': 'added during v0.3.4 interactive learning'
    })
    return objects


def handle_sample_decision(decision, label, sample_path, model, pool, objects, dataset_dir):
    if decision in ('confirmed', 'class', 'new') and label:
        objects = add_new_object_if_needed(objects, label)
        save_objects_file(dataset_dir, objects)
        saved_path = pool.add_confirmed(sample_path, label)
        model.add_confirmed_sample(label, saved_path)
        print('已加入长期训练库 confirmed/train:', label)
        print('保存为:', saved_path)
        return {'decision': 'confirmed', 'label': label, 'saved_to': saved_path}

    if decision == 'candidate' and label:
        saved_path = pool.add_candidate(sample_path, label)
        print('已加入 candidate，不进入正式学习:', label)
        print('保存为:', saved_path)
        return {'decision': 'candidate', 'label': label, 'saved_to': saved_path}

    if decision == 'reject':
        saved_path = pool.add_rejected(sample_path, 'rejected')
        print('已加入 rejected。')
        print('保存为:', saved_path)
        return {'decision': 'rejected', 'label': None, 'saved_to': saved_path}

    if decision == 'unknown':
        saved_path = pool.add_unknown(sample_path)
        print('已加入 unknown。')
        print('保存为:', saved_path)
        return {'decision': 'unknown', 'label': None, 'saved_to': saved_path}

    print('已跳过，文件仍保留在 unlabeled。')
    return {'decision': 'skip', 'label': None, 'saved_to': None}




def ask_correction_reason(predicted_label, true_label, pairwise_manager, adaptive_weights):
    """Ask why a wrong prediction happened and store pairwise experience."""
    if not predicted_label or not true_label or predicted_label == true_label:
        return None

    print('\n错误后追问：')
    print(f'系统刚才把 {true_label} 识别成了 {predicted_label}。为了避免下次再犯同类错误，请你选择主要原因。')
    options = pairwise_manager.correction_options(predicted_label, true_label)
    for key, reason_id, text, _ in options:
        print(f'{key}. {text}')
    ans = input('请输入选项，直接回车表示不确定: ').strip().upper()
    if not ans:
        ans = 'E'

    selected = None
    for item in options:
        if item[0].upper() == ans:
            selected = item
            break
    if selected is None:
        print('输入无法识别，本次不记录错因。')
        return None

    key, reason_id, reason_text, action = selected
    before, after = adaptive_weights.update(action.get('increase', []), action.get('decrease', []))
    pair = pairwise_manager.record_correction(predicted_label, true_label, reason_id, reason_text)

    print('已记录类别对经验:', f'{predicted_label} vs {true_label}')
    print('错因:', reason_text)
    print('权重更新前:', pretty_weights(before))
    print('权重更新后:', pretty_weights(after))

    return {
        'predicted': predicted_label,
        'true_label': true_label,
        'reason_id': reason_id,
        'reason_text': reason_text,
        'weights_before': before,
        'weights_after': after,
        'pairwise': pair,
    }

def make_experience_report(cfg, logs, final_weights, question_state, objects, pairwise_state=None):
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
        'final_feature_weights': final_weights,
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
                'reason_id': item.get('correction_reason', {}).get('reason_id'),
                'reason_text': item.get('correction_reason', {}).get('reason_text'),
            })
        elif item.get('asked') and item.get('answer_text'):
            report['learned_notes'].append({
                'sample': item.get('sample'),
                'type': 'question_feedback',
                'question': item.get('question'),
                'answer_text': item.get('answer_text'),
                'helpful': item.get('helpful'),
                'weights_after': item.get('weights_after')
            })

    low_weight = [k for k, v in final_weights.items() if v < 0.12]
    high_weight = [k for k, v in final_weights.items() if v > 0.30]
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


def main():
    parser = argparse.ArgumentParser(description='Ask2Know low-sample active teaching demo')
    parser.add_argument('--config', default='configs/fruit_demo.yaml')
    parser.add_argument('--preview', action='store_true', help='打开图片预览。默认关闭，避免 Windows 文件占用导致卡死。')
    parser.add_argument('--no-preview', action='store_true', help='兼容旧参数：不自动打开图片预览')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    task = cfg.get('task', {})
    dataset_dir = cfg['paths']['dataset_dir']
    output_dir = Path(cfg['paths']['output_dir'])
    project_root = cfg.get('paths', {}).get('project_root')
    ensure_dir(output_dir)
    ensure_dir(output_dir / 'logs')

    feature_names = [k for k, enabled in cfg.get('features', {}).items() if enabled]
    loader = DatasetLoader(dataset_dir)
    objects = loader.load_objects()
    concepts = loader.load_concepts()

    pool = SamplePoolManager(project_root=project_root, output_dir=output_dir, dataset_dir=dataset_dir)
    pairwise = PairwiseExperienceManager(metadata_dir=pool.metadata_dir)
    summarizer = ExperienceSummarizer(metadata_dir=pool.metadata_dir)
    pool.ensure_for_classes(class_names(objects))

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


    print_header(f'Ask2Know v{VERSION} - 精细提问与自我总结版')
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
    print('启用特征:', ', '.join(feature_names))

    aw = AdaptiveWeights(
        cfg['learning']['initial_weights'],
        cfg['learning'].get('update_step', 0.07),
        cfg['learning'].get('min_weight', 0.05),
        cfg['learning'].get('max_weight', 0.70)
    )
    aw.apply_concepts(concepts)

    model = PrototypeModel(feature_names, augmentation_config=cfg.get('augmentation', {})).fit(train_samples)
    q_selector = QuestionSelector(pairwise_manager=pairwise)
    logs = []
    ask_threshold = cfg['confidence'].get('ask_user_threshold', 0.12)

    for idx, sample in enumerate(unlabeled, 1):
        sample_path = sample['path']
        print_header(f'正在识别第 {idx}/{len(unlabeled)} 张未知样本')
        print('图片路径:', sample_path)
        print('提示：如果系统整体分不清，会先让你直接选择真实类别，避免乱问、乱学。')
        if args.preview and not args.no_preview:
            open_image_file(sample_path)

        print('\n当前特征权重:', pretty_weights(aw.export()))
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
            pool_info = handle_sample_decision(mode, label, sample_path, model, pool, objects, dataset_dir)
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
            decision, label = confirm_prediction(predicted_label, objects)
            pool_info = handle_sample_decision(decision, label, sample_path, model, pool, objects, dataset_dir)
            correction_info = None
            if decision in ('class', 'new') and label and label != predicted_label:
                correction_info = ask_correction_reason(predicted_label, label, pairwise, aw)
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
        q = q_selector.select(results[0], results[1])
        generated = generate_natural_question(results[0], results[1], q, aw.export(), sample_path, pairwise_manager=pairwise)

        print('\n系统分析:')
        print(generated['evidence'])
        print('\n问题:')
        print(generated['question'])
        for key, opt_text, _ in q['options']:
            print(f'{key}. ' + opt_text.format(a=results[0]['label'], b=results[1]['label']))

        ans = input('请输入选项，直接回车表示不确定: ').strip().upper()
        if not ans:
            ans = q['options'][-1][0]

        answer_text, before, after = apply_answer_to_weights(aw, q, ans)
        if answer_text is None:
            print('无效选项，跳过本次问题更新。')
            logs.append({'sample': sample_path, 'before': results, 'asked': True, 'question': q['id'], 'answer': ans, 'valid_answer': False})
            continue

        print('\n用户回答:', answer_text.format(a=results[0]['label'], b=results[1]['label']))
        print('权重更新前:', pretty_weights(before))
        print('权重更新后:', pretty_weights(after))

        print('\n重新识别结果:')
        new_results = model.predict(sample_path, aw.export())
        display_results(new_results)

        predicted_label = new_results[0]['label']
        decision, label = confirm_prediction(predicted_label, objects)
        pool_info = handle_sample_decision(decision, label, sample_path, model, pool, objects, dataset_dir)
        correction_info = None
        if decision in ('class', 'new') and label and label != predicted_label:
            correction_info = ask_correction_reason(predicted_label, label, pairwise, aw)
        helpful = False
        if pool_info.get('decision') == 'confirmed':
            old_gap = results[0]['score'] - results[1]['score']
            new_gap = new_results[0]['score'] - new_results[1]['score']
            helpful = new_gap >= old_gap and pool_info.get('label') == predicted_label

        old_qw, new_qw = update_question_reward(q_selector.question_weights, q['id'], helpful)
        pairwise.record_question_result(results[0]['label'], results[1]['label'], q['id'], helpful)
        print(f'\n问题权重更新: {q["id"]}: {old_qw:.2f} -> {new_qw:.2f}')

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
    summary_state = summarizer.update_from_project(objects=objects, pairwise_state=pairwise_state, model=model.export(), logs=logs)
    report = make_experience_report(cfg, logs, final_weights, question_state, objects, pairwise_state=pairwise_state)

    save_json(output_dir / 'feature_weights.json', final_weights)
    save_json(output_dir / 'question_weights.json', question_state)
    save_json(output_dir / 'prototype_model.json', model.export())
    save_json(output_dir / 'logs' / 'demo_log.json', logs)
    save_json(output_dir / 'experience_report.json', report)
    save_json(output_dir / 'objects_runtime.json', {'objects': objects})
    save_json(output_dir / 'pairwise_experience_runtime.json', pairwise_state)
    save_json(output_dir / 'experience_summary_runtime.json', summary_state)

    print_header('演示结束')
    print('结果已保存到:', output_dir)
    print('特征权重:', output_dir / 'feature_weights.json')
    print('问题权重:', output_dir / 'question_weights.json')
    print('学习日志:', output_dir / 'logs' / 'demo_log.json')
    print('经验报告:', output_dir / 'experience_report.json')
    print('运行时类别:', output_dir / 'objects_runtime.json')
    print('类别对经验:', output_dir / 'pairwise_experience_runtime.json')
    print('自我总结:', output_dir / 'experience_summary_runtime.json')
    if project_root:
        print('样本池:', Path(project_root) / 'sample_pools')
        print('元数据:', Path(project_root) / 'metadata')


if __name__ == '__main__':
    main()
