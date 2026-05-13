import argparse
import os
import platform
import subprocess
from pathlib import Path

from ask2know.utils.io_utils import load_yaml, save_json, ensure_dir
from ask2know.data.dataset_loader import DatasetLoader
from ask2know.inference.prototype_model import PrototypeModel
from ask2know.questions.question_selector import QuestionSelector
from ask2know.questions.question_generator import generate_natural_question
from ask2know.learning.weights import AdaptiveWeights
from ask2know.learning.feedback_updater import apply_answer_to_weights, update_question_reward

VERSION = '0.3.0'


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


def make_experience_report(cfg, logs, final_weights, question_state):
    task = cfg.get('task', {})
    report = {
        'framework': 'Ask2Know',
        'version': VERSION,
        'task': task,
        'summary': {
            'total_samples': len(logs),
            'asked_count': sum(1 for x in logs if x.get('asked')),
            'confirmed_count': sum(1 for x in logs if x.get('confirmed') == 'y'),
            'corrected_count': sum(1 for x in logs if x.get('confirmed') == 'n'),
            'skipped_count': sum(1 for x in logs if x.get('confirmed') not in ('y', 'n')),
        },
        'final_feature_weights': final_weights,
        'question_state': question_state,
        'learned_notes': [],
        'next_suggestions': []
    }

    for item in logs:
        if item.get('asked') and item.get('answer_text'):
            report['learned_notes'].append({
                'sample': item.get('sample'),
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
    if report['summary']['skipped_count'] > 0:
        report['next_suggestions'].append('存在跳过样本，建议检查这些图片是否模糊、主体不明显或类别不在当前任务中。')
    if report['summary']['asked_count'] == 0:
        report['next_suggestions'].append('本轮没有触发主动提问。若希望测试教学机制，可调高 ask_user_threshold。')

    return report


def main():
    parser = argparse.ArgumentParser(description='Ask2Know low-sample active teaching demo')
    parser.add_argument('--config', default='configs/fruit_demo.yaml')
    parser.add_argument('--no-preview', action='store_true', help='不自动打开图片预览')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    task = cfg.get('task', {})
    dataset_dir = cfg['paths']['dataset_dir']
    output_dir = Path(cfg['paths']['output_dir'])
    ensure_dir(output_dir)
    ensure_dir(output_dir / 'logs')

    feature_names = [k for k, enabled in cfg.get('features', {}).items() if enabled]
    loader = DatasetLoader(dataset_dir)
    objects = loader.load_objects()
    concepts = loader.load_concepts()
    train_samples = loader.load_train_samples()
    unlabeled = loader.load_unlabeled_samples()

    if not objects:
        print('没有找到 objects.json 或 objects 为空。')
        print('可以运行：python scripts/init_task.py --name my_task --classes class_a class_b')
        return
    if not train_samples:
        print('没有找到训练样本。')
        print(f'请把每个类别的已知图片放到：{Path(dataset_dir) / "train" / "类别名"}')
        return
    if not unlabeled:
        print('没有找到待识别样本。')
        print(f'请把未知图片放到：{Path(dataset_dir) / "unlabeled"}')
        return

    print_header(f'Ask2Know v{VERSION} - 通用低样本主动教学框架')
    print('任务名:', task.get('name', 'unknown'))
    print('任务类型:', task.get('type', 'image_object_recognition'))
    if task.get('description'):
        print('任务说明:', task.get('description'))
    print('数据目录:', dataset_dir)
    print('输出目录:', output_dir)
    print('对象类别:', ', '.join([o['name'] for o in objects]))
    print(f'训练样本数: {len(train_samples)}，待识别样本数: {len(unlabeled)}')
    print('启用特征:', ', '.join(feature_names))

    aw = AdaptiveWeights(
        cfg['learning']['initial_weights'],
        cfg['learning'].get('update_step', 0.07),
        cfg['learning'].get('min_weight', 0.05),
        cfg['learning'].get('max_weight', 0.70)
    )
    aw.apply_concepts(concepts)

    model = PrototypeModel(feature_names).fit(train_samples)
    q_selector = QuestionSelector()
    logs = []
    ask_threshold = cfg['confidence'].get('ask_user_threshold', 0.12)

    for idx, sample in enumerate(unlabeled, 1):
        sample_path = sample['path']
        print_header(f'正在识别第 {idx}/{len(unlabeled)} 张未知样本')
        print('图片路径:', sample_path)
        print('提示：如果你不知道当前图是什么，最后确认时输入 skip，避免污染样本库。')
        if not args.no_preview:
            open_image_file(sample_path)

        print('\n当前特征权重:', pretty_weights(aw.export()))
        print('\n初始识别结果:')
        results = model.predict(sample_path, aw.export())
        display_results(results)

        if len(results) < 2:
            print('当前类别不足两个，无法进行混淆对比。')
            continue

        gap = results[0]['score'] - results[1]['score']
        print(f'\n第一名和第二名分数差距: {gap:.3f}')

        if gap > ask_threshold:
            print('候选差距较大，系统暂不主动提问。')
            confirm = input(f'是否确认该样本为 {results[0]["label"]} ? (y/n/skip): ').strip().lower()
            if confirm == 'y':
                model.add_confirmed_sample(results[0]['label'], sample_path)
                print('已加入正式样本库:', results[0]['label'])
            elif confirm == 'n':
                correct = input('请输入正确类别名，或回车跳过: ').strip()
                if correct:
                    model.add_confirmed_sample(correct, sample_path)
                    print('已按纠正类别加入正式样本库:', correct)
            else:
                print('已跳过，不加入正式样本库。')
            logs.append({
                'sample': sample_path,
                'before': results,
                'asked': False,
                'confirmed': confirm,
                'weights_after': aw.export()
            })
            continue

        print('候选差距较小，系统不确定，进入主动询问。')
        q = q_selector.select(results[0], results[1])
        generated = generate_natural_question(results[0], results[1], q, aw.export(), sample_path)

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
            logs.append({
                'sample': sample_path,
                'before': results,
                'asked': True,
                'question': q['id'],
                'answer': ans,
                'valid_answer': False
            })
            continue

        print('\n用户回答:', answer_text.format(a=results[0]['label'], b=results[1]['label']))
        print('权重更新前:', pretty_weights(before))
        print('权重更新后:', pretty_weights(after))

        print('\n重新识别结果:')
        new_results = model.predict(sample_path, aw.export())
        display_results(new_results)

        confirm = input(f'是否确认该样本为 {new_results[0]["label"]} ? (y/n/skip): ').strip().lower()
        helpful = False
        if confirm == 'y':
            model.add_confirmed_sample(new_results[0]['label'], sample_path)
            print('已加入正式样本库:', new_results[0]['label'])
            old_gap = results[0]['score'] - results[1]['score']
            new_gap = new_results[0]['score'] - new_results[1]['score']
            helpful = new_gap >= old_gap
        elif confirm == 'n':
            correct = input('请输入正确类别名，或回车跳过: ').strip()
            if correct:
                model.add_confirmed_sample(correct, sample_path)
                print('已按纠正类别加入正式样本库:', correct)
            helpful = False
        else:
            print('已跳过，不加入正式样本库。')

        old_qw, new_qw = update_question_reward(q_selector.question_weights, q['id'], helpful)
        print(f'\n问题权重更新: {q["id"]}: {old_qw:.2f} -> {new_qw:.2f}')

        logs.append({
            'sample': sample_path,
            'before': results,
            'gap_before': gap,
            'asked': True,
            'question': q['id'],
            'generated_question': generated,
            'answer': ans,
            'answer_text': answer_text.format(a=results[0]['label'], b=results[1]['label']),
            'weights_before': before,
            'weights_after': aw.export(),
            'after': new_results,
            'confirmed': confirm,
            'helpful': helpful,
            'question_state': q_selector.export(),
        })

    final_weights = aw.export()
    question_state = q_selector.export()
    report = make_experience_report(cfg, logs, final_weights, question_state)

    save_json(output_dir / 'feature_weights.json', final_weights)
    save_json(output_dir / 'question_weights.json', question_state)
    save_json(output_dir / 'prototype_model.json', model.export())
    save_json(output_dir / 'logs' / 'demo_log.json', logs)
    save_json(output_dir / 'experience_report.json', report)

    print_header('演示结束')
    print('结果已保存到:', output_dir)
    print('特征权重:', output_dir / 'feature_weights.json')
    print('问题权重:', output_dir / 'question_weights.json')
    print('学习日志:', output_dir / 'logs' / 'demo_log.json')
    print('经验报告:', output_dir / 'experience_report.json')


if __name__ == '__main__':
    main()
