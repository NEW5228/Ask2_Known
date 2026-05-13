import argparse
import os
import platform
import subprocess
from pathlib import Path

from ask2know.utils.io_utils import load_yaml, save_json, ensure_dir
from ask2know.data.dataset_loader import DatasetLoader
from ask2know.inference.prototype_model import PrototypeModel
from ask2know.questions.question_bank import QUESTION_BANK
from ask2know.learning.weights import AdaptiveWeights
from ask2know.learning.feedback_updater import apply_answer_to_weights, update_question_reward


def open_image_file(image_path):
    image_path = str(image_path)

    try:
        system = platform.system()

        if system == "Windows":
            os.startfile(image_path)
        elif system == "Darwin":
            subprocess.Popen(["open", image_path])
        else:
            subprocess.Popen(["xdg-open", image_path])
    except Exception:
        pass


def display_results(results, max_items=3):
    for i, r in enumerate(results[:max_items], 1):
        detail = ", ".join(f"{k}:{v:.2f}" for k, v in r["detail"].items())
        print(f"{i}. {r['label']}: {r['score']:.3f}  ({detail})")


def select_question(top_a, top_b, question_weights, last_question_id, ask_counts):
    detail_a = top_a.get("detail", {})
    detail_b = top_b.get("detail", {})

    candidates = []

    for q in QUESTION_BANK:
        question_id = q["id"]
        feature = q["feature"]

        score_a = detail_a.get(feature, 0.0)
        score_b = detail_b.get(feature, 0.0)

        feature_gap = abs(score_a - score_b)

        uncertainty_score = 1.0 - min(feature_gap, 1.0)
        difference_score = min(feature_gap, 1.0)

        history_weight = question_weights.get(question_id, 1.0)

        repeat_penalty = 0.55 if question_id == last_question_id else 0.0
        count_penalty = min(ask_counts.get(question_id, 0) * 0.08, 0.40)

        final_score = (
            0.35 * uncertainty_score
            + 0.35 * difference_score
            + 0.30 * history_weight
            - repeat_penalty
            - count_penalty
        )

        candidates.append((final_score, q))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def print_header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def normalize_weights_for_print(weights):
    return {k: round(v, 3) for k, v in weights.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fruit_demo.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    dataset_dir = cfg["paths"]["dataset_dir"]
    output_dir = Path(cfg["paths"]["output_dir"])

    ensure_dir(output_dir)
    ensure_dir(output_dir / "logs")

    feature_names = [k for k, enabled in cfg.get("features", {}).items() if enabled]

    loader = DatasetLoader(dataset_dir)
    objects = loader.load_objects()
    concepts = loader.load_concepts()
    train_samples = loader.load_train_samples()
    unlabeled = loader.load_unlabeled_samples()

    if not train_samples:
        print("没有找到训练样本。")
        print("请先把图片放到 datasets/fruit_demo/train/apple 和 train/strawberry")
        print("或者运行：python scripts/create_demo_dataset.py")
        return

    if not unlabeled:
        print("没有找到待识别样本。")
        print("请把图片放到 datasets/fruit_demo/unlabeled")
        print("或者运行：python scripts/create_demo_dataset.py")
        return

    print_header("Ask2Know v0.1 Demo")
    print("对象类别:", ", ".join([o["name"] for o in objects]))
    print(f"训练样本数: {len(train_samples)}")
    print(f"待识别样本数: {len(unlabeled)}")
    print("启用特征:", ", ".join(feature_names))

    aw = AdaptiveWeights(
        cfg["learning"]["initial_weights"],
        cfg["learning"].get("update_step", 0.07),
        cfg["learning"].get("min_weight", 0.05),
        cfg["learning"].get("max_weight", 0.70),
    )

    aw.apply_concepts(concepts)

    model = PrototypeModel(feature_names).fit(train_samples)

    question_weights = {q["id"]: 1.0 for q in QUESTION_BANK}
    ask_counts = {q["id"]: 0 for q in QUESTION_BANK}
    last_question_id = None

    logs = []

    ask_threshold = cfg["confidence"].get("ask_user_threshold", 0.12)

    for idx, sample in enumerate(unlabeled, 1):
        sample_path = sample["path"]

        print_header(f"正在识别第 {idx}/{len(unlabeled)} 张未知样本")
        print("图片路径:", sample_path)
        print("提示：如果你看不清这张图，或者不确定它是什么，最后确认时输入 skip。")

        open_image_file(sample_path)

        print("\n当前特征权重:")
        print(normalize_weights_for_print(aw.export()))

        print("\n初始识别结果:")
        results = model.predict(sample_path, aw.export())
        display_results(results)

        if len(results) < 2:
            print("候选对象不足，跳过主动提问。")
            continue

        gap = results[0]["score"] - results[1]["score"]
        print(f"\n第一名和第二名分数差距: {gap:.3f}")

        if gap > ask_threshold:
            print("候选差距较大，系统暂不主动提问。")

            confirm = input(
                f"是否确认该样本为 {results[0]['label']} ? (y/n/skip): "
            ).strip().lower()

            if confirm == "y":
                model.add_confirmed_sample(results[0]["label"], sample_path)
                print("已加入正式样本库:", results[0]["label"])
            elif confirm == "n":
                correct = input("请输入正确类别名，或回车跳过: ").strip()
                if correct:
                    model.add_confirmed_sample(correct, sample_path)
                    print("已按纠正类别加入正式样本库:", correct)
            else:
                print("已跳过，不加入正式样本库。")

            logs.append(
                {
                    "sample": sample_path,
                    "before": results,
                    "asked": False,
                    "confirmed": confirm,
                    "weights_after": aw.export(),
                }
            )
            continue

        print("候选差距较小，系统不确定，进入主动询问。")

        q = select_question(
            results[0],
            results[1],
            question_weights,
            last_question_id,
            ask_counts,
        )

        last_question_id = q["id"]
        ask_counts[q["id"]] = ask_counts.get(q["id"], 0) + 1

        question_text = q["template"].format(
            a=results[0]["label"],
            b=results[1]["label"],
        )

        print("\n问题:", question_text)
        for key, opt_text, _ in q["options"]:
            print(f"{key}. " + opt_text.format(a=results[0]["label"], b=results[1]["label"]))

        ans = input("请输入选项，直接回车表示不确定: ").strip().upper()

        if not ans:
            last_option = q["options"][-1][0]
            ans = last_option

        answer_text, before_weights, after_weights = apply_answer_to_weights(aw, q, ans)

        if answer_text is None:
            print("无效选项，跳过本次问题更新。")
            logs.append(
                {
                    "sample": sample_path,
                    "before": results,
                    "asked": True,
                    "question": q["id"],
                    "answer": ans,
                    "valid_answer": False,
                    "weights_after": aw.export(),
                }
            )
            continue

        print("\n用户回答:", answer_text.format(a=results[0]["label"], b=results[1]["label"]))
        print("权重更新前:", normalize_weights_for_print(before_weights))
        print("权重更新后:", normalize_weights_for_print(after_weights))

        print("\n重新识别结果:")
        new_results = model.predict(sample_path, aw.export())
        display_results(new_results)

        confirm = input(
            f"是否确认该样本为 {new_results[0]['label']} ? (y/n/skip): "
        ).strip().lower()

        helpful = False

        if confirm == "y":
            model.add_confirmed_sample(new_results[0]["label"], sample_path)
            print("已加入正式样本库:", new_results[0]["label"])

            old_gap = results[0]["score"] - results[1]["score"]
            new_gap = new_results[0]["score"] - new_results[1]["score"]
            helpful = new_gap >= old_gap

        elif confirm == "n":
            correct = input("请输入正确类别名，或回车跳过: ").strip()

            if correct:
                model.add_confirmed_sample(correct, sample_path)
                print("已按纠正类别加入正式样本库:", correct)

            helpful = False

        else:
            print("已跳过，不加入正式样本库。")
            helpful = False

        old_qw, new_qw = update_question_reward(question_weights, q["id"], helpful)

        print(f"\n问题权重更新: {q['id']}: {old_qw:.2f} -> {new_qw:.2f}")
        print("当前问题被询问次数:", ask_counts[q["id"]])

        logs.append(
            {
                "sample": sample_path,
                "before": results,
                "gap_before": gap,
                "asked": True,
                "question": q["id"],
                "question_text": question_text,
                "answer": ans,
                "answer_text": answer_text.format(
                    a=results[0]["label"],
                    b=results[1]["label"],
                ),
                "weights_before": before_weights,
                "weights_after": aw.export(),
                "after": new_results,
                "confirmed": confirm,
                "helpful": helpful,
                "question_weights": dict(question_weights),
                "ask_counts": dict(ask_counts),
            }
        )

    save_json(output_dir / "feature_weights.json", aw.export())
    save_json(
        output_dir / "question_weights.json",
        {
            "question_weights": dict(question_weights),
            "ask_counts": dict(ask_counts),
            "last_question_id": last_question_id,
        },
    )
    save_json(output_dir / "prototype_model.json", model.export())
    save_json(output_dir / "logs" / "demo_log.json", logs)

    print_header("演示结束")
    print("结果已保存到:", output_dir)
    print("特征权重:", output_dir / "feature_weights.json")
    print("问题权重:", output_dir / "question_weights.json")
    print("学习日志:", output_dir / "logs" / "demo_log.json")


if __name__ == "__main__":
    main()