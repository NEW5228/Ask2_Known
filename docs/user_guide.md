# Ask2Know User Guide

Ask2Know 用于低样本图像识别任务。用户先提供少量已知样本，系统识别未知图片；当候选类别接近或整体不确定时，系统会询问用户，并把确认、纠错、错因和样本流转记录下来。

## 安装

```bat
conda activate ask2know
cd D:\ask2know_framework
pip install -r requirements.txt
```

## 创建任务

```bat
python scripts\init_task.py --name fruit_task --classes apple banana pear grape orange --output D:\a2k_test
```

创建任务时可以选择特征预设和用户可见特征：

```bat
python scripts\init_task.py --name fruit_task --classes apple banana pear --output D:\a2k_test --feature-preset fruit --features color shape texture size
```

用户可见特征为 `color`、`shape`、`texture`、`size`。系统会内部启用 `quality` 做样本质量检查，但不会把它当作用户选择过的训练特征显示。

任务创建后，已知样本放入：

```text
D:\a2k_test\fruit_task\datasets\train\类别名
```

待识别图片放入：

```text
D:\a2k_test\fruit_task\datasets\unlabeled
```

## 运行

```bat
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml
```

默认不弹图。需要预览图片时使用：

```bat
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml --preview
```

## 添加类别

不要重新创建项目。要向已有任务加入新类别：

```bat
python scripts\add_class.py --project D:\a2k_test\fruit_task --class cherry
```

然后把该类别训练图片放入：

```text
D:\a2k_test\fruit_task\datasets\train\cherry
```

## 持续学习流程

第一次运行时，每类建议至少放 3 张已知样本。第二次以后，继续把新图片放入 `datasets/unlabeled/`，系统会读取已经积累在 `datasets/train/` 中的 confirmed 样本，并把新的确认样本继续加入长期训练库。

确认识别结果时：

- `Y`：正确，加入 confirmed 训练库。
- `N`：错误，选择正确类别。
- `C`：暂时放入 candidate，不进入正式学习。
- `R`：拒绝样本。
- `S`：跳过，文件仍保留在 unlabeled。

如果系统整体分不清，会先让用户选择真实类别，避免基于错误候选继续提问。

## 命名和样本池

- `datasets/unlabeled/` 中的图片可以任意命名，运行时会整理为 `img_001.jpg`。
- `datasets/train/<class>/` 中的图片可以任意命名，运行时会整理为 `class_001.jpg`。
- confirmed 样本进入 `datasets/train/<class>/`。
- candidate、rejected、unknown 样本进入 `sample_pools/`。
- 样本流转记录写入 `metadata/sample_history.jsonl`。

## 错误后追问

如果系统预测错误，用户选择正确类别后，系统会继续询问主要错因。可以多选，例如：

```text
A,B,D
```

或：

```text
ABD
```

错因会记录到 `metadata/pairwise_experience.json`，并影响后续同类混淆时的问题选择和特征权重。

## 输出文件

运行结束后会写入：

- `outputs/feature_weights.json`
- `outputs/question_weights.json`
- `outputs/prototype_model.json`
- `outputs/logs/demo_log.json`
- `outputs/experience_report.json`
- `outputs/experience_summary.json`
- `metadata/experience_summary.json`

`experience_summary.json` 是系统根据纠错记录生成的弱总结，不代表最终真理。

## 数据增强

默认启用温和增强，用于少样本原型建立：亮度、轻微旋转、轻微裁剪。可以在任务配置中关闭：

```yaml
augmentation:
  enable: false
```
# Current Dataset Layout

- `datasets/train/<class>/`: confirmed training samples and long-term memory.
- `datasets/unknown/`: learning samples that Ask2Know should recognize, ask about, cluster, or promote into `train`.
- `datasets/unlabeled/<class>/`: labeled evaluation samples for accuracy checks. These samples are not used for active learning by `run_demo.py`.
- `sample_pools/unknown/`: runtime pool for samples the system could not decide; this is separate from `datasets/unknown/`.
