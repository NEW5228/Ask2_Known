# Ask2Know / a2k_v0.3.2

Ask2Know，简称 a2k，是一个面向个人和小团队的低样本主动教学训练框架。

v0.3.2 的重点不是大幅提升识别准确率，而是让任务数据可以持续积累：用户少量种子样本启动任务，之后不断把新图片放进 `unlabeled`，系统识别、询问、确认后，会自动把样本归档到长期训练库里。下一轮运行时，confirmed 样本会继续参与学习。

## 这版解决什么问题

- 不再每轮都像重新开始。
- confirmed 图片会进入 `datasets/train/<class>/`，并自动命名为 `class_001.jpg`、`class_002.jpg`。
- `unlabeled` 是临时入口，处理过的图片会被移走，下一轮只处理新图片。
- 用户放入 `unlabeled` 的图片可以是任意名字，系统会自动规范为 `img_001.jpg`、`img_002.jpg`。
- 不再要求用户手动写 `task_config.yaml`、`objects.json`、`concepts.json`。

## 安装依赖

```bat
conda activate ask2know
cd D:\ask2know_framework\a2k_v0.3.2
pip install -r requirements.txt
```

## 创建一个水果任务

例如你要训练五类水果：

```bat
python scripts\init_task.py --name fruit_task --classes apple banana pear grape orange --output D:\a2k_test
```

系统会生成：

```text
D:\a2k_test\fruit_task\
├─ configs\task_config.yaml
├─ datasets\
│  ├─ objects.json
│  ├─ concepts.json
│  ├─ train\
│  │  ├─ apple\
│  │  ├─ banana\
│  │  ├─ pear\
│  │  ├─ grape\
│  │  └─ orange\
│  └─ unlabeled\
├─ metadata\
│  ├─ dataset_index.json
│  ├─ sample_history.jsonl
│  └─ unlabeled_import_map.jsonl
├─ sample_pools\
│  ├─ candidate\
│  ├─ rejected\
│  └─ unknown\
└─ outputs\
```

## 放图片

初始已知样本放进：

```text
D:\a2k_test\fruit_task\datasets\train\apple
D:\a2k_test\fruit_task\datasets\train\banana
D:\a2k_test\fruit_task\datasets\train\pear
D:\a2k_test\fruit_task\datasets\train\grape
D:\a2k_test\fruit_task\datasets\train\orange
```

未标注图片放进：

```text
D:\a2k_test\fruit_task\datasets\unlabeled
```

`unlabeled` 里的图片可以叫任何名字，运行时会自动重命名成 `img_001.jpg` 这种格式。

## 运行

```bat
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml
```

不想自动弹图：

```bat
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml --no-preview
```

## 运行后图片会去哪

- 你确认正确 / 纠正类别后：进入 `datasets/train/<class>/<class>_序号.jpg`
- 暂时不确定：进入 `sample_pools/candidate/<class>/`
- 不适合学习：进入 `sample_pools/rejected/`
- 新类别但暂不建立：后续会进入 `sample_pools/unknown/`；当前版本主要支持输入新类别名后直接建立类别并确认。
- 跳过：仍留在 `unlabeled`，下一轮还会处理。

## 重要说明

v0.3.2 的主要目标是“数据持续累积”，不是让模型突然变得很强。当前特征仍是 color / size / contour / texture，识别能力有限。后续需要继续加入数据增强、主体裁剪、预训练视觉特征等能力。
