# Ask2Know / a2k_v0.3.3

Ask2Know，简称 a2k，是一个面向个人和小团队的低样本主动教学训练框架。

v0.3.3 在 v0.3.2 的“持续学习与数据集累积”基础上，新增 **错误后追问** 和 **类别对经验**。这版不是为了立刻把识别准确率拉满，而是让系统在识别错时不只保存正确类别，还继续追问“为什么错”，并把错因保存下来，减少后续问题循环。

## 这版解决什么问题

- 保留 v0.3.2 的持续学习能力：confirmed 样本进入 `datasets/train/<class>/`，下一轮继续参与学习。
- `unlabeled` 仍然是临时入口：处理过的图片会被移走，下一轮只处理新图片。
- 未标注图片仍会自动重命名为 `img_001.jpg`、`img_002.jpg`。
- 新增错误后追问：系统预测错后，会问用户主要错因。
- 新增类别对经验：例如 `apple vs grape`、`apple vs pear`，记录哪些特征有用、哪些特征容易误导。
- 问题选择开始参考历史类别对经验，减少重复问没用的问题。

## 安装依赖

```bat
conda activate ask2know
cd D:\ask2know_framework\a2k_v0.3.3
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
│  ├─ unlabeled_import_map.jsonl
│  └─ pairwise_experience.json
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

`unlabeled` 里的图片可以叫任何名字，运行时会自动规范成 `img_001.jpg` 这种格式。

## 运行

```bat
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml
```

不想自动弹图：

```bat
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml --no-preview
```

## 错误后追问

如果系统预测 `apple`，你纠正为 `grape`，系统会继续问：

```text
系统刚才把 grape 识别成了 apple。为了避免下次再犯同类错误，请你选择主要原因。
A. 颜色或色系差异明显
B. 整体形状、结构或轮廓不同
C. 表面纹理、颗粒感或局部重复结构不同
D. 背景、光线、遮挡或主体不清楚影响了判断
E. 不确定 / 其他原因
```

回答后，系统会记录到：

```text
D:\a2k_test\fruit_task\metadata\pairwise_experience.json
```

下次遇到同样类别对混淆时，系统会优先参考这份经验。

## 运行后图片会去哪

- 你确认正确 / 纠正类别后：进入 `datasets/train/<class>/<class>_序号.jpg`
- 暂时不确定：进入 `sample_pools/candidate/<class>/`
- 不适合学习：进入 `sample_pools/rejected/`
- 新类别但暂不建立：进入 `sample_pools/unknown/`
- 跳过：仍留在 `unlabeled`，下一轮还会处理。

## 重要说明

v0.3.3 的主要目标是：**让系统从错误中学到“为什么错”，而不是只保存正确标签。**

当前特征仍是 color / size / contour / texture，识别能力有限。后续仍需要数据增强、主体裁剪、预训练视觉特征等能力。
