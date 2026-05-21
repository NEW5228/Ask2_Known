# Ask2Know

Ask2Know 是一个面向低样本图像识别任务的主动教学框架。

当前版本：`0.4.5`

核心流程：

```text
图片
-> 浅层可解释视觉特征
-> CLIP image embedding
-> prototype 相似度 + kNN 相似样本证据 + 概念相似度
-> 不确定性判断 / 主动提问
-> 用户确认或纠错
-> 更新权重、原型、样本池和经验记录
```

从 `0.4.1` 开始，CLIP 是必需依赖。系统不会回退到 OpenCV embedding。如果
`torch`、`open_clip_torch` 或配置的 CLIP 权重不可用，运行会直接失败。

## 安装

需要 Python 3.9+。

```bat
pip install -r requirements.txt
pip install -e .
```

默认 CLIP 配置：

```yaml
model_name: ViT-B-32
pretrained: laion2b_s34b_b79k
```

第一次运行时 OpenCLIP 可能需要下载模型权重。如果当前机器没有网络，也没有本地缓存，初始化会失败。这是当前版本的预期行为。

## 创建任务

```bat
python scripts\init_task.py --name fruit_task --classes apple banana pear --output D:\a2k_test
```

指定任务预设和用户可见特征：

```bat
python scripts\init_task.py --name pet_task --classes cat dog --output D:\a2k_test --feature-preset pet --features color shape texture surface part size
```

可用预设：

- `auto`
- `general`
- `fruit`
- `pet`
- `traffic_sign`

用户可选特征：

- `color`
- `shape`
- `texture`
- `surface`
- `part`
- `size`
- `text`
- `sign`

`quality` 是系统内部质量特征，用于判断主体清晰度、背景干扰、模糊和样本是否适合学习，不作为用户可选训练特征展示。

## 放入图片

已知训练图片放到：

```text
D:\a2k_test\<task_name>\datasets\train\<class_name>
```

待学习 unknown 图片放到：

```text
D:\a2k_test\<task_name>\datasets\unknown
```

图片文件名可以任意。运行时会根据配置自动规范化训练集和 unknown 图片文件名。

## v0.4.4 数据目录和推荐流程

v0.4.4 推荐的主流程是：先为每个类别准备少量已知样本，放入 `datasets/train/<class>`，让系统从一个可靠的小训练集开始。`datasets/unknown/` 的自动粗分功能保留为辅助整理工具，不建议直接把粗分结果当成最终训练集。

```text
datasets/train/<class>/       已确认训练样本，推荐主入口
datasets/unknown/             待学习、待整理的混合未知图片
datasets/unlabeled/<class>/   带真实标签的验证集，用于计算准确率
```

`run_demo.py` 会读取 `datasets/unknown/` 作为主动学习样本。`datasets/unlabeled/<class>/` 不进入主动学习，只用于评估。

## v0.4.4 Bootstrap 粗分功能

如果用户手里只有一堆混合图片，可以先放入 `datasets/unknown/`，再用 bootstrap 脚本粗分：

```bat
python scripts\bootstrap_clusters.py --config D:\a2k_test\<task_name>\configs\task_config.yaml
```

如果用户已经知道有哪些类别名，可以传入名字：

```bat
python scripts\bootstrap_clusters.py --config D:\a2k_test\<task_name>\configs\task_config.yaml --names 张三 李四 王五 赵六
```

脚本会读取 `datasets/unknown/`，用 CLIP embedding 聚类，展示每组代表图片和低相似度离群候选，让用户确认每组叫什么。复制前会生成 review 清单，并允许按单张图片跳过；只有 review 计划中保留的图片才会复制到 `datasets/train/<class>/`。原始 `unknown/` 图片会保留。

只想检查聚类和复制计划，不写入训练集时使用：

```bat
python scripts\bootstrap_clusters.py --config D:\a2k_test\<task_name>\configs\task_config.yaml --report-only
```

注意：bootstrap 是粗分辅助工具，不保证每个 cluster 都纯净。正式任务仍建议从人工整理的少量 `train/<class>` 样本开始；bootstrap 结果应由用户复核后再进入训练库。

## 验证准确率

把验证图片按真实类别放入 `datasets/unlabeled/<class>/` 后运行：

```bat
python scripts\evaluate_unlabeled.py --config D:\a2k_test\<task_name>\configs\task_config.yaml
```

评估报告会写入：

```text
outputs/evaluation_report.json
```

报告包含总准确率、每类准确率、混淆情况和错误样本列表。

## 运行

```bat
python run_demo.py --config D:\a2k_test\<task_name>\configs\task_config.yaml
```

需要手动预览图片时：

```bat
python run_demo.py --config D:\a2k_test\<task_name>\configs\task_config.yaml --preview
```

## 关键配置

新任务默认使用 CLIP + hybrid similarity：

```yaml
deep_features:
  enable: true
  provider: open_clip
  model_name: ViT-B-32
  pretrained: laion2b_s34b_b79k
  device: auto
  feature_name: image_embedding
  cache: true
  fallback_to_opencv: false
  include_augmented: false

similarity:
  mode: hybrid
  knn:
    enable: true
    k: 3
    score_weight: 0.20
  sub_prototypes:
    enable: true
    max_centers: 3
    min_samples_per_center: 8
    score_weight: 0.06
    mode: conservative
    min_gain_over_prototype: 0.015
    min_top_gap: 0.0
    allow_rank_flip: true
    max_base_margin_for_flip: 0.010
    rank_flip_prototype_veto_margin: 0.003
  text_semantic:
    enable: true
    score_weight: 0.08
    prompt_templates:
      - "a photo of a {label}"
      - "a close-up photo of a {label}"
  pairwise_rerank:
    enable: true
    local_k: 5
    score_weight: 0.25
    max_score_margin: 0.018
    min_pair_similarity: 0.90
    min_local_gap: 0.008
  robust_prototype:
    enable: true
    deep_only: true
    min_samples: 12
    trim_fraction: 0.12
    report_margin: 0.015
    top_outliers_per_class: 5
  concept_gate:
    enable: true
    min_top_gap: 0.035
    weak_score_weight: 0.00

diagnostics:
  low_margin_threshold: 0.015
  weak_signal_threshold: 0.005

concepts:
  enable: true
  score_weight: 0.05
```

预测结果会显示分数来源，例如：

```text
score sources: proto:0.731, subproto:0.814, knn:0.802, text:0.691, concept:0.744
nearest: 0.817 D:\...\datasets\train\apple\apple_003.jpg
```

含义：

- `proto`：当前图片和类别原型的相似度。
- `knn`：当前图片和最近 confirmed 训练样本的相似度。
- 	ext：当前图片 CLIP embedding 和类别名文本 prompt embedding 的相似度。
- concept：当前图片和类别可解释概念原型的相似度。
- `nearest`：系统认为最相似的已知训练样本。

## 用户反馈

系统不确定时，会询问用户可判断的问题，而不是让用户判断 raw embedding。

典型问题包括：

- 颜色是否可靠
- 形状是否重要
- 纹理或表面特征是否可靠
- 部位结构是否重要
- 文字/标识是否有区分作用
- 当前图片是否适合加入训练

用户确认或纠错后，系统会更新：

- 特征权重
- 类别原型
- CLIP/kNN 证据缓存
- 类别对混淆经验
- 样本池
- 运行日志和总结

## 输出文件

主要输出在任务的 `outputs/` 目录：

```text
outputs/feature_weights.json
outputs/internal_feature_weights.json
outputs/question_weights.json
outputs/prototype_model.json
outputs/logs/demo_log.json
outputs/experience_report.json
outputs/experience_summary.json
outputs/class_understanding_summary.json
outputs/class_understanding_summary.md
```

样本池和元数据在任务项目目录：

```text
metadata/
sample_pools/
```

## 添加类别

已有任务中追加新类别：

```bat
python scripts\add_class.py --project D:\a2k_test\fruit_task --class cherry
```

然后把该类别训练图片放到：

```text
D:\a2k_test\fruit_task\datasets\train\cherry
```

## 说明

- Ask2Know 不是大模型训练框架。
- CLIP 用作必需的图像 embedding 提取器。
- OpenCV 浅层特征仍然保留，用于可解释提问和概念总结。
- confirmed 样本进入 `datasets/train/<class>`。
- candidate、rejected、unknown 样本进入 `sample_pools/`。


