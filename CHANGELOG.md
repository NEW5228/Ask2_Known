# a2k_v0.4.2 更新日志

## v0.4.2 新增/调整

- 新增 `datasets/unknown/` 作为待学习样本入口。
- 调整 `datasets/unlabeled/<class>/` 为带真实标签的评估集入口。
- 新增 `scripts/bootstrap_clusters.py`，支持从 unknown 混合图片聚类建训练集。
- 新增 `scripts/evaluate_unlabeled.py`，支持基于 unlabeled 评估集输出准确率报告。
- 项目版本统一升级为 `0.4.2`。

# a2k_v0.4.1.1 更新日志

## v0.4.1.1 新增/调整

<<<<<<< HEAD
- 面向多数图像识别项目校准默认评分策略：CLIP embedding 作为主要分类信号，浅层 OpenCV 特征主要服务于解释、主动提问和用户反馈。
- 默认评分权重改为 CLIP embedding 强主导：`embedding: 5.00`，浅层可解释特征降低到 `0.02~0.03`。
- `concepts.score_weight` 从 `0.25` 降到 `0.05`，概念层保留为辅助解释和弱校正。
- `learning.max_weight` 从 `0.70` 提升到 `0.95`，避免归一化后过度压低 embedding 权重。
- 默认主动提问阈值从 `ask_user_threshold: 0.12` 降到 `0.03`，适配 CLIP 相似度分数通常更密集的尺度，减少不必要打扰。
=======
- 基于 Kaggle fruits 70/30 评估校准默认配置：训练 251 张、测试 108 张，最终默认配置达到 107/108，准确率 99.07%。
- 默认评分权重改为 CLIP embedding 强主导：`embedding: 5.00`，浅层 OpenCV 特征降低到 `0.02~0.03`。
- `concepts.score_weight` 从 `0.25` 降到 `0.05`，概念层保留为辅助解释和弱校正。
- `learning.max_weight` 从 `0.70` 提升到 `0.95`，避免归一化后过度压低 embedding 权重。
- 默认主动提问阈值从 `ask_user_threshold: 0.12` 降到 `0.03`，适配 CLIP 分数尺度；在该评估上模拟询问 9/108，并捕获唯一错误样本。
>>>>>>> 22b91cdd204495d26cd3e019b80277f7c2f284a0
- 项目版本统一升级为 `0.4.1.1`。

# a2k_v0.4.1 更新日志

## v0.4.1 新增/调整

- OpenCLIP 成为必需 embedding provider：默认 `provider: open_clip`，不再回退到 OpenCV embedding。
- `requirements.txt` 和 `pyproject.toml` 加入 `torch`、`torchvision`、`pillow`、`open_clip_torch`。
- 旧任务如果没有 `deep_features` 配置，会按 v0.4.1 默认值启用 OpenCLIP；如果显式关闭 deep features 或配置非 CLIP provider，会直接报错。
- 新建任务模板默认使用 `ViT-B-32` + `laion2b_s34b_b79k`，并继续缓存 embedding。
- 保留 v0.4.0 的 hybrid similarity、kNN 相似样本证据和概念层，但底层 embedding 从轻量 OpenCV 过渡到 CLIP。
- 项目版本统一升级为 `0.4.1`。

# a2k_v0.4.0 更新日志

## v0.4.0 新增/调整

- 新增 `ask2know.features.deep_adapter.DeepFeatureAdapter`，默认提供轻量 OpenCV image embedding，作为后续 CLIP/DINO/ResNet/MobileNet 适配器入口。
- `PrototypeModel` 升级为 hybrid similarity：prototype 分数为主，kNN 相似样本证据可按权重混入总分。
- 新增内部评分组 `embedding -> image_embedding`。它参与分类和权重输出，但不作为用户主动提问项。
- `run_demo.py` 输出 `proto`、`knn`、`concept` 分数来源，并显示最相似训练样本。
- 新建任务配置默认启用 `deep_features` 和 `similarity.knn`，同时保留 OpenCV 浅层特征、概念层、用户反馈和样本池流程。
- 项目版本统一升级为 `0.4.0`。

# a2k_v0.3.7.3 更新日志

## v0.3.7.3 新增/调整

- 新增用户可选特征组：`part`，用于果皮、果肉、籽点、果核、切面、瓣状结构和厚皮/瓜皮感。
- 新增内部特征：`fruit_part`，继续使用 OpenCV / NumPy 轻量规则，帮助处理切开水果和带明显内部结构的样本。
- fruit 新建任务默认启用 `part`；general 和 traffic_sign 默认不启用 `part`，避免非水果任务概念污染。
- 新增概念：`peel_like`、`flesh_like`、`cut_surface`、`seed_like`、`core_like`、`segment_like`、`rind_like`。
- 新增 `Q_PART_IMPORTANCE`，纠错追问支持选择果皮/果肉/籽/切面等部位原因。
- 修复 v0.3.7.2 默认特征边界：`surface` 仍可用于通用任务，`part` 仅作为水果任务默认启用。

# a2k_v0.3.7.2 更新日志

## v0.3.7.2 新增/调整

- 新增用户可选特征组：`surface`，用于表面绒毛、粗糙表皮、斑点/籽点和反光感。
- 新增内部特征：`surface_mark`，继续使用 OpenCV / NumPy 轻量规则，不引入深度模型。
- 扩展颜色概念：新增 `brown`、`black`、`white`、`gray`、`pink`、`cyan`，让类别总结可以输出“偏棕、偏灰、偏粉、偏青绿”等。
- fruit 和 general 新建任务默认启用 `surface`；traffic_sign 新建任务默认继续启用 `color`、`shape`、`text`、`sign`。
- 新增 `Q_SURFACE_IMPORTANCE`，纠错追问也支持选择 surface 原因。
- 修复概念污染：未启用的 `text`、`sign`、`surface` 概念不会进入当前任务的概念原型和类别理解总结。

# a2k_v0.3.7.1 更新日志

## v0.3.7.1 新增/调整

- 训练结束后新增类别理解总结：`class_understanding_summary.json` 和 `class_understanding_summary.md`。
- 类别理解总结会输出系统目前认为每个类别具有哪些可解释概念，例如偏红、接近圆形、有文字/数字感、像禁止标识等，方便用户检查。
- 新增轻量交通标识特征组：`text` 和 `sign`。
- `text` 用 OpenCV 浅层规则模拟文字/数字区域感，不做真实 OCR。
- `sign` 用 OpenCV 浅层规则模拟箭头、禁止、方向性符号和标识图案，不做完整目标检测。
- 新增 `traffic_sign` 特征预设，`auto` 会根据常见交通标识类名尝试启用。
- `scripts/init_task.py --features` 支持选择 `text` 和 `sign`。

# a2k_v0.3.7 更新日志

## v0.3.7 新增/调整

- 新增水果优先的细化特征：`fruit_color`、`fruit_shape`、`fruit_texture`、`fruit_structure`。
- 新增用户可选特征组：`color`、`shape`、`texture`、`size`。
- `quality` 改为系统内部样本质量检查，不再作为用户可选训练特征展示。
- 新版任务配置使用 `features.preset`、`features.groups`、`features.system`。
- `scripts/init_task.py` 支持 `--feature-preset` 和 `--features`。
- `feature_weights.json` 输出用户可见特征组权重，内部权重另存为 `internal_feature_weights.json`。

## 版本定位

v0.3.6 是基础视觉概念层的首个尝试版，目标是在不增加用户标注负担的前提下，让系统开始形成“颜色、形状、纹理、聚集、主体清晰”等可复用概念。

## v0.3.6 新增/修复

- 新增 `ask2know/concepts/basic_concepts.py`，从现有浅层特征推导 red/yellow/round/elongated/cluster_like/single_object 等基础视觉概念。
- `PrototypeModel` 增加 concept prototype，每个类别同时保存特征原型和概念原型。
- 预测结果新增 `concept_score`，默认以 `concepts.score_weight: 0.25` 混入总分。
- 问题生成器会输出概念观察，例如“我观察到偏红、接近圆形、有聚集感”，减少机械式特征问法。
- 类别对经验新增 `useful_concepts` / `weak_concepts`，错因追问会开始沉淀可迁移概念经验。
- `experience_summary.json` 开始输出类别对相关的基础视觉概念。
- 默认配置和新建任务模板加入 `concepts.enable` 与 `concepts.score_weight`。

## v0.3.5.1 新增/修复

v0.3.5.1 是 v0.3.5 的稳定性补丁版，目标是在不破坏旧项目结构的前提下，修复样本流转、配置项生效和提问价值问题。

- confirmed 样本移动改为文件移动成功后再更新 `dataset_index.json`，降低元数据漂移风险。
- `datasets/unlabeled/` 规范化只处理新增/非标准命名文件，已存在的 `img_001.jpg` 等文件保持稳定。
- confirmed 样本在线加入模型时也使用同样的轻量增强视图，减少本轮增量学习和下轮重跑训练的差异。
- `auto_accept_threshold`、`require_confirm_before_learning`、`enable_question_reward`、`sample_pool.enable`、`move_unlabeled_after_decision` 开始在主流程中生效。
- 默认启用 `quality` 特征并加入初始权重，样本质量问题不再只是文字提示。
- 问题选择器改为优先选择当前图片中最能解释 top1/top2 混淆的关键特征，并在生成问题时说明为什么问这个问题。
- 交互中新类别会同步更新 `objects.json`、训练目录和 `project_meta.json`。
- README 增加内置 demo 生成步骤，避免默认配置指向尚未生成的数据集。

## v0.3.5 新增/修复

- 默认关闭图片弹窗预览，避免 Windows 图片查看器占用文件导致程序卡死。
- 支持在旧项目中继续学习，不要求重新创建任务目录。
- 新增 `scripts/add_class.py`，可在旧项目中追加新类别，例如 cherry。
- 新增 `scripts/normalize_project.py`，可整理旧项目的 train/unlabeled 文件名。
- 训练集图片自动规范化：用户拖入 `datasets/train/<class>/` 的任意文件名会自动重命名为 `class_001.jpg` 格式。
- 错误后追问支持多选，例如同时选择颜色、形状、纹理/聚集结构。
- 提问上下文更清楚，明确当前图片、系统预测、混淆对象和用户应该回答什么。
- 新增弱自我总结 `experience_summary.json`，让系统开始总结自己学到的类别对经验。
- 增加轻量数据增强配置，默认启用温和增强，用于少样本原型建立。
- 增加 future modules 留痕：crawler、visual_concept_layer、deep_feature_adapter。

## 暂不做

- 不实现真正爬虫自动学习。
- 不引入大模型训练。
- 不改变用户项目主结构。
- 不破坏旧版本已确认数据。

## 版本纪律

如果 v0.4.0 前仍有 bug，继续用更小版本修复，例如：

- v0.3.5.1：紧急修复
- v0.3.6：较明确的小功能/稳定性补丁

只有 v0.3.x 的核心流程稳定后，才进入 v0.4.0。
