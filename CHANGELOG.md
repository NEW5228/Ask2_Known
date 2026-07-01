# a2k_v0.5.0 更新日志

## v0.5.0 调整

- 增加分层交互识别流程，支持用 taxonomy path 在 Top-K 候选内做可解释候选消解。
- 增加动态候选 ASK 兜底：当没有可用 taxonomy 问题时，可围绕当前候选生成候选级追问并只调整本次排序。
- 增加在线 ASK 模拟评估参数 `--simulate-ask-mode`，支持 `taxonomy`、`dynamic` 和 `auto` 三种模式。
- 增加 `texture` 特征预设和 DTD 数据准备脚本，用于验证纹理、材料和表面状态类任务。
- 增加交互教学后 held-out 评估脚本，用于检查教学、错例回流和导出前验证是否真正提升。
- 继续保留经典相似度识别模式；新建训练集时可在分层交互识别和经典相似度识别之间切换。
- 项目版本升级为 0.5.0。

# a2k_v0.4.63.1 更新日志

## v0.4.63.1 调整

- 增加桌面端“验证”页，使用已确认的 unknown 学习记录进行模型验证。
- 模型导出前要求验证通过。
- 验证状态对用户只显示“验证通过”或“验证未通过”，内部报告保留完整数据。
- 项目版本升级为 0.4.63.1。

## v0.4.63.0 调整

- 桌面窗口新增更完整的项目流程、学习确认和模型导出交互。
- 学习页新增明确的模型判断输出，并优化用户确认区布局。
- 项目页新增批量导入训练文件夹、类别删除、项目内容滚动查看能力。
- 数据加载器在存在 `objects.json` 时只加载项目清单中的类别，避免已移除类别继续参与训练。
- 新增离线模型导出和 Python 离线模型包能力，支持自定义导出位置和模型名称。
- 新增模型预测、文件夹预测、离线包生成和本地服务相关脚本。
- 项目版本升级为 0.4.63.0。

# a2k_v0.4.61.1 更新日志

## v0.4.61.1 调整

- 新增 car preset：auto 可根据常见汽车品牌类名选择汽车品牌识别模板，默认启用颜色、车身轮廓、纹理、表面、部件、尺寸、字标和标识线索。
- 新增 car_shape 和 car_part 内部视觉信号，用轻量特征模拟车身轮廓、格栅、灯组、车窗、徽标/字标区域等汽车品牌识别线索。
- init_task.py 支持 --feature-preset car，并为汽车品牌生成 car/vehicle/front grille/badge/logo/wordmark 相关 CLIP 文本 prompt。
- evaluate_unlabeled.py 的 --online-experience 默认使用 --online-model-update all，让用户确认过的正确样本和纠错样本都能增量加入模型。
- 项目版本升级为 0.4.61.1。

# a2k_v0.4.61.0 更新日志

## v0.4.61.0 调整

- 新增 similarity.late_fusion：对 top-k 候选做通用后融合重排，综合 base、kNN、text semantic 和 crop rerank 等证据，减少真类已在 top-3 但排序错误的混淆问题。
- evaluate_unlabeled.py 的在线学习新增 --online-model-update，默认在用户纠错后把错误样本增量加入模型，让 prototype、kNN、crop evidence 和 pairwise 相似度真正更新。
- 在线经验默认只使用正向支持证据，不再默认用历史误导源做负向扣分；top-1 翻转需要至少两个证据源支持，可通过 --online-allow-negative-adjustments 和 --online-min-sources-for-flip 调整。
- PairVisualRuleMemory 默认不直接翻转 top-1，只作为保守证据参与；需要实验时可显式使用 --visual-rule-allow-rank-flip。
- 37 类宠物评估集 a2k_0442_all37_train30_eval20 上，late_fusion 离线评估达到 672/740 = 90.81%，开启在线增量学习后达到 673/740 = 90.95%。
- 项目版本升级为 0.4.61.0。

# a2k_v0.4.6.2b 更新日志

## v0.4.6.2b 调整

- 新增 PairVisualRuleMemory：从在线纠错中学习 pair-specific concept 视觉规则，而不是只记录哪个评分源曾经误导。
- evaluate_unlabeled.py 默认在 --online-experience 下启用视觉规则，可用 --disable-visual-rules 关闭，并输出 visual_rule_memory_report.json。
- 评估报告新增 visual_rule_delta、visual_rule_evidence、visual_rule_adjustment 和 visual_rule_memory，便于判断规则救回/改坏样本。
- 版本命名为 0.4.6.2b，继续保留 0.4.6.x 实验线，不升级到 0.4.7。

# a2k_v0.4.6.2a 更新日志

## v0.4.6.2a 调整

- 新增 evaluate_unlabeled.py 的 --shuffle/--seed/--online-experience 流式评估模式，先预测再揭示真值，只把当前错误用于后续样本。
- 新增 OnlineConfusionExperience：从早先的易混淆错误中学习 pair/source 证据，对低 margin 的 top1/top2 做轻量重排。
- 评估报告新增 evaluation_mode、online_experience_summary、online_experience_report.json 和 raw_top_predictions，便于对比在线经验是否真实提升。
- 版本命名为 0.4.6.2a，继续保留 0.4.6.2 主线，不升级到 0.4.7。

# a2k_v0.4.6.2 更新日志

## v0.4.6.2 调整

- 新增 confusion_experience_report：评估时自动汇总易混淆类别对、错误来源、true label 是否出现在 top-k，以及候选判别特征。
- PrototypeModel 新增 pair_discriminative_summary，用类别 prototype 和概念 prototype 自动总结类别对之间的潜在区分点。
- 交互纠错后允许用户写一句区分依据，并写入 pairwise_experience.json，后续同类混淆提问会优先显示这条经验。
- 版本号固定为 0.4.6.2；后续实验不再自动升级到 0.4.7。

# a2k_v0.4.6.1 更新日志

## v0.4.6.1 调整

- deep_features.multi_crop 新增 object 和 head 两个语义倾向 crop：object 使用前景启发式主体框，head 使用主体上半部分作为头脸倾向区域。
- similarity.crop_rerank 新增 trigger_mode，默认使用 margin_and_pair_similarity，避免 crop rerank 大范围介入非混淆样本。
- 默认 crop rerank 权重下调为 0.18，触发 margin 收紧为 0.018，混淆对相似度阈值提高到 0.94。
- 项目版本升级为 0.4.6.1。

# a2k_v0.4.6 更新日志

## v0.4.6 调整

- 新增 deep_features.multi_crop，默认缓存 full、center 和 five_crop 的 CLIP embedding。
- 新增 similarity.crop_rerank：仅在 top 候选低 margin 或高相似混淆对时，用局部 crop 最近邻证据修正候选类别分数。
- 评估报告新增 crop_rerank_score、crop_rerank_gate_reason、crop_rerank_local_gap 等字段，便于判断 0.4.6 是否优于 0.4.5。
- 项目版本升级为 0.4.6。

# a2k_v0.4.5 更新日志

## v0.4.5 调整

- 新增 similarity.robust_prototype：默认只对 CLIP embedding prototype 使用保守 trimmed mean；少于 24 个样本时只生成质量报告，避免低样本下误删有效类内变化。
- 评估报告新增 training_quality_report，输出每类低相似训练样本和更接近混淆类的风险样本，帮助定位为什么更多训练图不一定更好。
- 项目版本升级为 0.4.5。

# a2k_v0.4.4.2 更新日志

## v0.4.4.2 调整

- 新增 similarity.pairwise_rerank：当 top1/top2 属于高相似类别且最终分差很小时，使用这两个类别的局部近邻证据重新加权，降低相似品种互相混淆的风险。
- 评估报告新增 pairwise_score、pairwise_score_weight_used、pairwise_gate_reason、pairwise_pair_similarity 和 pairwise_local_gap，便于复盘 pairwise rerank 是否参与预测。
- 项目版本升级为 0.4.4.2。

# a2k_v0.4.4.1 更新日志

## v0.4.4.1 调整

- 将 sub-prototypes 改为保守门控：默认权重从 0.12 降到 0.06，只在局部中心相对主 prototype 有足够增益时参与加权。
- rank flip 只在基础分差很小且原 top 类别没有明显 prototype 优势时允许，降低“修正一个错误但改错一个正确样本”的风险。
- 评估报告新增 subprototype_score_weight_used、subprototype_gate_reason、subprototype_gain_over_prototype 和 subprototype_top_gap，便于复盘 sub-prototypes 是否真正参与了当前预测。
- 项目版本升级为 0.4.4.1。

# a2k_v0.4.4 更新日志

## v0.4.4 新增/调整

- 新增 sub-prototypes 多原型评分：每类样本足够多时，会基于 CLIP embedding 生成多个类别子中心。
- similarity.sub_prototypes 支持 enable、max_centers、min_samples_per_center 和 score_weight 配置。
- 预测结果和评估报告新增 subprototype_score，用于观察局部类别中心是否修正单一 prototype 的平均化问题。
- 项目版本升级为 0.4.4。

# a2k_v0.4.3.1 更新日志

## v0.4.3.1 新增/调整

- 新增通用识别诊断模块，评估报告会输出 low margin、prototype/kNN/text/concept 冲突或主导等 reason codes。
- evaluate_unlabeled.py 为每个样本写入 diagnosis，并汇总 needs_review_count 与 reason_counts。
- 新增 similarity.concept_gate，concept 只有在当前候选间有足够区分力时才按完整权重参与最终分数；弱区分时默认不强拉分。
- 新增 diagnostics.low_margin_threshold 和 diagnostics.weak_signal_threshold 配置。
- 项目版本升级为 0.4.3.1。

# a2k_v0.4.3 更新日志

## v0.4.3 新增/调整

- 新增 CLIP text semantic scoring：用类别名生成文本 prompt embedding，与图片 embedding 的相似度作为辅助识别信号。
- similarity.text_semantic 支持配置 enable、score_weight 和 prompt_templates，默认新任务启用，权重为 0.08。
- run_demo.py 和 evaluate_unlabeled.py 输出新增 text_semantic_score，便于观察文本语义分数对最终结果的影响。
- 项目版本升级为 0.4.3。

# a2k_v0.4.2.1n 更新日志

## v0.4.2.1n 修复

- 优化 `bootstrap_clusters.py` 的训练集写入流程：复制前生成 review 清单，标出每组低相似度离群候选，并支持按单张图片跳过。
- 新增 `--report-only` / `--dry-run`，只输出聚类与复制计划报告，不写入 `datasets/train/<class>/`。
- `--no-copy` 保留为兼容参数，行为等同于 report-only。
- 运行时版本升级为 `0.4.2.1n`；Python package 元数据使用 PEP 440 合法版本 `0.4.2.1.post1`。

# a2k_v0.4.2.1 更新日志

## v0.4.2.1 修复

- 修复 deep feature 缓存键未包含 CLIP 模型和权重配置的问题，避免切换 `model_name` 或 `pretrained` 后复用旧 embedding。
- 修复 `bootstrap_clusters.py` 未复用 embedding path cache 的问题，粗分 unknown 图片时会优先走 `DeepFeatureAdapter.extract_path()`。
- 修复 bootstrap 更新项目元数据时只保留本次映射类别、可能丢失既有类别的问题。
- 增强样本池类别索引，记录 `storage_name`、原始 `label` 和 `display_name`，降低带空格/特殊字符类别名造成目录名和显示名混淆的风险。
- 清理 v0.4.1.1 更新日志中的合并冲突标记。
- 项目版本统一升级为 `0.4.2.1`。

# a2k_v0.4.2 更新日志

## v0.4.2 新增/调整

- 新增 `datasets/unknown/` 作为待学习样本入口。
- 调整 `datasets/unlabeled/<class>/` 为带真实标签的评估集入口。
- 新增 `scripts/bootstrap_clusters.py`，支持从 unknown 混合图片聚类建训练集。
- 新增 `scripts/evaluate_unlabeled.py`，支持基于 unlabeled 评估集输出准确率报告。
- 项目版本统一升级为 `0.4.2`。

# a2k_v0.4.1.1 更新日志

## v0.4.1.1 新增/调整

- 面向多数图像识别项目校准默认评分策略：CLIP embedding 作为主要分类信号，浅层 OpenCV 特征主要服务于解释、主动提问和用户反馈。
- 基于 Kaggle fruits 70/30 评估校准默认配置：训练 251 张、测试 108 张，最终默认配置达到 107/108，准确率 99.07%。
- 默认评分权重改为 CLIP embedding 强主导：`embedding: 5.00`，浅层可解释特征降低到 `0.02~0.03`。
- `concepts.score_weight` 从 `0.25` 降到 `0.05`，概念层保留为辅助解释和弱校正。
- `learning.max_weight` 从 `0.70` 提升到 `0.95`，避免归一化后过度压低 embedding 权重。
- 默认主动提问阈值从 `ask_user_threshold: 0.12` 降到 `0.03`，适配 CLIP 相似度分数通常更密集的尺度，减少不必要打扰。
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
