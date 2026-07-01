# Ask2Know

当前版本：`0.5.0`

Ask2Know 是一个面向低样本图像识别任务的本地交互式学习系统。系统以 CLIP 图像嵌入、原型相似度、kNN 近邻证据、可解释视觉特征和用户反馈为核心，支持在少量已知样本基础上逐步完善类别理解，并可导出 Python 离线模型包用于独立部署。

## 0.5.0 重点

- 分层交互识别：先用颜色、形状、符号、字段等可解释标签缩小候选范围，再在候选集合内做细粒度识别。
- 二问 ASK 流程：默认从 Top10 候选中组织最多 8 个选项，最多连续询问两轮，把用户修正转化为可复用的在线经验。
- 算法可选：创建训练集时可选择“分层交互识别”或“经典相似度识别”，旧的单层识别模式仍保留。
- 交通指示牌支持：提供交通标志数据准备脚本，支持生成训练集、`unlabeled` 评估集和语义元数据，用于验证多层识别效果。
- 预缓存：支持提前缓存图片、裁剪图和文本标签 embedding，减少评估和交互时的重复计算。
- 桌面端流程重构：启动页分为创建项目和加载项目；工作台包含项目管理、添加数据集、创建训练集、训练模型、评估模型、分析图片、模型导出。
- 数据目录规范：带真实标签的评估数据使用 `datasets/unlabeled/<class>/`，待交互学习或临时分析的图片使用 `datasets/unknown/`。
- 导出约束调整：需要先完成评估模型流程，之后才能导出模型；分析图片不影响导出。

## 核心能力

- 低样本图像分类：基于每类少量训练样本构建类别原型。
- 相似度证据融合：结合 prototype、sub-prototype、kNN、文本语义、局部视觉规则和分层标签信号评分。
- 主动提问与用户确认：在低置信度或类别混淆时向用户询问可解释视觉差异。
- 在线经验积累：记录用户修正、类别混淆、pairwise 经验和视觉规则，用于后续样本重排。
- 桌面窗口操作：支持项目创建/加载、添加数据集、训练、评估、分析图片和模型导出。
- 离线模型导出：生成 `.a2kmodel.json` 和可复制使用的 Python 离线模型包。

## 环境要求

- Python 3.9+
- OpenCLIP 相关依赖
- Windows 环境下可使用 `app_desktop.py` 桌面窗口

安装依赖：

```bat
pip install -r requirements.txt
pip install -e .
```

从 v0.4.1 起，OpenCLIP 是必需运行依赖。系统不会回退到 OpenCV embedding。如果 `torch`、`open_clip_torch` 或配置的 CLIP 权重不可用，模型初始化会失败。

默认 CLIP 配置：

```yaml
model_name: ViT-B-32
pretrained: laion2b_s34b_b79k
```

## 桌面窗口

启动桌面窗口：

```bat
python app_desktop.py
```

推荐流程：

1. 在启动页创建项目或加载已有项目。
2. 在“添加数据集”中导入训练图片，必要时导入 `unlabeled` 评估图片。
3. 在“创建训练集”中选择算法模式、在线/离线、ASK 问题数、候选数量和选项数量。
4. 在“训练模型”中生成或刷新当前项目模型。
5. 在“评估模型”中使用 `datasets/unlabeled/<class>/` 评估当前模型。
6. 在“分析图片”中添加图片并查看训练后模型的识别结果。
7. 评估完成后，在“模型导出”中导出离线模型包。

训练图片支持两种导入方式：

- 单类导入：为指定类别补充图片。
- 批量导入：选择一个包含类别子文件夹的目录，系统按子文件夹名自动导入。

批量导入目录示例：

```text
train_images/
  class_a/
    001.jpg
    002.jpg
  class_b/
    001.jpg
```

## 创建项目

命令行创建项目：

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
- `car`
- `traffic_sign`
- `texture`

用户可见特征：

- `color`
- `shape`
- `texture`
- `surface`
- `part`
- `size`
- `text`
- `sign`

`quality` 是系统内部质量特征，用于评估清晰度、背景干扰、模糊和样本可学习性，不作为用户可选特征展示。

## 数据目录

项目创建后，主要数据目录如下：

```text
datasets/train/<class>/       已确认训练样本
datasets/unlabeled/<class>/   带真实标签的评估样本
datasets/unknown/             用户添加的待交互学习或待分析图片
outputs/                      运行报告和模型缓存
metadata/                     经验数据和项目元数据
sample_pools/                 candidate、rejected 等样本池
```

系统在运行时会根据配置规范化训练集、`unlabeled` 评估集和 `unknown` 分析图片文件名。删除类别时，桌面窗口只会从项目清单和配置中移除类别，不会删除本地图片文件。

## 运行学习

命令行运行：

```bat
python run_demo.py --config D:\a2k_test\<task_name>\configs\task_config.yaml
```

需要图片预览时：

```bat
python run_demo.py --config D:\a2k_test\<task_name>\configs\task_config.yaml --preview
```

命令行交互会显示模型判断、候选类别、分数、Top2 差距，并在需要时触发分层 ASK 问题。判断正确时直接确认；判断错误时选择真实类别或输入新类别后提交修正。

## 交通指示牌数据

将交通标志数据集准备为 Ask2Know 项目：

```bat
python scripts\prepare_traffic_dataset.py --zip D:\projects\Ask2_Known\data\traffic.zip --output D:\a2k_test --name traffic_signs
```

脚本会生成训练样本、`datasets/unlabeled/<class>/` 评估样本、类别元数据和分层识别配置。默认使用多层识别；如需对比旧流程，可在创建训练集或配置中选择经典相似度识别。

## 评估

将评估图片按真实类别放入：

```text
datasets/unlabeled/<class>/
```

运行基础评估：

```bat
python scripts\evaluate_unlabeled.py --config D:\a2k_test\<task_name>\configs\task_config.yaml
```

启用在线经验模拟：

```bat
python scripts\evaluate_unlabeled.py --config D:\a2k_test\<task_name>\configs\task_config.yaml --online-experience
```

启用二问 ASK 模拟：

```bat
python scripts\evaluate_unlabeled.py --config D:\a2k_test\<task_name>\configs\task_config.yaml --simulate-ask-resolution --simulate-ask-questions 2 --simulate-ask-candidate-top-k 10 --simulate-ask-options 8
```
选择 ASK 模拟模式：

```bat
python scripts\evaluate_unlabeled.py --config D:\a2k_test\<task_name>\configs\task_config.yaml --simulate-ask-resolution --simulate-ask-mode auto
```

`taxonomy` 只使用固定分层问题；`dynamic` 只使用候选级动态追问；`auto` 会优先使用 taxonomy，无法生成分层问题时再使用动态候选追问。

启用预缓存：

```bat
python scripts\evaluate_unlabeled.py --config D:\a2k_test\<task_name>\configs\task_config.yaml --precache --precache-eval
```

主要评估报告输出：

```text
outputs/evaluation_report.json
outputs/unlabeled_validation_report.json
```

为兼容旧流程，系统仍可能同时写出 `outputs/unknown_validation_report.json`，但 0.5.0 的主报告名是 `unlabeled_validation_report.json`。

## 模型导出

桌面窗口支持一键导出离线模型包。导出前需要先完成“评估模型”流程；用户可以自定义导出位置和模型名称，如果不填写，则使用默认位置和默认命名。

默认输出：

```text
outputs_deploy/<task>_offline_model_<timestamp>.a2kmodel.json
outputs_deploy/<task>_offline_model_<timestamp>_offline_model_package/
```

命令行导出模型：

```bat
python scripts\export_model.py --config D:\a2k_test\<task_name>\configs\task_config.yaml
```

使用已有模型缓存导出：

```bat
python scripts\export_model.py --config D:\a2k_test\<task_name>\configs\task_config.yaml --model-cache D:\a2k_test\<task_name>\outputs\prototype_model_cache.json
```

生成 Python 离线模型包：

```bat
python scripts\package_model.py --model D:\path\model.a2kmodel.json --output D:\path\offline_model_package
```

离线包可用于单张图片预测、文件夹批量预测，或按需启动本地服务脚本。

## 主要输出文件

```text
outputs/feature_weights.json
outputs/internal_feature_weights.json
outputs/question_weights.json
outputs/prototype_model.json
outputs/prototype_model_cache.json
outputs/evaluation_report.json
outputs/unlabeled_validation_report.json
outputs/logs/demo_log.json
outputs/experience_report.json
outputs/experience_summary.json
outputs/class_understanding_summary.json
outputs/class_understanding_summary.md
```

## 版本说明

v0.5.0 重点更新：

- 增加分层交互识别流程，支持颜色、形状、符号、字段和局部候选重排。
- 增加 Top10、最多 8 选项、最多二问的 ASK 候选消解流程。
- 增加在线 ASK 模拟评估，可对比静态一问/二问与在线一问/二问效果。
- 保留经典相似度识别模式，用户创建训练集时可选择新旧算法。
- 增加交通指示牌数据准备脚本和多层识别文档。
- 增加 CLIP 图片、裁剪图和文本标签预缓存。
- 优化桌面端首页、工作台导航、数据添加、训练集创建、评估、分析图片和模型导出流程。
- 将评估数据目录规范为 `datasets/unlabeled/`，将分析图片目录规范为 `datasets/unknown/`。

v0.4.63.1 重点更新：

- 增加基于已确认学习记录的模型验证页。
- 模型导出前要求先完成验证，验证通过后才能导出。
- 验证结果对用户只显示“验证通过”或“验证未通过”。

v0.4.63.0 重点更新：

- 优化桌面窗口布局和学习页交互。
- 增加批量导入训练文件夹。
- 增加项目类别删除能力。
- 增加项目内容表格滚动条。
- 增加学习页明确的模型判断输出。
- 增加一键导出离线模型包，并支持自定义导出位置和模型名称。
- 增加 Python 离线模型包、预测脚本和本地服务脚本。
