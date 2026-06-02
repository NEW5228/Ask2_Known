# Ask2Know

当前版本：`0.4.63.0`

Ask2Know 是一个面向低样本图像识别任务的本地交互式学习系统。系统以 CLIP 图像嵌入、原型相似度、kNN 近邻证据、可解释视觉特征和用户反馈为核心，支持在少量已知样本基础上逐步完善类别理解，并可导出 Python 离线模型包用于独立部署。

## 核心能力

- 低样本图像分类：基于每类少量训练样本构建类别原型。
- 相似度证据融合：结合 prototype、sub-prototype、kNN、文本语义、局部视觉规则等信号进行评分。
- 主动提问与用户确认：在低置信度或类别混淆时向用户询问可解释视觉差异。
- 在线经验积累：记录用户修正、类别混淆、pairwise 经验和视觉规则。
- 桌面窗口操作：支持新建项目、导入数据、学习确认、类别维护和一键导出模型。
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

窗口流程：

1. 新建项目或打开已有项目配置。
2. 导入训练图片、unknown 图片和可选评估图片。
3. 加载项目并开始学习。
4. 根据模型判断进行确认或修正。
5. 在需要时维护类别清单。
6. 导出离线模型包。

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
datasets/unknown/             待学习、待确认样本
datasets/unlabeled/<class>/   带真实标签的评估样本
outputs/                      运行报告和模型缓存
metadata/                     经验数据和项目元数据
sample_pools/                 candidate、rejected 等样本池
```

系统在运行时会根据配置规范化训练集和 unknown 图片文件名。删除类别时，桌面窗口只会从项目清单和配置中移除类别，不会删除本地图片文件。

## 运行学习

命令行运行：

```bat
python run_demo.py --config D:\a2k_test\<task_name>\configs\task_config.yaml
```

需要图片预览时：

```bat
python run_demo.py --config D:\a2k_test\<task_name>\configs\task_config.yaml --preview
```

桌面窗口中，点击“加载项目”后可在“学习”页开始逐张确认。模型会显示当前判断、候选类别、分数和 Top2 差距。若判断正确，直接确认；若判断错误，选择真实类别或输入新类别后提交修正。

## 评估

将评估图片按真实类别放入：

```text
datasets/unlabeled/<class>/
```

运行评估：

```bat
python scripts\evaluate_unlabeled.py --config D:\a2k_test\<task_name>\configs\task_config.yaml
```

启用在线经验模拟：

```bat
python scripts\evaluate_unlabeled.py --config D:\a2k_test\<task_name>\configs\task_config.yaml --online-experience
```

评估报告输出：

```text
outputs/evaluation_report.json
```

## 离线模型导出

桌面窗口支持一键导出离线模型包。用户可以自定义导出位置和模型名称；如果不填写，则使用默认位置和默认命名。

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
outputs/logs/demo_log.json
outputs/experience_report.json
outputs/experience_summary.json
outputs/class_understanding_summary.json
outputs/class_understanding_summary.md
```

## 版本说明

v0.4.63.0 重点更新：

- 优化桌面窗口布局和学习页交互。
- 增加批量导入训练文件夹。
- 增加项目类别删除能力。
- 增加项目内容表格滚动条。
- 增加学习页明确的模型判断输出。
- 增加一键导出离线模型包，并支持自定义导出位置和模型名称。
- 增加 Python 离线模型包、预测脚本和本地服务脚本。

