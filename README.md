# Ask2Know

Ask2Know 是一个面向个人和小团队的低样本主动教学训练框架。v0.4.0 开始转向 embedding + similarity + user feedback pipeline：系统同时使用可解释浅层特征、轻量 image embedding、prototype 相似度、kNN 相似样本证据、概念层和用户反馈来完成低样本持续学习。

v0.4.0 默认启用轻量 OpenCV embedding，不引入 CLIP/DINO 等重型依赖；后续可以通过 `deep_features.provider` 接入本地 CLIP/DINO 适配器。embedding 是内部评分通道，主动提问仍优先围绕颜色、形状、纹理、表面、部位、文字和标识等用户能判断的概念。

## 安装

```bat
conda activate ask2know
cd D:\ask2know_framework
pip install -r requirements.txt
```

## 新建任务

```bat
python scripts\init_task.py --name fruit_test3 --classes apple banana pear grape orange --output D:\a2k_test
```

可以在创建任务时选择特征预设和用户可见特征：

```bat
python scripts\init_task.py --name fruit_test3 --classes apple banana pear --output D:\a2k_test --feature-preset fruit --features color shape texture surface part size
```

交通标识类任务可以启用文字和标识特征：

```bat
python scripts\init_task.py --name sign_test --classes stop turn_left no_entry --output D:\a2k_test --feature-preset traffic_sign --features color shape text sign
```

用户可选特征包括 `color`、`shape`、`texture`、`surface`、`part`、`size`、`text`、`sign`。`surface` 用来描述绒毛、粗糙表皮、斑点/籽点和反光感；`part` 用来描述果皮、果肉、籽点、果核、切面、瓣状结构和厚皮/瓜皮感。`quality` 是系统内部的样本质量检查，用来判断主体是否清晰、背景是否干扰，不作为用户选择的训练特征展示。`surface`、`part`、`text` 和 `sign` 是轻量 OpenCV 模拟特征，不做真实 OCR、语义分割或完整目标检测。

## 运行内置 demo

第一次运行内置 demo 前先生成示例图片：

```bat
python scripts\create_demo_dataset.py
python run_demo.py --config configs\fruit_demo.yaml
```

## 在旧项目中追加新类别

不要重新建项目。如果你已经有 `D:\a2k_test\fruit_test3`，要加入 cherry：

```bat
python scripts\add_class.py --project D:\a2k_test\fruit_test3 --class cherry
```

然后把 cherry 图片放到：

```text
D:\a2k_test\fruit_test3\datasets\train\cherry
```

## 运行旧项目继续学习

v0.3.7 默认不弹窗，避免图片查看器卡死。

```bat
python run_demo.py --config D:\a2k_test\fruit_test3\configs\task_config.yaml
```

需要预览时才加：

```bat
python run_demo.py --config D:\a2k_test\fruit_test3\configs\task_config.yaml --preview
```

## 文件命名规则

- `datasets/unlabeled/` 里的图片可以任意命名，运行时会自动整理为 `img_001.jpg`。
- `datasets/train/<class>/` 里的图片也可以任意命名，运行时会自动整理为 `class_001.jpg`、`class_002.jpg`。
- confirmed 后的样本会进入长期训练库 `datasets/train/<class>/`，并顺延编号。

## 错误后追问多选

如果系统把 grape 识别成 apple，用户纠正后，系统会追问为什么错。v0.3.7 支持多选：

```text
A. 颜色或色系差异明显
B. 整体形状、结构或轮廓不同
C. 表面纹理、颗粒感或局部重复结构不同
D. 单体/聚集结构不同
E. 表面绒毛、粗糙皮、斑点/籽点或反光不同
F. 果皮、果肉、籽点、果核、切面或瓣状结构不同
G. 背景、光线、遮挡影响
H. 不确定 / 其他原因
```

可以输入：

```text
A,B,D
```

或：

```text
ABD
```

## 自我总结

运行结束后会生成：

```text
outputs/experience_summary.json
metadata/experience_summary.json
outputs/class_understanding_summary.json
outputs/class_understanding_summary.md
```

这是系统根据错误经验和基础视觉概念形成的弱总结，不是最终真理，但会帮助后续问题生成、概念层设计和经验迁移。
其中 `class_understanding_summary.md` 面向人工检查，会按类别输出“系统目前认为这个类别更像什么”。

## 基础视觉概念层

v0.3.7 默认启用轻量概念层：

```yaml
concepts:
  enable: true
  score_weight: 0.25
```

系统会从现有 OpenCV 特征中推导基础概念，例如颜色概念、形状概念、纹理/重复结构、主体清晰度和背景干扰。预测时会同时比较类别的特征原型和概念原型；提问时也会尝试用“我看到偏红、接近圆形、有聚集感”这类语言解释自己的观察。

## v0.4.0 混合相似度

新任务默认包含：

```yaml
deep_features:
  enable: true
  provider: opencv
  feature_name: image_embedding

similarity:
  mode: hybrid
  knn:
    enable: true
    k: 3
    score_weight: 0.20
```

最终分数由 prototype、kNN 相似样本和概念原型共同构成。运行输出会显示 `proto`、`knn`、`concept` 分数来源，并给出最相似训练样本路径，方便人工检查。

## 数据增强

v0.3.7 默认启用温和增强，用于建立类别原型：亮度、轻微旋转、轻微裁剪。可以在 `task_config.yaml` 中关闭：

```yaml
augmentation:
  enable: false
```

## 后续留痕

`docs/future_modules.md` 中保留了后续模块规划：

- crawler_external_candidates：外部候选图片采集，不直接进入 confirmed。
- visual_concept_layer：全局视觉概念层。
- deep_feature_adapter：CLIP / ResNet / MobileNet 等深度特征适配器。

## 版本纪律

v0.4.0 之前如果还有 bug，不强行进入 v0.4.0。继续用 v0.3.7 等补丁版本修复，直到 v0.3.x 稳定。
