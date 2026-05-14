# Ask2Know

Ask2Know 是一个面向个人和小团队的低样本主动教学训练框架。它不是做大模型训练，而是用少量已知图片建立类别原型，在识别不确定时主动询问用户，并把确认、纠错和错因记录为可累积的经验。

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

默认不弹窗，避免图片查看器占用文件。

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

如果系统把 grape 识别成 apple，用户纠正后，系统会追问为什么错，并支持多选：

```text
A. 颜色或色系差异明显
B. 整体形状、结构或轮廓不同
C. 表面纹理、颗粒感或局部重复结构不同
D. 单体/聚集结构不同
E. 背景、光线、遮挡影响
F. 不确定 / 其他原因
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
```

这是系统根据错误经验形成的弱总结，不是最终真理，但会帮助后续问题生成和概念层设计。

## 数据增强

默认启用温和增强，用于建立类别原型：亮度、轻微旋转、轻微裁剪。可以在 `task_config.yaml` 中关闭：

```yaml
augmentation:
  enable: false
```

## 文档

- `docs/user_guide.md`：完整使用流程。
- `docs/framework_design.md`：框架设计。
- `docs/future_modules.md`：后续模块规划。
- `CHANGELOG.md`：版本更新记录。
