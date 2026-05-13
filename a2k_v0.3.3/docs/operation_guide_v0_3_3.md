# a2k_v0.3.3 操作指南

这一版在 v0.3.2 的持续学习基础上，加入两个重点：

1. **错误后追问**：当系统识别错、用户选择正确类别后，系统会继续问“为什么错”。
2. **类别对经验**：系统把错因记录到 `metadata/pairwise_experience.json`，下次遇到同类混淆时优先参考这些经验，减少循环问无效问题。

## 创建任务

```bat
python scripts\init_task.py --name fruit_task --classes apple banana pear grape orange --output D:\a2k_test
```

## 放图片

已知样本放入：

```text
D:\a2k_test\fruit_task\datasets\train\类别名
```

未标注图片放入：

```text
D:\a2k_test\fruit_task\datasets\unlabeled
```

`unlabeled` 里的图片可以任意命名，运行时会自动重命名为 `img_001.jpg`、`img_002.jpg`。

## 运行

```bat
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml
```

不弹图片：

```bat
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml --no-preview
```

## 错误后追问怎么用

如果系统预测 apple，但你选择真实类别 grape，它会继续问：

```text
系统刚才把 grape 识别成了 apple。为了避免下次再犯同类错误，请你选择主要原因。
A. 颜色或色系差异明显
B. 整体形状、结构或轮廓不同
C. 表面纹理、颗粒感或局部重复结构不同
D. 背景、光线、遮挡或主体不清楚影响了判断
E. 不确定 / 其他原因
```

你选完后，系统会记录到：

```text
metadata/pairwise_experience.json
```

这份文件会保存类似 `apple vs grape`、`apple vs pear` 的类别对经验。

## 这一版重点观察什么

- 识别错后是否会追问错因。
- `metadata/pairwise_experience.json` 是否生成。
- 同一类别对多次混淆后，问题是否比之前更贴近重点。
- `datasets/train/<class>/` 是否继续累积 confirmed 样本。
- `datasets/unlabeled/` 处理后是否被移走。

## 仍未完成

- 还没有深度特征。
- 还没有主体裁剪/背景弱化。
- 还没有真正的类别专属部件检测。
- 现在的类别对经验主要影响提问和权重，还不是强识别模型。
