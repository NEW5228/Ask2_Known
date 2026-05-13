# Ask2Know v0.3.0 操作指南

## 目标

v0.3.0 的目标是把 Ask2Know 从水果专用演示改成用户可配置的通用任务框架。

## 推荐操作流程

### 1. 安装依赖

```bat
conda create -n ask2know python=3.10
conda activate ask2know
pip install -r requirements.txt
```

### 2. 运行内置 demo

```bat
python scripts/create_demo_dataset.py
python run_demo.py --config configs/fruit_demo.yaml
```

### 3. 创建自己的任务

```bat
python scripts/init_task.py --name my_task --classes class_a class_b
```

### 4. 放图片

已知样本：

```text
datasets/my_task/train/class_a/
datasets/my_task/train/class_b/
```

未知样本：

```text
datasets/my_task/unlabeled/
```

### 5. 运行自己的任务

```bat
python run_demo.py --config configs/my_task.yaml
```

## 确认样本时怎么选？

- `y`：确认系统判断正确，该图片会加入对应类别原型。
- `n`：系统判断错误，你可以输入正确类别。
- `skip`：跳过，不加入正式样本库。看不清、不确定时建议用 skip。

## 真实图片建议

第一批真实图片最好满足：

- 单个主体。
- 背景尽量简单。
- 不要太糊。
- 每个类别至少 3 张，推荐 5 张。
- `unlabeled` 里先放少量测试图，不要一上来放太多。

## 当前不适合的图片

- 一张图里有多个主体。
- 主体太小。
- 背景和主体颜色接近。
- 光线极端。
- 严重遮挡。

这些不是永久不能支持，而是 v0.3.0 浅层特征版本不擅长。
