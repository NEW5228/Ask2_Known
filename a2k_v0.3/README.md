# Ask2Know v0.3.0

**Ask2Know / 问知** 是一个面向个人和小团队的低样本主动教学训练框架原型。

它不是水果识别器、宠物识别器或车辆识别器，而是一个通用框架：用户可以自己配置想训练的任务，例如水果、宠物、车辆、植物、零件、图标等。

当前 v0.3.0 的目标是：

- 不再把代码写死为 apple / strawberry。
- 用户可以通过配置文件和数据文件夹定义自己的类别。
- 系统用少量样本建立对象原型。
- 系统在不确定时主动提问。
- 用户回答后，系统更新特征权重和问题权重。
- 系统保存学习日志和经验报告。

> 注意：v0.3.0 仍然是浅层特征原型系统，不是深度学习大模型。它用于验证 Ask2Know 的训练框架思想。

---

## 1. 当前框架思想

传统方式通常是：

```text
大量图片 → 大量标注 → 训练模型 → 得到结果
```

Ask2Know 想尝试的是：

```text
少量样本
↓
系统提取基础概念特征
↓
系统建立初始对象原型
↓
系统识别未知样本
↓
不确定时主动问用户
↓
用户给少量高价值反馈
↓
系统调整特征权重、问题权重和样本经验
↓
下一次判断更贴近当前任务
```

核心目标：**减少用户数据量、减少人工标注量、减少重训成本。**

---

## 2. 安装环境

建议使用 Conda：

```bat
conda create -n ask2know python=3.10
conda activate ask2know
```

进入项目目录：

```bat
cd D:\ask2know_framework_v0_3_0
```

安装依赖：

```bat
pip install -r requirements.txt
```

如果你想用开发者安装方式：

```bat
pip install -e .
```

这一步不是必须，但后面 Ask2Know 会逐步往 Python 包方向发展。

---

## 3. 快速运行水果 demo

生成演示数据：

```bat
python scripts/create_demo_dataset.py
```

运行：

```bat
python run_demo.py --config configs/fruit_demo.yaml
```

如果不想每次自动打开图片：

```bat
python run_demo.py --config configs/fruit_demo.yaml --no-preview
```

运行后你会看到：

```text
正在识别第 1/4 张未知样本
图片路径: datasets/fruit_demo/unlabeled/unknown_apple_red.jpg

初始识别结果:
1. apple: 0.951
2. strawberry: 0.942

候选差距较小，系统不确定，进入主动询问。
```

然后系统会根据当前证据生成自然语言问题，用户用选项回答。

---

## 4. 创建你自己的任务

例如你想训练猫狗识别任务：

```bat
python scripts/init_task.py --name catdog_demo --classes cat dog
```

它会创建：

```text
datasets/catdog_demo/
├─ objects.json
├─ concepts.json
├─ train/
│  ├─ cat/
│  └─ dog/
└─ unlabeled/
```

同时生成：

```text
configs/catdog_demo.yaml
```

然后你把已知图片放进去：

```text
datasets/catdog_demo/train/cat/
datasets/catdog_demo/train/dog/
```

把未知图片放进去：

```text
datasets/catdog_demo/unlabeled/
```

运行：

```bat
python run_demo.py --config configs/catdog_demo.yaml
```

---

## 5. 数据集格式

每个任务建议使用下面结构：

```text
datasets/task_name/
├─ objects.json
├─ concepts.json
├─ train/
│  ├─ class_a/
│  ├─ class_b/
│  └─ class_c/
└─ unlabeled/
```

### train/

这里放已知类别样本。文件夹名就是类别名。

例如：

```text
train/bmw/
train/audi/
train/benz/
```

### unlabeled/

这里放待系统判断的未知图片。

---

## 6. objects.json

定义类别，例如：

```json
{
  "objects": [
    {
      "object_id": "C001",
      "name": "cat",
      "display_name": "猫",
      "description": "用户自定义类别"
    },
    {
      "object_id": "C002",
      "name": "dog",
      "display_name": "狗",
      "description": "用户自定义类别"
    }
  ]
}
```

注意：`name` 要和 `train/` 下面的文件夹名字一致。

---

## 7. concepts.json

这是可选的用户初始概念提示。

例如：

```json
{
  "concepts": [
    {
      "object_a": "cat",
      "object_b": "dog",
      "hint": "猫和狗都可能有相似颜色，因此颜色不是唯一依据；整体轮廓和局部结构可能更重要。",
      "important_features": ["contour", "texture"],
      "weak_features": ["size"]
    }
  ]
}
```

v0.3.0 会根据这些提示调整初始特征权重。

---

## 8. 配置文件

示例：

```yaml
task:
  name: catdog_demo
  type: image_object_recognition
  description: 用户自定义 Ask2Know 任务

paths:
  dataset_dir: datasets/catdog_demo
  output_dir: outputs/catdog_demo

features:
  color: true
  size: true
  contour: true
  texture: true

learning:
  initial_weights:
    color: 0.25
    size: 0.15
    contour: 0.30
    texture: 0.30
  update_step: 0.07
  min_weight: 0.05
  max_weight: 0.70

confidence:
  auto_accept_threshold: 0.80
  ask_user_threshold: 0.12

question:
  max_questions_per_sample: 1
  enable_question_reward: true
```

---

## 9. 当前输出文件

运行后会在 `outputs/task_name/` 生成：

```text
feature_weights.json       当前特征权重
question_weights.json      问题权重和提问次数
prototype_model.json       对象原型
experience_report.json     本轮经验报告
logs/demo_log.json         详细学习日志
```

`experience_report.json` 是 v0.3.0 新增的，它用于记录本轮学习得到的经验。

---

## 10. 当前版本限制

v0.3.0 还没有实现：

- 深度模型特征，如 CLIP / ResNet。
- 图像分割与自动抠图。
- 种子区域教学正式功能。
- 多任务经验迁移。
- GUI / Web 界面。
- 真正的 pip 发布版。

当前最重要的是先跑通通用框架闭环。

---

## 11. 版本规划

```text
v0.1.0  最小水果 demo
v0.2.0  证据驱动自然语言提问
v0.3.0  通用任务配置，不再写死水果
v0.4.0  样本池机制：confirmed / candidate / rejected
v0.5.0  经验库增强和任务经验复用
v0.6.0  可选种子区域教学
v0.7.0  Python 包化和命令行工具
v1.0.0  稳定公开版本
```

---

## 12. Ask2Know 主线提醒

Ask2Know 的主线是：

> 它是一个低样本主动教学训练框架，不是单一识别模型。核心目标是减少用户数据量、标注量和重训成本，通过主动提问、用户反馈、权重更新、问题收益和经验复用，让模型快速适配用户自定义任务。

所有新功能都要问：

```text
它有没有减少用户工作量？
它有没有减少样本需求？
它有没有增强主动提问、反馈学习或经验复用？
它有没有让框架更通用？
```

如果没有，就先暂缓。
