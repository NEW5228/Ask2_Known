# Ask2Know / a2k_v0.3.1

Ask2Know（简称 a2k）是一个面向个人和小团队的低样本主动教学训练框架。它不是水果识别器、宠物识别器或车辆识别器，而是一个可配置框架：用户决定训练什么，a2k 负责少样本导入、主动提问、用户反馈、权重更新、样本池防污染和经验记录。

## v0.3.1 这版解决什么

v0.3.0 在真实图片上容易出现这些问题：

- 需要用户手动创建 configs、objects.json、concepts.json，容易漏文件。
- 多个类别分数全都很接近时，系统还会强行问 top1/top2 的差异问题。
- 系统预测错时，如果用户误确认，样本容易污染。
- size、contour、texture 相似度容易接近 1.00，真实图区分能力差。

v0.3.1 的目标不是一下子变成强模型，而是先把真实使用流程修安全：

- init_task 支持在外部目录自动创建完整任务。
- 自动生成 task_config.yaml、objects.json、concepts.json。
- 加入 confirmed / candidate / rejected 样本池。
- 加入整体不确定判断：如果 top5 分数都挤在一起，先让用户选择真实类别，不再乱问差异。
- 调整基础特征与相似度算法，减少所有类别都接近 1.00 的问题。

## 安装依赖

在 Anaconda 环境中：

```bat
conda activate ask2know
cd D:\ask2know_framework\a2k_v0.3.1
pip install -r requirements.txt
```

## 创建一个外部水果任务

比如你要创建 5 类水果任务：apple、banana、pear、grape、orange。

```bat
cd D:\ask2know_framework\a2k_v0.3.1
python scripts\init_task.py --name fruit_task --classes apple banana pear grape orange --output D:\a2k_test
```

它会自动生成：

```text
D:\a2k_test\fruit_task\
├─ configs\
│  └─ task_config.yaml
├─ datasets\
│  ├─ objects.json
│  ├─ concepts.json
│  ├─ train\
│  │  ├─ apple\
│  │  ├─ banana\
│  │  ├─ pear\
│  │  ├─ grape\
│  │  └─ orange\
│  └─ unlabeled\
├─ outputs\
├─ experience\
├─ logs\
└─ sample_pools\
   ├─ confirmed\
   ├─ candidate\
   └─ rejected\
```

## 放图片

已知训练样本放入：

```text
D:\a2k_test\fruit_task\datasets\train\apple\
D:\a2k_test\fruit_task\datasets\train\banana\
D:\a2k_test\fruit_task\datasets\train\pear\
D:\a2k_test\fruit_task\datasets\train\grape\
D:\a2k_test\fruit_task\datasets\train\orange\
```

未知图片放入：

```text
D:\a2k_test\fruit_task\datasets\unlabeled\
```

图片名暂时可以随便叫，例如 `apple_001.jpg`、`img_001.jpg`，只要后缀是 jpg、png、jpeg、bmp、webp 即可。

## 开始运行

```bat
cd D:\ask2know_framework\a2k_v0.3.1
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml
```

如果不想每次自动弹出图片：

```bat
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml --no-preview
```

## 运行时怎么回答

如果系统整体分不清，会先让你直接选择真实类别：

```text
1. apple
2. banana
3. pear
4. grape
5. orange
N. 新类别
R. 不适合学习
S. 跳过
```

这是 v0.3.1 的关键改动：当 top5 分数都很接近时，不再强行问 “orange 和 banana 怎么区分”，而是先让用户确认真实类别。

如果系统只是 top1/top2 不确定，它会继续进入主动提问，例如颜色、轮廓、纹理、样本质量等问题。

## 样本池说明

- `confirmed`：用户确认过的样本，系统可以用它更新当前运行中的原型。
- `candidate`：系统猜测但未确认的样本，暂不正式学习。
- `rejected`：模糊、背景复杂、主体不明显、不适合学习的样本。

这一步是为了避免自学习把错误样本加入正式知识库。

## 输出文件

运行结束后会生成：

```text
D:\a2k_test\fruit_task\outputs\
├─ feature_weights.json
├─ question_weights.json
├─ prototype_model.json
├─ objects_runtime.json
├─ experience_report.json
└─ logs\demo_log.json
```

其中 `experience_report.json` 记录本次学到了什么、哪些样本整体不确定、哪些进入了 confirmed/candidate/rejected。

## 当前版本局限

v0.3.1 仍然不是强识别模型。它只是把 v0.3.0 的真实使用流程修得更安全：不乱问、不乱学、自动建任务。真实图片背景复杂时仍可能识别不准，后面版本需要继续加入主体裁剪、简单窗口、未知类别管理和更强特征。
