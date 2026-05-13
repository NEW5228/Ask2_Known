# Ask2Know / a2k_v0.3.4

Ask2Know，简称 a2k，是一个面向个人和小团队的低样本主动教学训练框架。

v0.3.4 的重点不是把识别率一下子拉满，而是让系统：

```text
更会问
更会总结
更稳定运行
为后续概念层和爬虫采集预留接口
```

## 本版新增

```text
1. 默认关闭图片弹窗，避免卡死。
2. 禁用默认 GrabCut，使用更稳定的前景粗提取。
3. 错因经验参与后续问题选择。
4. 新增自我总结 experience_summary.json。
5. 新增温和数据增强，用少量样本生成轻量变化视图。
6. 预留 concepts/ 和 external_data/，方便后续加入概念层与爬虫模块。
```

## 安装

```bat
conda activate ask2know
cd D:\ask2know_framework\a2k_v0.3.4
pip install -r requirements.txt
```

## 创建任务

例如创建 5 类水果任务：

```bat
python scripts\init_task.py --name fruit_task --classes apple banana pear grape orange --output D:\a2k_test
```

它会生成：

```text
D:\a2k_test\fruit_task\
├─ configs\task_config.yaml
├─ datasets\objects.json
├─ datasets\concepts.json
├─ datasets\train\apple
├─ datasets\train\banana
├─ datasets\train\pear
├─ datasets\train\grape
├─ datasets\train\orange
├─ datasets\unlabeled
├─ metadata
├─ outputs
└─ sample_pools
```

## 放图片

已知样本放入：

```text
D:\a2k_test\fruit_task\datasets\train\类别名\
```

未标注图片放入：

```text
D:\a2k_test\fruit_task\datasets\unlabeled\
```

unlabeled 里的图片可以任意命名，运行时会自动规范为 `img_001.jpg` 这种格式。

## 运行

默认不弹窗：

```bat
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml
```

如需预览图片：

```bat
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml --preview
```

## 输出重点

```text
metadata/pairwise_experience.json       类别对错因经验
metadata/experience_summary.json        自我总结
outputs/experience_summary_runtime.json 本轮自我总结副本
outputs/experience_report.json          本轮运行报告
datasets/train/class/class_编号.jpg      confirmed 后的长期训练样本
```

## 关于爬虫

v0.3.4 不实现爬虫，只预留 `ask2know/external_data/crawler_stub.py`。

未来原则：网络图片只能进入 `external_candidate`，不能直接进入 confirmed，必须经过筛选或用户确认。

## 关于概念层

v0.3.4 不实现真正概念学习，只预留 `ask2know/concepts/visual_concepts.py`。

未来概念层会尝试沉淀：

```text
round
elongated
clustered
symmetric
repeated_round
```

这些全局视觉概念应该跨任务复用，而不是每个任务重新学习。
