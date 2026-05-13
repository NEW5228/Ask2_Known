# Changelog

## a2k_v0.3.2

持续学习与数据集自动累积补丁版。

### 新增
- confirmed 样本会直接移动并重命名进入 `datasets/train/<class>/`，下一轮自动参与学习。
- `datasets/unlabeled/` 作为临时入口；处理后的图片会被移动到 train / candidate / rejected / unknown，不再反复处理旧图。
- 未标注图片导入时自动规范命名为 `img_001.jpg`、`img_002.jpg` 等，降低中文名、空格、特殊符号带来的读取风险。
- 新增 `metadata/dataset_index.json`，只保存每个类别的 `next_id` 和 `count`。
- 新增 `metadata/sample_history.jsonl`，用追加日志记录样本流转。
- 新增 `metadata/unlabeled_import_map.jsonl`，记录未标注图片导入重命名过程。
- `init_task.py` 自动生成 metadata 目录和初始索引文件。

### 修正
- 不再把 confirmed 样本只复制到 outputs/sample_pools；确认样本现在会成为长期训练库的一部分。
- 减少下一轮运行重复询问旧 unlabeled 图片的问题。

### 仍未完成
- 还没有图形窗口。
- 还没有深度特征、主体裁剪、数据增强。
- 识别效果仍依赖当前浅层特征，重点是先让数据持续积累流程跑通。

## a2k_v0.3.1

- 自动创建外部任务目录、配置文件和 objects/concepts 文件。
- 加入 confirmed / candidate / rejected 样本池雏形。
- 加入整体不确定判断，避免候选错误时乱问 top1/top2。
