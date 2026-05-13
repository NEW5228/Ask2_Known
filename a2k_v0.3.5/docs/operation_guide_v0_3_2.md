# a2k_v0.3.3 操作指南

## 版本定位

a2k_v0.3.3 是持续学习与数据集自动累积版本。

它解决的问题：

1. confirmed 样本不再只保存到 outputs，而是进入长期训练库。
2. unlabeled 图片处理后会被移出，下一轮只处理新图片。
3. confirmed 图片自动按类别顺延命名。
4. metadata 只记录最小必要状态，不保存海量样本列表。

## 推荐工作流

### 第一次

1. 创建任务。
2. 每类放少量种子样本。
3. 放入一批 unlabeled。
4. 运行 a2k。
5. 根据系统提示确认、纠正、拒绝或跳过。

### 第二次以后

1. 继续往 `datasets/unlabeled/` 放新图片。
2. 再次运行同一条命令。
3. 系统自动读取已经积累在 `datasets/train/` 中的样本。
4. 新确认图片继续进入对应类别训练库。

## 命名规则

- 未标注入口：`img_001.jpg`、`img_002.jpg`
- 长期训练库：`apple_001.jpg`、`apple_002.jpg`

用户不需要自己改名，系统会自动处理。

## 文件说明

- `metadata/dataset_index.json`：每类下一个编号和当前数量。
- `metadata/sample_history.jsonl`：样本流转日志。
- `metadata/unlabeled_import_map.jsonl`：unlabeled 导入重命名记录。

## 注意

跳过的图片仍会保留在 `unlabeled`，因为系统不能确定它该去哪里。若不想再处理，请选择 rejected。
