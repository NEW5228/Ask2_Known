# Ask2Know / a2k 更新记录

## a2k_v0.3.4

版本类型：小版本增强 / 稳定性修复。

核心目标：让系统更会问、更会总结，并为后续概念层和外部数据采集留下接口痕迹。

### 修复
- 默认关闭图片弹窗预览，避免 Windows 图片查看器占用文件导致程序卡死。
- `--preview` 才会打开图片预览；旧的 `--no-preview` 参数仍保留兼容。
- 禁用默认 GrabCut 前景提取，改用稳定的 fallback 前景粗提取，减少真实图片卡死风险。

### 增强
- 历史类别对经验会更强地影响下一次问题选择。
- 如果 `apple_vs_grape` 等类别对已经记录过有用错因，后续会优先生成更具体的问题。
- 新增弱自我总结模块 `metadata/experience_summary.json`。
- 运行结束会输出 `outputs/experience_summary_runtime.json`。
- 新增轻量数据增强配置，训练原型时可使用亮度、旋转、轻裁剪等温和增强视图。

### 保留痕迹 / 后续接口
- 新增 `ask2know/concepts/visual_concepts.py`，预留未来全局视觉概念层。
- 新增 `ask2know/external_data/crawler_stub.py`，预留未来爬虫/外部候选样本采集。
- 明确外部图片未来必须进入 `external_candidate`，不能直接进入 confirmed。

### 暂不做
- 不接入 CLIP / ResNet / MobileNet。
- 不加入真正爬虫。
- 不加入完整概念学习。
- 不加入 tkinter 界面。

## a2k_v0.3.3
- 新增错误后追问。
- 新增类别对经验 `pairwise_experience.json`。

## a2k_v0.3.2
- 新增持续学习数据流。
- confirmed 样本自动进入长期训练库。
- unlabeled 处理后移出。
- 自动重命名与 dataset_index。
