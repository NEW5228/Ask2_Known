# Changelog

## a2k_v0.3.1

小版本修正，重点修真实图片流程，不追求一次性大幅提高识别能力。

### Added
- `scripts/init_task.py --output`：支持在外部目录自动创建完整任务项目。
- 自动生成 `task_config.yaml`、`objects.json`、`concepts.json`。
- 新增 `sample_pools/confirmed`、`sample_pools/candidate`、`sample_pools/rejected`。
- 新增整体不确定判断：多个类别分数过于接近时，先询问真实类别，而不是继续问错误 top1/top2 的差异。
- 新增 `objects_runtime.json`，记录运行中新加入的类别。
- 新增 `docs/operation_guide_v0_3_1.md`。

### Changed
- 初始权重调整：降低 `size` 默认权重，避免单张图片里的像素大小误导。
- 改进基础特征提取和相似度计算，减少 size/contour/texture 全部接近 1.00 的情况。
- 用户确认逻辑改为 confirmed/candidate/rejected/skip，降低样本污染风险。

### Known limits
- 仍然不是强识别模型，真实复杂背景图片可能识别不准。
- 还没有 tkinter 图形窗口。
- 还没有自动主体裁剪的交互确认。
- 还没有 pip 正式安装和 `a2k` 命令行入口。

## a2k_v0.3.0
- 通用任务配置雏形。
- 支持自定义类别。
- 支持证据驱动问题生成。
