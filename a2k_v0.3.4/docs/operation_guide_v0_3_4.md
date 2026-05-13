# a2k_v0.3.4 操作指南

## 1. 创建任务

```bat
python scripts\init_task.py --name fruit_task --classes apple banana pear grape orange --output D:\a2k_test
```

## 2. 放图片

已知样本：

```text
D:\a2k_test\fruit_task\datasets\train\apple
D:\a2k_test\fruit_task\datasets\train\banana
...
```

未标注图片：

```text
D:\a2k_test\fruit_task\datasets\unlabeled
```

## 3. 运行

默认不弹窗：

```bat
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml
```

需要预览：

```bat
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml --preview
```

## 4. 本版重点观察

```text
1. 是否不再因为弹窗卡死。
2. 错因经验是否会让后续问题更具体。
3. experience_summary.json 是否能总结当前学习经验。
4. 数据增强开启后，少量样本下结果是否更稳定。
```
