# a2k_v0.3.1 操作流程

## 1. 进入框架目录

```bat
conda activate ask2know
cd D:\ask2know_framework\a2k_v0.3.1
pip install -r requirements.txt
```

## 2. 自动创建外部任务

```bat
python scripts\init_task.py --name fruit_task --classes apple banana pear grape orange --output D:\a2k_test
```

不要自己手写 config、objects.json、concepts.json。v0.3.1 会自动创建。

## 3. 放图片

训练图片：

```text
D:\a2k_test\fruit_task\datasets\train\apple\
D:\a2k_test\fruit_task\datasets\train\banana\
D:\a2k_test\fruit_task\datasets\train\pear\
D:\a2k_test\fruit_task\datasets\train\grape\
D:\a2k_test\fruit_task\datasets\train\orange\
```

待识别图片：

```text
D:\a2k_test\fruit_task\datasets\unlabeled\
```

## 4. 运行

```bat
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml
```

不弹图：

```bat
python run_demo.py --config D:\a2k_test\fruit_task\configs\task_config.yaml --no-preview
```

## 5. 回答方式

如果系统整体分不清，会让你选真实类别。选类别后，样本进入 confirmed 样本池，并在本次运行中更新原型。

如果系统预测有一定把握，会问你：

```text
Y 正确，加入 confirmed
N 错误，选择正确类别
C 暂时放 candidate
R 拒绝样本
S 跳过
```

## 6. 看结果

```text
D:\a2k_test\fruit_task\outputs\experience_report.json
D:\a2k_test\fruit_task\outputs\logs\demo_log.json
D:\a2k_test\fruit_task\sample_pools\confirmed
D:\a2k_test\fruit_task\sample_pools\candidate
D:\a2k_test\fruit_task\sample_pools\rejected
```

## 7. 这版不是最终效果

v0.3.1 主要修正真实图片流程，不保证识别准确率已经很好。如果仍然效果差，下一步应继续改：主体裁剪、背景风险判断、简单 tkinter 导入窗口和更强深度特征。
