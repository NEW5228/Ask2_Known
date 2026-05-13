# Ask2Know 问知自适应学习框架 v0.1

第一版目标：用少量已知水果图片建立对象原型，对未知图片进行识别；当系统不确定时主动提问，根据用户反馈调整特征权重并重新判断。

## 快速开始

```bash
cd ask2know_framework
pip install -r requirements.txt
python scripts/create_demo_dataset.py
python run_demo.py --config configs/fruit_demo.yaml
```

也可以换成自己的图片：

```text
datasets/fruit_demo/train/apple/        放已知苹果图片
datasets/fruit_demo/train/strawberry/   放已知草莓图片
datasets/fruit_demo/unlabeled/          放待识别图片
```

## 第一版核心闭环

读取少量样本 → 提取颜色/尺寸/轮廓/纹理特征 → 建立对象原型 → 识别未知样本 → 不确定时主动提问 → 用户回答 → 更新特征权重 → 重新识别 → 保存日志和结果。
