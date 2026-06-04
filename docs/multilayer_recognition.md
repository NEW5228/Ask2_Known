# 多层识别与多向量分析方向

## 目标

Ask2Know 后续应支持多层识别，而不仅是扁平单标签分类。

多层识别的目标是把一个结果拆成可复用的结构化路径，例如：

```text
traffic_sign
-> speed_limit
-> number_30
-> speed_limit_30
```

这类结构适合交通指示牌、车辆、商品、工业缺陷、票据文档、动植物细分等任务。它可以让一次用户确认同时增强父类、子类、属性和最终类别，从而提高少样本任务的适配速度。

## 核心思想

当前模型主要围绕叶子类别建立原型，例如 `speed_limit_30`。多层识别应把一个类别拆成多个可学习节点：

```text
父类节点: traffic_sign
子类节点: speed_limit
属性节点: red_circle, white_background, number_text, number_30
叶子节点: speed_limit_30
样本节点: confirmed nearest samples
```

最终预测不只看叶子类相似度，而是综合多种证据：

```text
score(speed_limit_30)
= parent score
+ subtype score
+ attribute score
+ leaf prototype score
+ nearest confirmed sample evidence
+ user feedback adjustment
```

## 为什么适合交通指示牌

交通指示牌天然由稳定结构组成：

- 标志大类：禁令、警告、指示、指路
- 几何结构：圆形、三角形、矩形
- 颜色结构：红边、蓝底、黄底、白底
- 符号结构：箭头、禁止斜杠、人行、车辆、施工
- 文本或数字字段：30、40、60、80、P、STOP
- 最终类别：限速30、禁止停车、左转、注意行人

因此它不应该只被建模为几百个互不相关的扁平类别。多层结构可以共享通用知识，例如所有限速牌共享“红圈白底数字”的证据。

## 软树而不是硬树

实现时不应做硬分支：

```text
如果第一层判错，后续只能在错误分支里搜索。
```

更合适的是软树和 top-k 分支保留：

```text
level 1:
- traffic_sign: 0.93
- other_object: 0.12

level 2:
- speed_limit: 0.81
- no_entry: 0.34

level 3:
- number_30: 0.76
- number_80: 0.58

final:
- speed_limit_30: 0.84
- speed_limit_80: 0.68
```

这样前层不确定时不会过早剪掉正确答案。

## 配置方向

建议新增任务级 taxonomy 配置，例如：

```yaml
taxonomy:
  enable: true
  root: traffic_sign
  nodes:
    traffic_sign:
      children: [regulatory, warning, guide]
    regulatory:
      children: [speed_limit, no_entry, no_parking]
    speed_limit:
      attributes:
        shape: circle
        color: red_border_white_center
        field: number
      children: [speed_limit_30, speed_limit_40, speed_limit_60]
```

叶子类别仍可保留在 `classes` 中，以兼容现有训练、样本池和评估流程。

用户应该可以自己维护这棵树。系统提供默认模板，但最终项目里的
`similarity.taxonomy.label_paths` 必须是可编辑配置，而不是写死在代码里。
例如交通牌用户可以把机器生成的 `sign_4_7` 改成更清楚的语义节点：

```yaml
similarity:
  taxonomy:
    enable: true
    levels: [root, family, shape, color, symbol_type, symbol_value, leaf]
    label_paths:
      speed_limit_30:
        - traffic_sign
        - speed_limit
        - circle
        - red_white
        - number
        - number_30
        - speed_limit_30
      keep_right:
        - traffic_sign
        - mandatory_direction
        - circle
        - blue_white
        - arrow
        - arrow_keep_right
        - keep_right
```

这样用户可以按自己的业务理解调整层级，比如商品可以写成
`product -> category -> brand -> series -> flavor -> sku`，工业缺陷可以写成
`part -> region -> defect_type -> severity -> final_label`。

为了保留旧流程，项目还应该提供识别模式开关：

```yaml
similarity:
  recognition_mode: flat        # 旧单层识别，只输出扁平类别
```

或：

```yaml
similarity:
  recognition_mode: multilayer  # 多层识别，启用 taxonomy/path/fine-grained rerank
```

在 `flat` 模式下，taxonomy、reference icon rerank 和 fine-grained rerank 都应关闭，
预测仍按原来的单层 `label` 排序；在 `multilayer` 模式下，系统才输出路径候选和分层证据。

细粒度重排应该只在 top-k 候选内部工作。它不替代 CLIP 主干，而是在同一兄弟组内做对比，
例如 `speed_limit` 的 `number` 组或 `mandatory_direction` 的 `arrow` 组，把
`30/50/80`、`left/right/straight` 这类候选重新排序。

## 模型方向

后续可以新增 `TaxonomyModel` 或在 `PrototypeModel` 外增加一层结构化推理器：

- 为每个 taxonomy node 维护 node prototype
- 为属性维护 attribute prototype
- 为叶子类别维护现有 class prototype
- 用 CLIP embedding 做内部相似度
- 用 OpenCV/concept 特征提供可解释属性证据
- 用 confirmed 样本做 k-NN 近邻证据
- 用用户纠错更新路径权重和节点原型

## 用户反馈方向

用户反馈也应从“这个叶子类对不对”升级成“哪一层错了”：

```text
系统: 这是限速30吗？
用户: 不是，是限速80。

系统应学习:
- traffic_sign 正确
- speed_limit 正确
- number_30 错误
- number_80 正确
- speed_limit_80 叶子类正确
```

主动提问也应支持分层问题：

- 这是交通指示牌还是其他物体？
- 这是限速类还是禁止类？
- 中间数字更像 30 还是 80？
- 这个箭头方向是左转还是右转？

## 演进步骤

1. 保留当前扁平分类能力，作为叶子类预测和兼容层。
2. 增加 taxonomy 配置读取和校验。
3. 为 taxonomy 节点建立父类/子类 prototype。
4. 预测输出增加 `path_candidates`，而不是只输出 `label`。
5. 交通指示牌先作为第一个多层识别任务验证。
6. 增加分层评估指标：父类准确率、子类准确率、字段准确率、叶子类准确率。
7. 将用户纠错记录拆解成路径级反馈，增强跨任务复用能力。

## 准确率目标

多层识别的准确率目标应拆成两个阶段：

- 第一阶段：让叶子类 top-1 达到 80%，同时确保 top-3/top-5 召回高于 90%。这说明正确答案已经进入候选集合，重点优化排序、符号字段和用户提问。
- 第二阶段：把稳定任务的底线提升到 90%。这需要更强的字段识别模块，例如限速数字 OCR、箭头方向识别、警告图案识别，以及路径级用户纠错回放。

如果 top-5 已经接近 90% 但 top-1 低很多，问题通常不在“看不到正确类别”，而在“候选重排不够强”。此时不要盲目增加父类权重，应优先增强细字段识别。

## 设计边界

多层识别不是替代现有主动教学，而是增强它。

Ask2Know 仍应坚持：

- embedding 是内部评分信号
- 用户问题尽量围绕可解释概念
- 新样本进入 confirmed 前需要用户确认
- 不依赖大规模重新训练
- 不把所有任务硬编码成交通牌逻辑
