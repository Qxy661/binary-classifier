# 教学文档导航

> MiniTorch 框架的趣味应用项目。这套文档带你从"二分类原理"到"框架对比"。

## 文档目录

| 章节 | 内容 | 配套代码 |
|---|---|---|
| [01 · 二分类原理](01-二分类原理.md) | sigmoid / BCE / p-y 梯度简化 | `scripts/train_minitorch.py` |
| [02 · 数据管线](02-数据管线.md) | 数据采集/组织/加载 | `scripts/data_pipeline.py` |
| [03 · 框架对比](03-框架对比.md) | MiniTorch vs PyTorch 实测 | `scripts/train_pytorch.py` |
| [04 · 迁移学习](04-迁移学习.md) | 小数据最优解，89.5% 实战 | `scripts/train_pytorch_transfer.py` |

## 学习路径

1. 读完 **01 二分类原理**（数学基础）
2. 看 **02 数据管线**（真实项目的数据工程）
3. 跑通两个训练脚本，用 **03 框架对比** 理解差距
4. 用 `scripts/predict.py` 识别自己的图片

## 前置

- 熟悉 M1 的 `dl-hands-on`（反向传播）和 `minitorch`（框架）
- Python + NumPy + PyTorch 基础
