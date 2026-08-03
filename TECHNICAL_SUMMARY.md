# 技术总结 · Binary Classifier

> 展示态成果总结。教学向理解见 [docs/](docs/00-README.md)。
> 项目定位：M1C2——用 MiniTorch 和 PyTorch 双框架做二分类，打通数据→训练→部署全流程。

## 一句话

用 MiniTorch（手写框架）和 PyTorch（成熟框架）双框架做二分类，**同数据同架构对比**，看清框架差距与联系。从猫狗基准到奶龙塔菲趣味应用，验证迁移学习强大效果。

## 核心成果

| 场景 | 数据 | 方法 | 结果 |
|---|---|---|---|
| 猫狗基准 | 公开猫狗数据 | 双框架训练 | 迁移学习 **89.5%** |
| 奶龙 vs 塔菲 | 奶龙300 + 塔菲40 | 微调 ResNet18 | **100%**（2 epochs）|

## 架构设计

```
binary-classifier/
├── docs/            # 5篇教学文档（原理/数据/框架对比/迁移学习）
├── scripts/         # 数据下载/增强/训练/预测
│   ├── download_data.py      # 数据获取
│   ├── train_pytorch.py      # PyTorch 训练
│   ├── train_minitorch.py    # MiniTorch 训练
│   ├── train_pytorch_transfer.py  # 迁移学习
│   └── predict.py            # 推理预测
├── pyproject.toml
└── LICENSE
```

## 技术亮点

1. **双框架对比**：MiniTorch（手写）vs PyTorch（成熟），同架构直接对比
2. **迁移学习**：微调 ResNet18，小数据（300张）达 100%
3. **趣味应用**：奶龙 vs 塔菲，让技术学习有动力、可展示
4. **全流程**：数据下载→增强→训练→预测（闭环意识）

## 方法与能力

| 能力 | 体现 |
|---|---|
| 数据管线 | 下载/增强/处理完整流程 |
| 框架运用 | PyTorch + 手写 MiniTorch 双框架 |
| 迁移学习 | 预训练模型微调 |
| 对比分析 | 双框架同架构效果对比 |

## 与学习路径的衔接

- 前置：M1C MiniTorch（手写框架）
- 本模块：用框架做真实二分类任务（第一次"应用"）
- 后置：M2 YOLO（从分类到检测）
- 统一方法论：**数据→训练→评估→部署闭环**

## 复现

```bash
# 猫狗数据下载
python scripts/download_data.py

# PyTorch 迁移学习（猫狗）
python scripts/train_pytorch_transfer.py

# 奶龙 vs 塔菲
python scripts/download_nailong.py   # 下载数据
python scripts/train_pytorch_transfer.py --data 奶龙塔菲

# 推理
python scripts/predict.py --image 奶龙.jpg --classes 奶龙 塔菲
```
