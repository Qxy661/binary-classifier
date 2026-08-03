# 二分类趣味教学项目：奶龙 vs 塔菲 🐉🦝

> **用 MiniTorch 和 PyTorch 双框架，训练一个"识别奶龙还是塔菲"的二分类器。**
> 先用猫狗公开数据跑通流程，再上趣味数据。MiniTorch 是手写框架，PyTorch 是成熟框架——**同数据同架构对比，看清两者的差距与联系。**

## 为什么做这个

1. **趣味**：奶龙/塔菲是火的表情包形象，分类它们让人有动力
2. **完整闭环**：数据管线 → 训练 → 评估 → 识别，二分类是最基础完整的 CV 任务
3. **框架对比**：MiniTorch（手写 NumPy）vs PyTorch，同架构同数据，效果/速度差距一目了然
4. **能力验证**：MiniTorch 在 CIFAR-10 之外的真实任务上证明自己

## 项目结构

```
binary-classifier/
├── scripts/
│   ├── data_pipeline.py    # 数据管线：组织/加载/划分
│   ├── train_minitorch.py  # MiniTorch 训练（手写框架）
│   ├── train_pytorch.py    # PyTorch 训练（成熟框架）
│   └── predict.py          # 识别脚本（传图判断）
├── data/                   # 数据（train/val 按类别组织）
├── docs/                   # 教学文档
├── outputs/                # 模型 + 可视化
└── README.md
```

## 快速开始

```bash
conda activate dl
pip install -e .

# 1. 准备数据（猫狗公开数据 or 奶龙/塔菲图片）
#    原始图片放 raw/<类别>/ 下
python scripts/data_pipeline.py --source raw/ --data-dir data/

# 2. 训练（两个框架都训，对比）
python scripts/train_minitorch.py --data-dir data/ --epochs 20
python scripts/train_pytorch.py --data-dir data/ --epochs 20

# 3. 识别
python scripts/predict.py --image 奶龙.jpg --classes 奶龙 塔菲
python scripts/predict.py --image img.jpg --framework pytorch
```

## 设计要点

### 二分类的数学
- 输出 **1 个 logit**，过 **sigmoid** 得概率
- **Binary Cross-Entropy (BCE)** 损失
- 梯度简化：`d(loss)/d(logits) = p - y`（著名公式）

### 框架对比
| | MiniTorch | PyTorch |
|---|---|---|
| 实现 | 手写 NumPy (~700行框架) | 成熟框架 |
| 训练 | CPU，慢 | GPU 快 |
| 效果 | 看数据量 | 通常更好 |
| 教学 | 理解原理 | 高效实用 |

## 📊 结果

### 🎉 奶龙 vs 塔菲（趣味应用，终极形态）

| 项 | 结果 |
|---|---|
| 数据 | 奶龙 300 张（HuggingFace 官方）+ 塔菲 40 张（Bing）|
| 训练 | 迁移学习（微调 ResNet18），2 epochs 到 100% |
| **验证准确率** | **1.000（100%）** |
| 识别验证 | 奶龙→milkdragon、塔菲→taffy，全部高置信 |

### 猫狗基准（流程验证）

**同条件对比（从零、无增强、20 epochs）：MiniTorch 0.543 vs PyTorch 0.500** —— 手写框架从零不输成熟框架！

| 方案 | val acc | 说明 |
|---|---|---|
| MiniTorch 从零 | 0.543 | 手写框架，原理完整 |
| PyTorch 从零（无增强）| 0.500 | 同条件基线 |
| PyTorch 从零 + 增强 | 0.692 | 数据增强提升 |
| **PyTorch 微调 ResNet18** | **0.895** | **最优（迁移学习，生态优势）**|

**两个核心洞察**：
1. **MiniTorch 从零实现不输 PyTorch**——原理是对的，价值在理解
2. **真正的差距在生态**——预训练模型（迁移学习）+ 增强库，是手写框架无法比的

详见 [docs/03-框架对比.md](docs/03-框架对比.md) 和 [docs/04-迁移学习.md](docs/04-迁移学习.md)。

## 数据说明

- **阶段1**：猫狗公开数据（先跑通流程）
- **阶段2**：奶龙/塔菲图片（趣味应用）
  - 需要网络下载或手动收集，各 100-200 张
  - 放 `raw/奶龙/`、`raw/塔菲/`

## 验收标准

1. 猫狗二分类：MiniTorch + PyTorch 都训练成功并对比
2. 识别脚本能对单张图判断类别
3. 奶龙/塔菲数据可用，模型能区分
4. 教学文档完整

## License

MIT © Qxy661
