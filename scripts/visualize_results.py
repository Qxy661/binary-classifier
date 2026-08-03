"""
生成二分类结果展示图（作品集成果图）。

用训练好的 PyTorch 迁移学习模型，对奶龙/塔菲图片做预测，
生成"图片 + 预测标签 + 置信度"的网格展示图。

Usage:
    python scripts/visualize_results.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def load_predict_transfer(image_path, model_path, classes):
    """加载迁移学习模型并预测（复用 predict.py 逻辑）."""
    from predict import load_image, predict_transfer
    return predict_transfer(str(image_path), str(model_path), classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "outputs" / "pytorch_transfer.pt"))
    parser.add_argument("--classes", nargs=2, default=["milkdragon", "taffy"])
    parser.add_argument("--out", default=str(ROOT / "outputs" / "classification_results.png"))
    args = parser.parse_args()

    # 测试图片（milkdragon/taffy 各3张）
    nailong_dir = ROOT / "raw" / "milkdragon"
    taffy_dir = ROOT / "raw" / "taffy"
    test_imgs = []
    for d, label in [(nailong_dir, args.classes[0]), (taffy_dir, args.classes[1])]:
        imgs = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
        test_imgs += [(str(p), label) for p in imgs[:3]]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for idx, (img_path, true_label) in enumerate(test_imgs[:6]):
        ax = axes[idx // 3][idx % 3]
        img = Image.open(img_path)
        ax.imshow(img)

        # 预测
        pred_label, conf = load_predict_transfer(img_path, args.model, args.classes)
        correct = pred_label == true_label
        color = "green" if correct else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label} ({conf:.0%})",
                     fontsize=10, color=color)
        ax.axis("off")

    fig.suptitle("MilkDragon vs Taffy Binary Classification (ResNet18 transfer)",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"成果图已保存: {args.out}")


if __name__ == "__main__":
    main()
