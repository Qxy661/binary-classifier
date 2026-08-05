"""
生成猫狗基准分类效果图（作品集成果图）。

用训练好的 PyTorch 迁移学习模型，对猫狗图片做预测，
生成"图片 + 预测标签 + 置信度"的网格展示图。

Usage:
    python scripts/visualize_catdog.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def load_predict_transfer(image_path, model_path, classes):
    """加载迁移学习模型并预测."""
    from predict import load_image, predict_transfer
    return predict_transfer(str(image_path), str(model_path), classes)


def main():
    model_path = ROOT / "outputs" / "pytorch_transfer.pt"
    classes = ["cat", "dog"]

    # 猫狗测试图（各3张）
    cat_dir = ROOT / "raw" / "cat"
    dog_dir = ROOT / "raw" / "dog"
    test_imgs = []
    for d, label in [(cat_dir, "cat"), (dog_dir, "dog")]:
        imgs = sorted(d.glob("*.jpg"))[:3]
        test_imgs += [(str(p), label) for p in imgs]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for idx, (img_path, true_label) in enumerate(test_imgs):
        ax = axes[idx // 3][idx % 3]
        img = Image.open(img_path)
        ax.imshow(img)

        pred_label, conf = load_predict_transfer(img_path, model_path, classes)
        correct = pred_label == true_label
        color = "green" if correct else "red"
        ax.set_title(f"True: {true_label}\nPred: {pred_label} ({conf:.0%})",
                     fontsize=10, color=color)
        ax.axis("off")

    fig.suptitle("Cat vs Dog Binary Classification (ResNet18 transfer)",
                 fontsize=14)
    plt.tight_layout()
    out = ROOT / "outputs" / "catdog_results.png"
    plt.savefig(out, dpi=150)
    print(f"成果图已保存: {out}")


if __name__ == "__main__":
    main()
