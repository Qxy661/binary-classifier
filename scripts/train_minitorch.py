"""
Train a binary classifier (cat vs dog / milk-dragon vs taffy) with MiniTorch.

Uses our handwritten NumPy framework (from the minitorch project).
Binary classification = 1 logit + sigmoid + binary cross-entropy.

Usage:
    python scripts/train_minitorch.py --data-dir data/ --epochs 20
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse MiniTorch framework
sys.path.insert(0, "/root/projects/minitorch")
from minitorch.layers import (Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d,
                              Flatten, Linear)
from minitorch.optim import Adam

# Load data_pipeline directly (avoid 'scripts' package clash with ROS)
_dp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data_pipeline.py")
_spec = importlib.util.spec_from_file_location("data_pipeline", _dp_path)
_data_pipeline = importlib.util.module_from_spec(_spec)
sys.modules["data_pipeline"] = _data_pipeline
_spec.loader.exec_module(_data_pipeline)
build_dataset = _data_pipeline.build_dataset


def sigmoid(z):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def bce_loss(logits, labels):
    """Binary cross-entropy. labels in {0, 1}."""
    p = sigmoid(logits)
    # BCE = -[y*log(p) + (1-y)*log(1-p)]
    loss = -np.mean(labels * np.log(p + 1e-8)
                    + (1 - labels) * np.log(1 - p + 1e-8))
    return float(loss)


def bce_backward(logits, labels):
    """d(loss)/d(logits) = p - y  (the famous sigmoid+BCE simplification)."""
    p = sigmoid(logits)
    return (p - labels) / labels.size


def build_model():
    """Small CNN for 64x64 binary classification."""
    return Sequential(
        Conv2d(3, 16, k=3), BatchNorm2d(16), ReLU(), MaxPool2d(),   # 32x32
        Conv2d(16, 32, k=3), BatchNorm2d(32), ReLU(), MaxPool2d(),  # 16x16
        Conv2d(32, 64, k=3), BatchNorm2d(64), ReLU(), MaxPool2d(),  # 8x8
        Flatten(),
        Linear(64 * 8 * 8, 64), ReLU(),
        Linear(64, 1),
    )


def train_one_epoch(model, images, labels, batch_size, optimizer, augment=None):
    """One pass over data in mini-batches. augment: optional batch augmentation."""
    n = len(labels)
    idx = np.random.permutation(n)
    total_loss, correct = 0.0, 0

    for start in range(0, n, batch_size):
        batch = idx[start:start + batch_size]
        xb = images[batch]
        yb = labels[batch].astype(np.float64).reshape(-1, 1)

        # Apply data augmentation (on (N,H,W,C), before transpose to CHW)
        if augment is not None:
            # images are (N,C,H,W); augment expects (N,H,W,C)
            xb = augment(xb.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)

        model.zero_grad()
        logits = model.forward(xb)
        loss = bce_loss(logits, yb)

        dlogits = bce_backward(logits, yb)
        model.backward(dlogits)
        optimizer.step()

        total_loss += loss * len(batch)
        preds = (sigmoid(logits) > 0.5).astype(int).flatten()
        correct += np.sum(preds == labels[batch])

    return total_loss / n, correct / n


def evaluate(model, images, labels, batch_size=64):
    """Evaluate without updating params."""
    n = len(labels)
    total_loss, correct = 0.0, 0
    for start in range(0, n, batch_size):
        xb = images[start:start + batch_size]
        yb = labels[start:start + batch_size].astype(np.float64).reshape(-1, 1)
        logits = model.forward(xb)
        total_loss += bce_loss(logits, yb) * len(xb)
        preds = (sigmoid(logits) > 0.5).astype(int).flatten()
        correct += np.sum(preds == labels[start:start + batch_size])
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print("Loading dataset...")
    dataset, classes = build_dataset(args.data_dir, size=(64, 64))
    train = dataset["train"]
    val = dataset["val"]
    print(f"Classes: {classes}")

    # MiniTorch conv expects (N, C, H, W); our data pipeline returns (N, H, W, C)
    train["images"] = train["images"].transpose(0, 3, 1, 2)
    val["images"] = val["images"].transpose(0, 3, 1, 2)

    model = build_model()
    print(f"Model params: {sum(p.data.size for p in model.parameters())}")

    # Data augmentation (NumPy, research-recommended: moderate)
    import importlib.util as _ilu
    _aug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "augment.py")
    _aug_spec = _ilu.spec_from_file_location("augment", _aug_path)
    _aug_mod = _ilu.module_from_spec(_aug_spec)
    sys.modules["augment"] = _aug_mod
    _aug_spec.loader.exec_module(_aug_mod)
    augment = _aug_mod.augment_batch
    print("Data augmentation: ON (hflip / random-crop / brightness / contrast)")

    optimizer = Adam(model.parameters(), lr=args.lr)
    start = time.time()

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train["images"],
                                          train["labels"], args.batch_size,
                                          optimizer, augment=augment)
        va_loss, va_acc = evaluate(model, val["images"], val["labels"])
        print(f"epoch {epoch+1:2d}/{args.epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val acc {va_acc:.3f}", flush=True)

    elapsed = time.time() - start
    print(f"\nMiniTorch training took {elapsed:.0f}s")
    print(f"Final val accuracy: {va_acc:.3f}")

    # Save model params
    os.makedirs("outputs", exist_ok=True)
    save_dict = {f"param_{i}": p.data for i, p in enumerate(model.parameters())}
    np.savez("outputs/minitorch_binary.npz", **save_dict)
    print("Model saved to outputs/minitorch_binary.npz")


if __name__ == "__main__":
    main()
