"""
Train a binary classifier with PyTorch — the "concise" counterpart
to train_minitorch.py. Same data, same architecture, for fair comparison.

Usage:
    python scripts/train_pytorch.py --data-dir data/ --epochs 20
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load data_pipeline directly (avoid 'scripts' package clash with ROS)
_dp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data_pipeline.py")
_spec = importlib.util.spec_from_file_location("data_pipeline", _dp_path)
_data_pipeline = importlib.util.module_from_spec(_spec)
sys.modules["data_pipeline"] = _data_pipeline
_spec.loader.exec_module(_data_pipeline)
build_dataset = _data_pipeline.build_dataset


class BinaryCNN(nn.Module):
    """Same architecture as the MiniTorch model, in PyTorch."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.features(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    device = (torch.device("cpu") if args.no_gpu
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    dataset, classes = build_dataset(args.data_dir, size=(64, 64))
    train = dataset["train"]
    val = dataset["val"]

    # Prepare torch tensors
    def to_tensor(images, labels):
        x = torch.tensor(images.transpose(0, 3, 1, 2), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        return x.to(device), y.to(device)

    xtr, ytr = to_tensor(train["images"], train["labels"])
    xva, yva = to_tensor(val["images"], val["labels"])

    model = BinaryCNN().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        # Shuffle and batch
        perm = torch.randperm(len(xtr))
        total_loss, correct = 0.0, 0
        for i in range(0, len(xtr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = xtr[idx], ytr[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(xva)
            va_loss = criterion(logits, yva).item()
            va_preds = (torch.sigmoid(logits) > 0.5).float()
            va_acc = (va_preds == yva).float().mean().item()

        print(f"epoch {epoch+1:2d}/{args.epochs} | "
              f"train loss {total_loss/len(xtr):.3f} "
              f"acc {correct/len(xtr):.3f} | "
              f"val acc {va_acc:.3f}", flush=True)

    elapsed = time.time() - start
    print(f"\nPyTorch training took {elapsed:.0f}s")
    print(f"Final val accuracy: {va_acc:.3f}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/pytorch_binary.pt")
    print("Model saved to outputs/pytorch_binary.pt")


if __name__ == "__main__":
    main()
