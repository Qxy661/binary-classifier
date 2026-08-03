"""
From-scratch CNN designed for 奶龙 vs 塔菲 (nailong vs taffy) — 2-class.

This demonstrates the FULL design process: we design the network by
answering "what should each layer do", write it in PyTorch, and train
it to verify the design actually works.

Design logic (each layer's reason):
  - Input is 3-channel 64x64 image  -> start with Conv (not Linear)
  - Extract features with 3 blocks  -> Conv3x3 + BN + ReLU + MaxPool
  - Channel count grows 32->64->128 -> shallow=edges, deep=parts
  - Flatten then FC                -> turn features into a decision
  - Dropout before final FC        -> small data, prevent overfit
  - Output 2 logits                -> binary classification

Usage:
    python scripts/train_nailong_scratch.py --data-dir data --epochs 30
"""
import argparse
import importlib.util
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load data_pipeline (avoid 'scripts' ROS clash)
_dp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data_pipeline.py")
_spec = importlib.util.spec_from_file_location("data_pipeline", _dp_path)
_dp = importlib.util.module_from_spec(_spec)
sys.modules["data_pipeline"] = _dp
_spec.loader.exec_module(_dp)
build_dataset = _dp.build_dataset


class NailongCNN(nn.Module):
    """A from-scratch CNN designed for the nailong/taffy task.

    Design decisions (each justified):
      1. Conv3x3 padding=1: smallest kernel capturing local relations,
         keeps size constant for stacking.
      2. Channels 32->64->128: low-level (edges) -> high-level (parts).
      3. BN after each Conv: stabilize training, faster convergence.
      4. MaxPool 2x2: downsample, halve compute, robust to position.
      5. Flatten -> FC(256): turn spatial features into a decision.
      6. Dropout(0.3): small data (340 images) -> prevent overfitting.
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (3,64,64) -> (32,32,32)
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # Block 2: (32,32,32) -> (64,16,16)
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            # Block 3: (64,16,16) -> (128,8,8)
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ImageDS(Dataset):
    """Wrap numpy (N,H,W,C) images into a torch Dataset."""

    def __init__(self, images, labels):
        self.images = torch.tensor(images.transpose(0, 3, 1, 2),
                                   dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.images[i], self.labels[i]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds, classes = build_dataset(args.data_dir, size=(64, 64))
    print(f"Classes: {classes}")
    train_ds = ImageDS(ds["train"]["images"], ds["train"]["labels"])
    val_ds = ImageDS(ds["val"]["images"], ds["val"]["labels"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = NailongCNN(num_classes=len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: NailongCNN (from-scratch) | {n_params} params")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.time()
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        tr_correct, tr_total, tr_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(y)
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_total += len(y)

        model.eval()
        va_correct, va_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                va_correct += (out.argmax(1) == y).sum().item()
                va_total += len(y)

        tr_acc = tr_correct / tr_total
        va_acc = va_correct / va_total
        best_acc = max(best_acc, va_acc)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch+1:2d}/{args.epochs} | "
                  f"train acc {tr_acc:.3f} | val acc {va_acc:.3f}",
                  flush=True)

    elapsed = time.time() - start
    print(f"\nTraining took {elapsed:.0f}s")
    print(f"Best val accuracy: {best_acc:.3f}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/nailong_scratch.pt")
    print("Model saved to outputs/nailong_scratch.pt")


if __name__ == "__main__":
    main()
