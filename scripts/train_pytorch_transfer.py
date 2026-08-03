"""
PyTorch transfer learning for the binary classifier — best accuracy path.

Follows the PyTorch official transfer-learning tutorial best practices
(research-backed):
  - frozen ResNet18 backbone (ImageNet pretrained) + new linear head
  - moderate data augmentation (flip / random crop / rotation / color)
  - early stopping on val loss
  - OneCycleLR learning-rate schedule

Reference: pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

Usage:
    python scripts/train_pytorch_transfer.py --data-dir data/ --epochs 20
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load data_pipeline directly (avoid 'scripts' clash with ROS)
import importlib.util
_dp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data_pipeline.py")
_spec = importlib.util.spec_from_file_location("data_pipeline", _dp_path)
_data_pipeline = importlib.util.module_from_spec(_spec)
sys.modules["data_pipeline"] = _data_pipeline
_spec.loader.exec_module(_data_pipeline)
build_dataset = _data_pipeline.build_dataset


class ImageDataset(Dataset):
    """Wrap numpy arrays into a torch Dataset with transforms."""

    def __init__(self, images, labels, transform=None):
        self.images = images  # (N, H, W, C) in [0, 1]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = (self.images[idx] * 255.0).astype(np.uint8)  # back to [0,255]
        from PIL import Image
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_transforms():
    """Moderate augmentation for training (research: don't overdo it),
    and no augmentation for validation (or metrics are skewed)."""
    train_tf = transforms.Compose([
        transforms.Resize(80),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_transfer_model(finetune_layers=2):
    """ResNet18 with frozen backbone + optional fine-tuning of last layers.

    Args:
        finetune_layers: how many of the last residual blocks to unfreeze
            (0 = fully frozen feature extraction; 2 = fine-tune last 2 blocks
            + head, the research-recommended sweet spot for small data).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze all by default
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last `finetune_layers` residual blocks (layer4, layer3...)
    blocks = [model.layer4, model.layer3, model.layer2, model.layer1]
    for i in range(min(finetune_layers, len(blocks))):
        for param in blocks[i].parameters():
            param.requires_grad = True

    # Replace head (always trainable)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def train_model(model, dataloaders, criterion, optimizer, scheduler,
                num_epochs, device):
    """Train with early stopping on val loss. Returns best model + history."""
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience, trigger = 8, 0
    history = {"train_acc": [], "val_acc": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for inputs, labels in dataloaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * inputs.size(0)
            tr_correct += (outputs.argmax(1) == labels).sum().item()
            tr_total += labels.size(0)

        # Validate
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                va_loss += loss.item() * inputs.size(0)
                va_correct += (outputs.argmax(1) == labels).sum().item()
                va_total += labels.size(0)

        tr_acc = tr_correct / tr_total
        va_acc = va_correct / va_total
        va_loss /= va_total
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["val_loss"].append(va_loss)

        # Scheduler (OneCycle steps each batch, so step here once per epoch
        # for ReduceLROnPlateau; for OneCycleLR call inside batch loop instead)
        if scheduler is not None and hasattr(scheduler, "step_per_epoch"):
            scheduler.step_per_epoch()

        print(f"epoch {epoch+1:2d}/{num_epochs} | train acc {tr_acc:.3f} | "
              f"val acc {va_acc:.3f} | val loss {va_loss:.3f}", flush=True)

        # Early stopping on val loss
        if va_loss < best_loss:
            best_loss = va_loss
            best_acc = va_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            trigger = 0
        else:
            trigger += 1
            if trigger >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_wts)
    return model, best_acc, history


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
    train_tf, val_tf = get_transforms()

    train_ds = ImageDataset(dataset["train"]["images"],
                            dataset["train"]["labels"], train_tf)
    val_ds = ImageDataset(dataset["val"]["images"],
                          dataset["val"]["labels"], val_tf)

    dataloaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=0),
    }

    finetune_layers = 2
    model = build_transfer_model(finetune_layers=finetune_layers).to(device)

    # Discriminative learning rates (fastai trick): low lr for fine-tuned
    # backbone (don't destroy pretrained features), higher lr for the head.
    backbone_lr = args.lr / 10
    head_lr = args.lr
    params = [
        {"params": [p for p in model.layer4.parameters() if p.requires_grad],
         "lr": backbone_lr},
        {"params": [p for p in model.layer3.parameters() if p.requires_grad],
         "lr": backbone_lr},
        {"params": model.fc.parameters(), "lr": head_lr},
    ]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training ResNet18 (fine-tune last {finetune_layers} blocks, "
          f"backbone lr={backbone_lr:.5f}, head lr={head_lr:.4f}, "
          f"{args.epochs} epochs)...")
    start = time.time()
    model, best_acc, history = train_model(
        model, dataloaders, criterion, optimizer, None,
        args.epochs, device)
    elapsed = time.time() - start

    print(f"\nTransfer learning took {elapsed:.0f}s")
    print(f"Best val accuracy: {best_acc:.3f}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/pytorch_transfer.pt")
    print("Model saved to outputs/pytorch_transfer.pt")


if __name__ == "__main__":
    main()
