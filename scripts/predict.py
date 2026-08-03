"""
Predict: classify a single image (or folder) as class A or class B.

Usage:
    python scripts/predict.py --image path/to/img.jpg
    python scripts/predict.py --image path/to/img.jpg --model pytorch_transfer
    python scripts/predict.py --image img.jpg --model minitorch
    python scripts/predict.py --dir path/to/images/   # batch

Models:
    --model minitorch       : outputs/minitorch_binary.npz  (handwritten)
    --model pytorch         : outputs/pytorch_binary.pt     (from-scratch CNN)
    --model pytorch_transfer: outputs/pytorch_transfer.pt   (fine-tuned ResNet18)
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_script(name):
    """Load a sibling script as a module (avoids 'scripts' package clash)."""
    path = os.path.join(_HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_image(path, size=(64, 64)):
    """Load and preprocess a single image -> (1, H, W, C) in [0,1]."""
    img = Image.open(path).convert("RGB").resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr[np.newaxis]


def predict_minitorch(image, model_path, classes):
    """Predict with MiniTorch model (handwritten framework)."""
    sys.path.insert(0, "/root/projects/minitorch")
    tm = _load_script("train_minitorch")

    model = tm.build_model()
    data = np.load(model_path, allow_pickle=True)
    for i, p in enumerate(model.parameters()):
        key = f"param_{i}"
        if key in data:
            p.data[:] = data[key]

    logits = model.forward(image.transpose(0, 3, 1, 2))
    prob = tm.sigmoid(logits).item()
    return classes[1] if prob > 0.5 else classes[0], prob


def predict_pytorch(image, model_path, classes):
    """Predict with PyTorch from-scratch CNN."""
    import torch
    tp = _load_script("train_pytorch")

    model = tp.BinaryCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    x = torch.tensor(image.transpose(0, 3, 1, 2), dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()
    return classes[1] if prob > 0.5 else classes[0], prob


def predict_transfer(image_path, model_path, classes):
    """Predict with fine-tuned ResNet18 (transfer learning)."""
    import torch
    from torchvision import transforms
    tt = _load_script("train_pytorch_transfer")

    model = tt.build_transfer_model(finetune_layers=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # ResNet expects ImageNet-normalized PIL input
    tf = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, 1)[0, 1].item()
    return classes[1] if prob > 0.5 else classes[0], prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="path to a single image")
    parser.add_argument("--dir", help="path to a folder of images")
    parser.add_argument("--model", default="pytorch_transfer",
                        choices=["minitorch", "pytorch", "pytorch_transfer"])
    parser.add_argument("--classes", nargs=2, default=["cat", "dog"])
    args = parser.parse_args()

    images_to_check = []
    if args.image:
        images_to_check = [args.image]
    elif args.dir:
        images_to_check = sorted(
            os.path.join(args.dir, f) for f in os.listdir(args.dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not images_to_check:
        print("No images provided. Use --image or --dir.")
        return

    model_paths = {
        "minitorch": "outputs/minitorch_binary.npz",
        "pytorch": "outputs/pytorch_binary.pt",
        "pytorch_transfer": "outputs/pytorch_transfer.pt",
    }
    fns = {
        "minitorch": predict_minitorch,
        "pytorch": predict_pytorch,
        "pytorch_transfer": predict_transfer,
    }
    model_path = model_paths[args.model]
    predict_fn = fns[args.model]

    print(f"Model: {args.model} | classes: {args.classes[0]} vs {args.classes[1]}")
    for path in images_to_check:
        if args.model == "pytorch_transfer":
            # Transfer model reads PIL directly (no numpy array needed)
            cls, prob = predict_fn(path, model_path, args.classes)
        else:
            img = load_image(path)
            cls, prob = predict_fn(img, model_path, args.classes)
        print(f"  {os.path.basename(path):30s} -> {cls:12s} (p={prob:.3f})")


if __name__ == "__main__":
    main()
