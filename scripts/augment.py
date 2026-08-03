"""
NumPy data augmentation for the handwritten MiniTorch pipeline.

These are hand-written equivalents of common torchvision.transforms,
implemented with NumPy so the pure-NumPy framework can benefit from
data augmentation too (which is essential for small datasets).

Each augmentation is a pure function on an image array (H, W, C) in [0,1].
"""
import numpy as np


def random_hflip(img, p=0.5):
    """Random horizontal flip (prob p)."""
    if np.random.rand() < p:
        return img[:, ::-1, :]
    return img


def random_crop(img, crop_size, padding=4):
    """Random crop with padding (like torchvision RandomCrop)."""
    H, W, C = img.shape
    pad = padding
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    crop_h, crop_w = crop_size, crop_size
    top = np.random.randint(0, 2 * pad + 1)
    left = np.random.randint(0, 2 * pad + 1)
    return padded[top:top + crop_h, left:left + crop_w, :]


def random_brightness(img, max_delta=0.2):
    """Random brightness jitter in [-max_delta, max_delta]."""
    delta = np.random.uniform(-max_delta, max_delta)
    return np.clip(img + delta, 0, 1)


def random_contrast(img, lo=0.8, hi=1.2):
    """Random contrast scale in [lo, hi]."""
    factor = np.random.uniform(lo, hi)
    mean = img.mean(axis=(0, 1), keepdims=True)
    return np.clip((img - mean) * factor + mean, 0, 1)


def random_rotation90(img, p=0.5):
    """Random 90-degree rotation (for rotation-invariant classes)."""
    if np.random.rand() < p:
        k = np.random.randint(1, 4)
        return np.rot90(img, k, axes=(0, 1))
    return img


def compose(img):
    """Standard augmentation pipeline (applied at training time).

    Mirrors a reasonable torchvision Compose for small-image classification.
    """
    img = random_hflip(img)
    img = random_crop(img, crop_size=img.shape[0], padding=4)
    img = random_brightness(img)
    img = random_contrast(img)
    return img


def augment_batch(images):
    """Augment a batch of images (N, H, W, C) in place-safe way.

    Returns a new array with each image independently augmented.
    """
    out = np.empty_like(images)
    for i in range(len(images)):
        out[i] = compose(images[i])
    return out
