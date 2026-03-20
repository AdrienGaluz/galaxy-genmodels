"""
data/dataset.py — Galaxy10 DECals DataLoader
=============================================
Downloads Galaxy10 DECals automatically via astroNN on first run.
Images are 256x256 RGB; we resize to cfg.IMAGE_SIZE (64 or 128).

Classes (10 total):
  0: Disturbed Galaxies
  1: Merging Galaxies
  2: Round Smooth Galaxies
  3: In-between Round Smooth Galaxies
  4: Cigar Shaped Smooth Galaxies
  5: Barred Spiral Galaxies
  6: Unbarred Tight Spiral Galaxies
  7: Unbarred Loose Spiral Galaxies
  8: Edge-on Galaxies without Bulge
  9: Edge-on Galaxies with Bulge
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── Try importing astroNN ──────────────────────────────────────────────────────
try:
    from astroNN.datasets import load_galaxy10
    ASTRONN_AVAILABLE = True
except ImportError:
    ASTRONN_AVAILABLE = False
    print("[dataset] astroNN not found. Run: pip install astroNN")


class Galaxy10Dataset(Dataset):
    """
    PyTorch Dataset wrapping Galaxy10 DECals.
    Handles download, caching, resizing, and normalization.
    """

    def __init__(self, image_size=64, max_samples=None, transform=None):
        self.image_size  = image_size
        self.max_samples = max_samples
        self.transform   = transform or self._default_transform()

        self.images, self.labels = self._load_data()

    # ── Default augmentation pipeline ─────────────────────────────────────────
    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),
                               interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5]),   # → [-1, 1]
        ])

    # ── Load or download data ──────────────────────────────────────────────────
    def _load_data(self):
        cache_path = os.path.join("data", "galaxy10_cache.npz")

        if os.path.exists(cache_path):
            print("[dataset] Loading cached Galaxy10 DECals...")
            cache = np.load(cache_path)
            images, labels = cache["images"], cache["labels"]
        else:
            print("[dataset] Downloading Galaxy10 DECals (first run only ~1.5GB)...")
            if not ASTRONN_AVAILABLE:
                raise RuntimeError("astroNN is required. Install with: pip install astroNN")

            images, labels = load_galaxy10()
            # images: (N, 256, 256, 3) uint8
            # labels: (N,) int

            os.makedirs("data", exist_ok=True)
            np.savez_compressed(cache_path, images=images, labels=labels)
            print(f"[dataset] Saved cache to {cache_path}")

        # Subsample if requested
        if self.max_samples is not None and self.max_samples < len(images):
            idx = np.random.choice(len(images), self.max_samples, replace=False)
            images = images[idx]
            labels = labels[idx]

        print(f"[dataset] Loaded {len(images)} galaxy images "
              f"→ resizing to {self.image_size}x{self.image_size}")
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = self.images[idx]                     # (256, 256, 3) uint8
        label = int(self.labels[idx])

        pil_img = Image.fromarray(img.astype(np.uint8))
        tensor  = self.transform(pil_img)            # (3, H, W) in [-1, 1]
        return tensor, label


# ── Convenience factory ────────────────────────────────────────────────────────
def get_dataloader(image_size=64, batch_size=64, max_samples=None,
                   num_workers=4, shuffle=True):
    """
    Returns a DataLoader ready for training.
    """
    dataset = Galaxy10Dataset(image_size=image_size, max_samples=max_samples)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"[dataset] DataLoader ready: {len(dataset)} samples, "
          f"{len(loader)} batches/epoch (batch_size={batch_size})")
    return loader


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = get_dataloader(image_size=64, batch_size=8, max_samples=100)
    batch, labels = next(iter(loader))
    print(f"Batch shape : {batch.shape}")    # (8, 3, 64, 64)
    print(f"Value range : [{batch.min():.2f}, {batch.max():.2f}]")
    print(f"Labels      : {labels.tolist()}")
