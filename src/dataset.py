"""Slice dataset (2D training), intensity windowing and augmentation."""
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def window_normalise(x_hu, lo, hi):
    """Clip CT values (Hounsfield Units) to [lo, hi] and rescale to [0, 1]."""
    x = np.clip(x_hu.astype(np.float32), lo, hi)
    return (x - lo) / float(hi - lo)


def random_affine(img, mask, rotation_deg, scale, shift, flip_p):
    """Apply the SAME random rotation/scale/shift/flip to image and mask.
    """
    angle = math.radians(random.uniform(-rotation_deg, rotation_deg))
    s = random.uniform(*scale)
    tx, ty = random.uniform(-shift, shift), random.uniform(-shift, shift)
    flip = -1.0 if random.random() < flip_p else 1.0
    cos, sin = math.cos(angle) / s, math.sin(angle) / s
    # affine_grid maps OUTPUT coordinates to INPUT coordinates
    theta = torch.tensor([[cos * flip, -sin, tx], [sin * flip, cos, ty]], dtype=torch.float32)
    size = (1, 1, img.shape[-2], img.shape[-1])
    grid = F.affine_grid(theta.unsqueeze(0), size, align_corners=False)
    img = F.grid_sample(img.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    mask = F.grid_sample(mask.unsqueeze(0), grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]
    return img, mask


class FemurSliceDataset(Dataset):
    """Axial slices from preprocessed patients.
    """

    def __init__(self, data_dir, case_ids, hu_window, augment=False, aug_cfg=None,
                 neg_ratio=0.2, seed=0):
        self.data_dir = Path(data_dir)
        self.hu_window = tuple(hu_window)
        self.augment = augment
        self.aug_cfg = aug_cfg or {}
        rng = np.random.RandomState(seed)
        self.index = []  # list of (case_id, z)
        for cid in case_ids:
            mask = np.load(self.data_dir / cid / "mask.npy", mmap_mode="r")
            has_femur = mask.reshape(mask.shape[0], -1).any(axis=1)
            pos = np.where(has_femur)[0]
            neg = np.where(~has_femur)[0]
            n_neg = min(len(neg), int(round(neg_ratio * len(pos))))
            neg = rng.choice(neg, size=n_neg, replace=False) if n_neg > 0 else []
            self.index += [(cid, int(z)) for z in pos] + [(cid, int(z)) for z in neg]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        cid, z = self.index[i]
        img = np.load(self.data_dir / cid / "image.npy", mmap_mode="r")[z]
        msk = np.load(self.data_dir / cid / "mask.npy", mmap_mode="r")[z]
        img = torch.from_numpy(window_normalise(np.asarray(img), *self.hu_window)).unsqueeze(0)
        msk = torch.from_numpy(np.asarray(msk).astype(np.float32)).unsqueeze(0)
        if self.augment:
            c = self.aug_cfg
            img, msk = random_affine(img, msk, c.get("rotation_deg", 15), c.get("scale", [0.9, 1.1]),
                                     c.get("shift", 0.05), c.get("flip_p", 0.5))
            # mild intensity augmentation (image only)
            img = img * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)
            noise = c.get("noise_std", 0.0)
            if noise > 0:
                img = img + torch.randn_like(img) * noise
            img = img.clamp(0.0, 1.0)
        return img, msk
