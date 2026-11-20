"""Mask clean-up after prediction."""
import numpy as np
from scipy import ndimage as ndi


def keep_largest_components(mask, n=2, min_fraction=0.1):
    mask = mask.astype(bool)
    labels, num = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    if num == 0:
        return mask
    sizes = np.bincount(labels.ravel())[1:]
    order = np.argsort(sizes)[::-1]
    largest = sizes[order[0]]
    keep = [int(i) + 1 for i in order[:n] if sizes[i] >= min_fraction * largest]
    return np.isin(labels, keep)
