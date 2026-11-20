"""Shared pieces of the MONAI (3D) pipeline: model factory, transforms, sliding-window prediction."""
import math

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, SegResNet
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, RandAffined, RandCropByPosNegLabeld,
    RandFlipd, RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd, ScaleIntensityRanged,
    SpatialPadd,
)

from .dataset import window_normalise


def build_monai_model(mcfg):
    """3D network with 1 input channel (CT) and 1 output logit (femur), like the 2D model."""
    if mcfg["name"] == "unet":
        return UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=tuple(mcfg["channels"]),
                    strides=tuple(mcfg["strides"]), num_res_units=mcfg["num_res_units"], norm=mcfg["norm"])
    if mcfg["name"] == "segresnet":
        return SegResNet(spatial_dims=3, init_filters=mcfg["init_filters"], in_channels=1, out_channels=1,
                         blocks_down=tuple(mcfg["blocks_down"]), blocks_up=tuple(mcfg["blocks_up"]))
    raise ValueError(f"Unknown model name: {mcfg['name']}")


def make_items(data_dir, case_ids):
    """MONAI-style list of dicts pointing at the preprocessed .npy volumes (z, y, x)."""
    return [{"image": f"{data_dir}/{c}/image.npy", "label": f"{data_dir}/{c}/mask.npy"} for c in case_ids]


def train_transforms(cfg):
    lo, hi = cfg["data"]["hu_window"]
    patch = tuple(cfg["train"]["patch_size"])
    aug = cfg["augment"]
    rot = math.radians(aug["rotation_deg"])
    keys = ["image", "label"]
    return Compose([
        LoadImaged(keys=keys, reader="NumpyReader", image_only=True),
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        EnsureTyped(keys=keys, dtype=torch.float32),
        ScaleIntensityRanged(keys="image", a_min=lo, a_max=hi, b_min=0.0, b_max=1.0, clip=True),
        SpatialPadd(keys=keys, spatial_size=patch),                       # volumes shorter than the patch
        RandCropByPosNegLabeld(keys=keys, label_key="label", spatial_size=patch, pos=1, neg=1,
                               num_samples=cfg["train"]["patches_per_volume"], image_key="image"),
        # axis order is (z, y, x): the first rotation axis is the in-plane rotation of the axial slice
        RandAffined(keys=keys, prob=0.5, rotate_range=(rot, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest"), padding_mode="zeros"),
        RandFlipd(keys=keys, spatial_axis=2, prob=aug["flip_p"]),         # left-right flip (both femurs = one class)
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.05, prob=0.5),
        RandGaussianNoised(keys="image", prob=0.3, std=aug["noise_std"]),
    ])


@torch.no_grad()
def predict_volume_3d(model, vol_hu, device, hu_window, roi_size, sw_batch_size=2, overlap=0.5, amp=True):
    """Sliding-window prediction of a whole (z, y, x) HU volume. Returns float32 probabilities (z, y, x)."""
    model.eval()
    x = torch.from_numpy(window_normalise(np.asarray(vol_hu), *hu_window))[None, None].to(device)
    use_amp = amp and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        logits = sliding_window_inference(x, roi_size=tuple(roi_size), sw_batch_size=sw_batch_size,
                                          predictor=model, overlap=overlap, mode="gaussian")
    return torch.sigmoid(logits.float())[0, 0].cpu().numpy()
