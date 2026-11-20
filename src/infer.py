"""Predict full 3D volumes for a split and save them as NIfTI (raw and post-processed).
"""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from tqdm import tqdm

from .inference import predict_volume
from .model import build_model
from .postprocess import keep_largest_components
from .utils import add_common_args, get_device, load_config, load_meta, read_json


def save_nifti(arr, meta, path):
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing(meta["spacing_xyz"])
    img.SetOrigin(meta["origin_xyz"])
    img.SetDirection(meta["direction"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(path))


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ckpt", default=None, help="default: runs/<run_name>/best.pt")
    args = ap.parse_args()
    cfg = load_config(args.config, args.opt)
    device = get_device()
    run_dir = Path(cfg["paths"]["runs_dir"]) / cfg["train"]["run_name"]
    ckpt_path = Path(args.ckpt) if args.ckpt else run_dir / "best.pt"
    ck = torch.load(ckpt_path, map_location=device)
    model = build_model(ck["cfg"]["model"]).to(device)
    model.load_state_dict(ck["model"])
    print(f"Loaded {ckpt_path} (epoch {ck['epoch']}, val dice {ck['val_dice']:.4f})")

    icfg = cfg["infer"]
    data_dir = Path(cfg["paths"]["processed_dir"])
    cases = read_json(cfg["paths"]["splits_file"])[args.split]
    out_root = run_dir / "predictions" / args.split
    for cid in tqdm(cases):
        vol = np.load(data_dir / cid / "image.npy", mmap_mode="r")
        meta = load_meta(data_dir / cid)
        prob = predict_volume(model, vol, device, cfg["data"]["hu_window"], batch_size=icfg["batch_size"],
                              amp=cfg["train"]["amp"], tta_flip=icfg["tta_flip"])
        raw = prob > icfg["threshold"]
        post = keep_largest_components(raw, icfg["keep_components"], icfg["min_component_fraction"])
        save_nifti(raw, meta, out_root / "raw" / f"{cid}.nii.gz")
        save_nifti(post, meta, out_root / "post" / f"{cid}.nii.gz")
    print(f"Saved predictions to {out_root}/{{raw,post}}")


if __name__ == "__main__":
    main()
