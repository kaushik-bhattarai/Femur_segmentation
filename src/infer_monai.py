"""Predict full 3D volumes with the trained MONAI model (sliding window) and save NIfTI predictions.

Output layout is identical to src.infer, so src.evaluate scores both models with the same code:
  runs/<run_name>/predictions/<split>/{raw,post}/<case_id>.nii.gz

Usage: python -m src.infer_monai --config configs/monai3d.yaml --split test
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .infer import save_nifti
from .monai_utils import build_monai_model, predict_volume_3d
from .postprocess import keep_largest_components
from .utils import add_common_args, get_device, load_config, load_meta, read_json


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
    model = build_monai_model(ck["cfg"]["model"]).to(device)
    model.load_state_dict(ck["model"])
    print(f"Loaded {ckpt_path} (epoch {ck['epoch']}, val dice {ck['val_dice']:.4f})")

    icfg, tcfg = cfg["infer"], ck["cfg"]["train"]
    data_dir = Path(cfg["paths"]["processed_dir"])
    out_root = run_dir / "predictions" / args.split
    for cid in tqdm(read_json(cfg["paths"]["splits_file"])[args.split]):
        vol = np.load(data_dir / cid / "image.npy", mmap_mode="r")
        meta = load_meta(data_dir / cid)
        prob = predict_volume_3d(model, vol, device, cfg["data"]["hu_window"], tcfg["patch_size"],
                                 icfg["sw_batch_size"], icfg["overlap"], tcfg["amp"])
        raw = prob > icfg["threshold"]
        post = keep_largest_components(raw, icfg["keep_components"], icfg["min_component_fraction"])
        save_nifti(raw, meta, out_root / "raw" / f"{cid}.nii.gz")
        save_nifti(post, meta, out_root / "post" / f"{cid}.nii.gz")
    print(f"Saved predictions to {out_root}/{{raw,post}}")


if __name__ == "__main__":
    main()
