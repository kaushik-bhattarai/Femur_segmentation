"""Overlay figures: CT with ground-truth (green) and prediction (red) contours, coronal + axial views.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from .dataset import window_normalise
from .utils import add_common_args, load_config, read_json


def draw(ax, img2d, gt2d, pred2d, title):
    ax.imshow(img2d, cmap="gray", origin="lower", vmin=0, vmax=1)
    if gt2d.any():
        ax.contour(gt2d.astype(float), levels=[0.5], colors="lime", linewidths=1.0)
    if pred2d.any():
        ax.contour(pred2d.astype(float), levels=[0.5], colors="red", linewidths=1.0, linestyles="--")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()
    cfg = load_config(args.config, args.opt)
    data_dir, pred_dir, out_dir = Path(cfg["paths"]["processed_dir"]), Path(args.pred_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cid in read_json(cfg["paths"]["splits_file"])[args.split]:
        ct = window_normalise(np.load(data_dir / cid / "image.npy"), *cfg["data"]["hu_window"])
        gt = np.load(data_dir / cid / "mask.npy") > 0
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_dir / f"{cid}.nii.gz"))) > 0
        zc, yc, _ = [int(np.mean(i)) for i in np.where(gt)]
        xs = np.where(gt)[2]
        xc = int(np.median(xs[xs >= xs.mean()]))  # a sagittal cut through one femur, not the midline
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        draw(axes[0], ct[:, yc, :], gt[:, yc, :], pred[:, yc, :], f"{cid} coronal (y={yc})")
        draw(axes[1], ct[zc], gt[zc], pred[zc], f"axial (z={zc})")
        draw(axes[2], ct[:, :, xc], gt[:, :, xc], pred[:, :, xc], f"sagittal (x={xc})")
        fig.suptitle("green = ground truth, red dashed = prediction", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"{cid}.png", dpi=110)
        plt.close(fig)
    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
