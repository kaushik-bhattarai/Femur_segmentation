"""Evaluate predictions against ground truth, per patient, in mm.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from . import metrics
from .utils import add_common_args, load_config, load_meta, read_json, write_json


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out", default=None, help="results CSV (default: <pred_dir>/metrics.csv)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.opt)
    data_dir, pred_dir = Path(cfg["paths"]["processed_dir"]), Path(args.pred_dir)
    cases = read_json(cfg["paths"]["splits_file"])[args.split]

    rows = []
    for cid in cases:
        pred_img = sitk.ReadImage(str(pred_dir / f"{cid}.nii.gz"))
        pred = sitk.GetArrayFromImage(pred_img) > 0
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(data_dir / cid / "mask.nii.gz"))) > 0
        if pred.shape != gt.shape:
            raise ValueError(f"{cid}: prediction shape {pred.shape} != ground-truth shape {gt.shape}")
        sx, sy, sz = load_meta(data_dir / cid)["spacing_xyz"]
        rows.append({"case": cid, **metrics.evaluate_case(pred, gt, (sz, sy, sx))})
        r = rows[-1]
        print(f"{cid}: Dice {r['dice']:.4f} | HD95 {r['hd95_mm']:.2f} mm | ASD {r['asd_mm']:.2f} mm")

    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else pred_dir / "metrics.csv"
    df.to_csv(out, index=False)

    print("\n=== Summary (mean +/- std over patients, n=%d) ===" % len(df))
    summary = {}
    for col in ["dice", "iou", "hd95_mm", "asd_mm", "precision", "recall"]:
        summary[col] = {"mean": float(np.nanmean(df[col])), "std": float(np.nanstd(df[col], ddof=1)) if len(df) > 1 else 0.0}
        print(f"{col:10s} {summary[col]['mean']:.4f} +/- {summary[col]['std']:.4f}")
    write_json(summary, out.with_suffix(".json"))
    worst = df.sort_values("dice").head(3)
    print("\nWorst 3 cases by Dice (look at these first!):")
    print(worst[["case", "dice", "hd95_mm"]].to_string(index=False))
    print(f"\nPer-patient results saved to {out}")


if __name__ == "__main__":
    main()
