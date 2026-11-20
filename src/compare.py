"""Compare two models on the same patients: summary table plus paired Wilcoxon signed-rank tests.

Usage:
  python -m src.compare \
      --a runs/unet2d_base32/predictions/test/post/metrics.csv --a_name "2D U-Net" \
      --b runs/unet3d_monai/predictions/test/post/metrics.csv  --b_name "MONAI 3D"

With ~12 test patients the tests have little power: treat p-values as a rough guide, not proof.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = ["dice", "hd95_mm", "asd_mm"]


def paired_p(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return float("nan")
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True); ap.add_argument("--a_name", default="A")
    ap.add_argument("--b", required=True); ap.add_argument("--b_name", default="B")
    args = ap.parse_args()
    a = pd.read_csv(args.a).set_index("case").sort_index()
    b = pd.read_csv(args.b).set_index("case").sort_index()
    if list(a.index) != list(b.index):
        raise SystemExit("The two result files do not contain the same patients: use the same split.")

    print(f"{'metric':10s} {args.a_name:>28s} {args.b_name:>28s}   median A / B    Wilcoxon p")
    for m in METRICS:
        fa, fb = f"{a[m].mean():.4f} +/- {a[m].std():.4f}", f"{b[m].mean():.4f} +/- {b[m].std():.4f}"
        print(f"{m:10s} {fa:>28s} {fb:>28s}   {a[m].median():.3f} / {b[m].median():.3f}   {paired_p(a[m], b[m]):.3f}")
    print("\nPer-patient differences (B minus A; for Dice higher is better, for HD95/ASD lower is better):")
    diff = pd.DataFrame({f"d_{m}": b[m] - a[m] for m in METRICS})
    print(diff.round(4).to_string())
    print(f"\nB better on Dice in {(diff['d_dice'] > 0).sum()} of {len(diff)} patients; "
          f"B better on HD95 in {(diff['d_hd95_mm'] < 0).sum()} of {len(diff)}.")


if __name__ == "__main__":
    main()
