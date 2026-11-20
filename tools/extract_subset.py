"""Count/extract the TotalSegmentator cases usable for a structure.

"""
import argparse
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

DEFAULT_STRUCTURES = ["femur_left", "femur_right"]


def femur_extent_mm(case_dir, structures=DEFAULT_STRUCTURES):
    """Visible femur length in mm, 0.0 if no femur, or None if SimpleITK cannot read the file
    (a few TotalSegmentator files have non-orthonormal direction cosines)."""
    arr, ref = None, None
    try:
        for name in structures:
            p = case_dir / "segmentations" / f"{name}.nii.gz"
            if p.exists():
                img = sitk.DICOMOrient(sitk.ReadImage(str(p)), "LPS")
                a = sitk.GetArrayFromImage(img) > 0
                arr = a if arr is None else (arr | a)
                ref = img
    except RuntimeError:
        return None
    if arr is None or not arr.any():
        return 0.0
    zs = np.where(arr.any(axis=(1, 2)))[0]
    return float((zs.max() - zs.min() + 1) * ref.GetSpacing()[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path")
    ap.add_argument("out_dir")
    ap.add_argument("--num_cases", type=int, default=120, help="a bit more than data.num_cases (80)")
    ap.add_argument("--min_extent_mm", type=float, default=150.0)
    ap.add_argument("--structures", nargs="+", default=DEFAULT_STRUCTURES,
                    help="TotalSegmentator label names to merge, e.g. hip_left hip_right")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extract", action="store_true", help="actually extract the CT files")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip_path) as z:
        names = {i.filename: i for i in z.infolist()}
        cases = sorted({n.split("/")[0] for n in names if n.endswith("/ct.nii.gz")})
        print(f"{len(cases)} cases in the zip. Extracting femur masks (small)...")
        for cid in tqdm(cases):
            for name in args.structures:
                member = f"{cid}/segmentations/{name}.nii.gz"
                if member in names and not (out / member).exists():
                    z.extract(member, out)

        extents = {c: femur_extent_mm(out / c, args.structures) for c in tqdm(cases, desc="measuring femur")}
        unreadable = [c for c, e in extents.items() if e is None]
        usable = [c for c, e in extents.items() if e is not None and e >= args.min_extent_mm]
        print(f"{len(unreadable)} cases skipped (SimpleITK cannot read their masks).")
        print(f"{len(usable)} of {len(cases)} cases show >= {args.min_extent_mm:.0f} mm of femur.")
        random.Random(args.seed).shuffle(usable)
        chosen = sorted(usable[:args.num_cases])
        need = sum(names[f"{c}/ct.nii.gz"].file_size for c in chosen) / 1e9
        print(f"Chosen {len(chosen)} cases; their CTs need {need:.1f} GB of disk.")

        for c in cases:  # drop masks of cases we will not use
            if c not in chosen:
                shutil.rmtree(out / c, ignore_errors=True)
        if not args.extract:
            print("Dry run finished. Re-run with --extract to extract the CTs.")
            return
        if "meta.csv" in names:
            z.extract("meta.csv", out)
        for c in tqdm(chosen, desc="extracting CTs"):
            z.extract(f"{c}/ct.nii.gz", out)
    print(f"Done. Set paths.raw_dir to {out} (it contains s0000-style folders).")


if __name__ == "__main__":
    main()
