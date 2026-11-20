"""Select usable TotalSegmentator cases, preprocess them, and write patient-level splits.

Per case this script:
  1. merges femur_left + femur_right into one binary femur mask,
  2. skips cases where the femur is barely visible (cut off by the scan),
  3. reorients to LPS, resamples to isotropic spacing (linear for CT, nearest for mask),
  4. crops the z-range to [first femur slice - margin, last femur slice + margin],
  5. centre-crops / pads each axial slice to image_size x image_size,
  6. saves image.npy (int16 HU), mask.npy, image.nii.gz, mask.nii.gz and meta.json.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from .utils import add_common_args, load_config, write_json

FEMUR_FILES = ("femur_left.nii.gz", "femur_right.nii.gz")
AIR_HU = -1024


def read_femur_mask(case_dir):
    """Union of left and right femur masks as a uint8 SimpleITK image, or None if missing."""
    seg_dir = Path(case_dir) / "segmentations"
    arr, ref = None, None
    for name in FEMUR_FILES:
        p = seg_dir / name
        if not p.exists():
            continue
        img = sitk.ReadImage(str(p))
        a = sitk.GetArrayFromImage(img) > 0
        arr = a if arr is None else (arr | a)
        ref = img
    if arr is None:
        return None
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.CopyInformation(ref)
    return out


def resample(img, spacing_mm, is_mask):
    old_size = np.array(img.GetSize(), dtype=float)
    old_sp = np.array(img.GetSpacing(), dtype=float)
    new_sp = np.array([spacing_mm] * 3, dtype=float)
    new_size = np.maximum(np.round(old_size * old_sp / new_sp).astype(int), 1)
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing([float(s) for s in new_sp])
    f.SetSize([int(s) for s in new_size])
    f.SetOutputOrigin(img.GetOrigin())
    f.SetOutputDirection(img.GetDirection())
    f.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    f.SetDefaultPixelValue(0 if is_mask else AIR_HU)
    return f.Execute(img)


def crop_or_pad_inplane(arr, size, pad_value):
    """Centre crop/pad the last two axes (y, x) to `size`. Returns (array, (start_x, start_y)).
    A negative start means the image was padded on that side."""
    z, h, w = arr.shape
    sy, sx = (h - size) // 2, (w - size) // 2
    out = np.full((z, size, size), pad_value, dtype=arr.dtype)
    y0, y1 = max(sy, 0), min(sy + size, h)
    x0, x1 = max(sx, 0), min(sx + size, w)
    out[:, y0 - sy:y1 - sy, x0 - sx:x1 - sx] = arr[:, y0:y1, x0:x1]
    return out, (sx, sy)


def process_case(case_dir, out_dir, dcfg):
    """Returns True if the case was accepted and written, False if rejected."""
    mask_img = read_femur_mask(case_dir)
    if mask_img is None:
        return False
    mask_img = sitk.DICOMOrient(mask_img, "LPS")

    # quick rejection on the ORIGINAL-resolution mask (cheap, before touching the CT)
    m = sitk.GetArrayFromImage(mask_img) > 0
    zs = np.where(m.any(axis=(1, 2)))[0]
    if len(zs) == 0:
        return False
    extent_mm = (zs.max() - zs.min() + 1) * mask_img.GetSpacing()[2]
    if extent_mm < dcfg["min_femur_extent_mm"]:
        return False

    ct = sitk.DICOMOrient(sitk.ReadImage(str(Path(case_dir) / "ct.nii.gz")), "LPS")
    ct = sitk.Cast(ct, sitk.sitkFloat32)
    ct_r = resample(ct, dcfg["spacing_mm"], is_mask=False)
    mask_r = resample(mask_img, dcfg["spacing_mm"], is_mask=True)
    if ct_r.GetSize() != mask_r.GetSize():
        return False

    ct_a = np.rint(sitk.GetArrayFromImage(ct_r)).astype(np.int16)
    mask_a = (sitk.GetArrayFromImage(mask_r) > 0).astype(np.uint8)

    zs = np.where(mask_a.any(axis=(1, 2)))[0]
    if len(zs) == 0:
        return False
    z0 = max(int(zs.min()) - dcfg["z_margin_slices"], 0)
    z1 = min(int(zs.max()) + dcfg["z_margin_slices"] + 1, mask_a.shape[0])
    ct_a, mask_a = ct_a[z0:z1], mask_a[z0:z1]
    ct_a, (sx, sy) = crop_or_pad_inplane(ct_a, dcfg["image_size"], AIR_HU)
    mask_a, _ = crop_or_pad_inplane(mask_a, dcfg["image_size"], 0)

    # geometry of the saved volume (so predictions can be written with the same geometry)
    origin = ct_r.TransformIndexToPhysicalPoint((int(sx), int(sy), int(z0)))
    spacing = ct_r.GetSpacing()
    direction = ct_r.GetDirection()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "image.npy", ct_a)
    np.save(out_dir / "mask.npy", mask_a)
    for name, arr in (("image", ct_a), ("mask", mask_a)):
        im = sitk.GetImageFromArray(arr)
        im.SetSpacing(spacing)
        im.SetOrigin(origin)
        im.SetDirection(direction)
        sitk.WriteImage(im, str(out_dir / f"{name}.nii.gz"))
    write_json({
        "id": Path(case_dir).name,
        "spacing_xyz": list(spacing),
        "origin_xyz": list(origin),
        "direction": list(direction),
        "shape_zyx": list(ct_a.shape),
        "femur_voxels": int(mask_a.sum()),
        "original_spacing_xyz": list(mask_img.GetSpacing()),
    }, out_dir / "meta.json")
    return True


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    cfg = load_config(args.config, args.opt)
    dcfg, pcfg = cfg["data"], cfg["paths"]
    raw_dir, proc_dir = Path(pcfg["raw_dir"]), Path(pcfg["processed_dir"])

    candidates = sorted(p for p in raw_dir.iterdir() if p.is_dir() and (p / "ct.nii.gz").exists())
    if not candidates:
        raise SystemExit(f"No cases (folders with ct.nii.gz) found in {raw_dir}. Check paths.raw_dir.")
    random.Random(cfg["seed"]).shuffle(candidates)  # seeded random order -> reproducible selection
    print(f"{len(candidates)} candidate cases found; selecting {dcfg['num_cases']} usable ones.")

    accepted, rejected, unreadable = [], 0, 0
    for case_dir in tqdm(candidates):
        if len(accepted) >= dcfg["num_cases"]:
            break
        out_dir = proc_dir / case_dir.name
        if (out_dir / "meta.json").exists():  # resume support
            accepted.append(case_dir.name)
            continue
        try:
            ok = process_case(case_dir, out_dir, dcfg)
        except RuntimeError as e:  # e.g. NIfTI with non-orthonormal direction cosines
            tqdm.write(f"Skipping {case_dir.name}: cannot read file ({str(e).splitlines()[0][:90]})")
            ok, unreadable = False, unreadable + 1
        if ok:
            accepted.append(case_dir.name)
        else:
            rejected += 1
    print(f"Accepted {len(accepted)} cases, rejected {rejected} "
          f"(femur missing / too short / geometry mismatch / {unreadable} unreadable files).")
    if len(accepted) < min(10, dcfg["num_cases"]):
        print("WARNING: fewer than 10 usable cases; splits will be tiny. Try the full dataset or lower "
              "data.min_femur_extent_mm.")

    # patient-level split
    ids = sorted(accepted)
    random.Random(cfg["seed"]).shuffle(ids)
    n = len(ids)
    n_train = int(round(dcfg["split"][0] * n))
    n_val = int(round(dcfg["split"][1] * n))
    splits = {"train": sorted(ids[:n_train]),
              "val": sorted(ids[n_train:n_train + n_val]),
              "test": sorted(ids[n_train + n_val:])}
    write_json(splits, pcfg["splits_file"])
    print({k: len(v) for k, v in splits.items()}, "->", pcfg["splits_file"])


if __name__ == "__main__":
    main()
