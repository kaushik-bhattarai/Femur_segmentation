"""3D segmentation metrics (computed per patient, in millimetres where relevant).
"""
import numpy as np
from scipy import ndimage as ndi

_STRUCT = ndi.generate_binary_structure(3, 1)  # 6-connectivity


def dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    s = pred.sum() + gt.sum()
    if s == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / s)


def iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    u = np.logical_or(pred, gt).sum()
    if u == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / u)


def precision_recall(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    precision = tp / pred.sum() if pred.sum() > 0 else float("nan")
    recall = tp / gt.sum() if gt.sum() > 0 else float("nan")
    return float(precision), float(recall)


def _surface(mask):
    """Boundary voxels: the mask minus its erosion."""
    return mask & ~ndi.binary_erosion(mask, structure=_STRUCT, border_value=0)


def _bbox(mask_union, margin=2):
    idx = np.where(mask_union)
    sl = []
    for ax in range(3):
        lo = max(int(idx[ax].min()) - margin, 0)
        hi = min(int(idx[ax].max()) + margin + 1, mask_union.shape[ax])
        sl.append(slice(lo, hi))
    return tuple(sl)


def surface_distances(pred, gt, spacing_zyx):
    """Return (HD95, ASD) in mm. HD95 = max of the two directed 95th percentiles (as in MONAI)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan"), float("nan")
    sl = _bbox(pred | gt)
    pred, gt = pred[sl], gt[sl]
    sp, sg = _surface(pred), _surface(gt)
    dist_to_gt = ndi.distance_transform_edt(~sg, sampling=spacing_zyx)
    dist_to_pred = ndi.distance_transform_edt(~sp, sampling=spacing_zyx)
    d_pg = dist_to_gt[sp]   # pred surface -> nearest GT surface
    d_gp = dist_to_pred[sg]  # GT surface -> nearest pred surface
    hd95 = max(np.percentile(d_pg, 95), np.percentile(d_gp, 95))
    asd = (d_pg.mean() + d_gp.mean()) / 2.0
    return float(hd95), float(asd)


def evaluate_case(pred, gt, spacing_zyx):
    hd95, asd = surface_distances(pred, gt, spacing_zyx)
    prec, rec = precision_recall(pred, gt)
    voxel_mm3 = float(np.prod(spacing_zyx))
    return {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "hd95_mm": hd95,
        "asd_mm": asd,
        "precision": prec,
        "recall": rec,
        "gt_volume_ml": float(gt.sum() * voxel_mm3 / 1000.0),
        "pred_volume_ml": float(pred.sum() * voxel_mm3 / 1000.0),
    }
