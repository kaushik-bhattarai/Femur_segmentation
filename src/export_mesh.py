"""Turn a predicted (or ground-truth) NIfTI mask into smooth STL meshes + a rough femur-length estimate.
"""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage as ndi
from skimage import measure


def component_to_mesh(comp, spacing_xyz, origin_xyz, smooth_iters=15):
    """Marching cubes on one boolean component (z, y, x) -> trimesh in mm (LPS)."""
    sx, sy, sz = spacing_xyz
    padded = np.pad(comp.astype(np.uint8), 1)  # pad so surfaces touching the border are closed
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5, spacing=(sz, sy, sx))
    verts = verts[:, ::-1] - np.array([sx, sy, sz])  # (z,y,x) -> (x,y,z); undo the 1-voxel padding offset
    verts = verts + np.array(origin_xyz)             # identity direction (LPS) assumed
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if smooth_iters > 0:
        trimesh.smoothing.filter_taubin(mesh, iterations=smooth_iters)  # smoothing without shrinkage
    return mesh


def long_axis_length_mm(mesh):
    """Rough length: extent of the vertices along their first principal axis (PCA)."""
    pts = mesh.vertices - mesh.vertices.mean(axis=0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    proj = pts @ vt[0]
    return float(proj.max() - proj.min()), vt[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mask", required=True, help="NIfTI mask (e.g. a prediction)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--smooth_iters", type=int, default=15)
    ap.add_argument("--min_voxels", type=int, default=2000, help="ignore tiny components")
    args = ap.parse_args()

    img = sitk.ReadImage(args.mask)
    arr = sitk.GetArrayFromImage(img) > 0
    spacing, origin = img.GetSpacing(), img.GetOrigin()
    labels, num = ndi.label(arr, structure=ndi.generate_binary_structure(3, 1))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meshes, k = [], 0
    for lab in range(1, num + 1):
        comp = labels == lab
        if comp.sum() < args.min_voxels:
            continue
        mesh = component_to_mesh(comp, spacing, origin, args.smooth_iters)
        length, axis = long_axis_length_mm(mesh)
        k += 1
        mesh.export(out_dir / f"component{k}.stl")
        meshes.append(mesh)
        print(f"component {k}: {comp.sum()} voxels | {len(mesh.faces)} faces | "
              f"watertight={mesh.is_watertight} | PCA long-axis extent {length:.1f} mm")
    if not meshes:
        raise SystemExit("No component large enough to mesh.")
    trimesh.util.concatenate(meshes).export(out_dir / "combined.stl")
    print(f"Saved STL files to {out_dir}")


if __name__ == "__main__":
    main()
