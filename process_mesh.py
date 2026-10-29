#!/usr/bin/env python3
"""Extract a surface mesh from a NIfTI volume, color it with a NOCS map,
and export PLY/VTK plus inverse-mapping metadata."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import filters
import pyvista as pv


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #
def compute_mask(data, closing_iter):
    """Threshold the volume and clean up topology without eroding the surface."""
    pos = data[data > 0]
    
    threshold = filters.threshold_otsu(pos)

    mask = data > threshold
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity (gentle)
    mask = ndimage.binary_closing(mask, structure=struct, iterations=closing_iter)
    mask = ndimage.binary_fill_holes(mask)

    # Keep only the largest connected component
    labels, n = ndimage.label(mask, structure=struct)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))
        mask = labels == (np.argmax(sizes) + 1)

    return mask.astype(np.uint8), float(threshold)


def compute_sdf(mask, sigma):
    """Signed distance field (negative inside) in VOXEL units, then smoothed."""
    dist_interior = ndimage.distance_transform_edt(mask)
    dist_exterior = ndimage.distance_transform_edt(1 - mask)
    sdf = dist_exterior - dist_interior
    return ndimage.gaussian_filter(sdf, sigma=sigma)


def extract_surface(sdf, upsample):
    """Contour the zero level set. SDF lives in POINT data so contour points
    equal voxel indices (no half-voxel shift), which keeps the affine exact."""
    if upsample != 1.0:
        sdf = ndimage.zoom(sdf, zoom=upsample, order=1)

    grid = pv.ImageData(
        dimensions=sdf.shape,                 # points, not cells
        spacing=(1.0 / upsample,) * 3,        # back into original index units
        origin=(0.0, 0.0, 0.0),
    )
    grid.point_data["sdf"] = sdf.ravel(order="F")  # VTK is column-major
    mesh = grid.contour(isosurfaces=[0.0], scalars="sdf")
    return mesh.extract_largest()


def smooth_mesh(mesh, method, n_iter):
    if method == "taubin":   # volume-preserving — best for surface detail
        return mesh.smooth_taubin(n_iter=n_iter, pass_band=0.1)
    if method == "laplacian":  # shrinks/rounds the surface
        return mesh.smooth(n_iter=n_iter, relaxation_factor=0.1)
    return mesh  # "none"


def apply_affine(mesh, affine):
    """Map voxel-index coordinates into NIfTI world space (mm)."""
    v = mesh.points
    homog = np.c_[v, np.ones(len(v))]
    mesh.points = (affine @ homog.T).T[:, :3]
    return mesh


def compute_nocs(points):
    """Normalize vertex positions to [0,1] within the bbox -> (x,y,z)->(R,G,B)."""
    bbox_min = points.min(axis=0)
    bbox_range = points.max(axis=0) - bbox_min
    bbox_range[bbox_range < 1e-8] = 1.0
    return (points - bbox_min) / bbox_range


def write_ply(path, mesh, nocs_rgb, color_format):
    """Binary PLY writer with NOCS vertex colors."""
    mesh = mesh.triangulate()  # guarantee the all-triangles assumption below
    verts = np.asarray(mesh.points, dtype=np.float32)
    faces_raw = mesh.faces.reshape(-1, 4)
    assert (faces_raw[:, 0] == 3).all(), "Mesh is not all-triangles."
    faces = faces_raw[:, 1:].astype(np.int32)
    n_verts, n_faces = len(verts), len(faces)

    if color_format == "uchar":
        col = (np.clip(nocs_rgb, 0, 1) * 255).astype(np.uint8)
        col_prop, col_np = "uchar", "u1"
    else:  # "float"
        col = np.clip(nocs_rgb, 0, 1).astype(np.float32)
        col_prop, col_np = "float", "<f4"

    with open(path, "wb") as f:
        f.write((
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n_verts}\n"
            "property float x\nproperty float y\nproperty float z\n"
            f"property {col_prop} red\nproperty {col_prop} green\nproperty {col_prop} blue\n"
            f"element face {n_faces}\n"
            "property list uchar int vertex_indices\nend_header\n"
        ).encode("ascii"))

        vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("r", col_np), ("g", col_np), ("b", col_np)])
        vd = np.empty(n_verts, dtype=vdt)
        vd["x"], vd["y"], vd["z"] = verts.T
        vd["r"], vd["g"], vd["b"] = col.T
        f.write(vd.tobytes())

        fdt = np.dtype([("n", "u1"), ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4")])
        fd = np.empty(n_faces, dtype=fdt)
        fd["n"] = 3
        fd["i0"], fd["i1"], fd["i2"] = faces.T
        f.write(fd.tobytes())


def visualize(mesh, nocs_colors):
    plotter = pv.Plotter(shape=(1, 2), window_size=(2400, 800))
    plotter.subplot(0, 0)
    plotter.add_text("Surface", font_size=14)
    plotter.add_mesh(mesh, color="white")
    plotter.add_axes()

    plotter.subplot(0, 1)
    plotter.add_text("NOCS Map (XYZ->RGB)", font_size=14)
    plotter.add_mesh(mesh, scalars=nocs_colors, rgb=True,
                     show_scalar_bar=False, lighting=False)
    plotter.add_axes()

    plotter.link_views()
    plotter.show()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="NIfTI -> surface mesh -> NOCS color map -> PLY/VTK export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", nargs="?",
                   default="input/nifti/t1_mprage_sag_p2_iso_12_ras_PJ.nii",
                   help="Path to the input NIfTI (.nii / .nii.gz).")
    p.add_argument("-o", "--output-dir", default="output",
                   help="Base output directory (a timestamped subfolder is created).")

    g = p.add_argument_group("masking")
    g.add_argument("--closing-iter", type=int, default=1,
                   help="Binary closing iterations.")
    g = p.add_argument_group("surface")
    g.add_argument("--sdf-sigma", type=float, default=1.5,
                   help="Gaussian sigma (voxels) applied to the SDF.")
    g.add_argument("--upsample", type=float, default=1.0,
                   help="SDF upsample factor before ßcontouring (>1 = finer).")
    g.add_argument("--smooth-method", choices=["taubin", "laplacian", "none"],
                   default="laplacian", help="Mesh smoothing (taubin preserves volume).")
    g.add_argument("--smooth-iter", type=int, default=20)

    g = p.add_argument_group("output")
    g.add_argument("--color-format", choices=["float", "uchar"], default="float",
                   help="PLY vertex color encoding.")
    g.add_argument("--no-show", action="store_true",
                   help="Skip interactive visualization (use for headless runs).")
    return p.parse_args()


def main():
    args = parse_args()

    img = nib.load(args.input)
    data = img.get_fdata().astype(np.float32)
    affine = img.affine
    orientation = "".join(c for c in "".join(nib.aff2axcodes(affine)) if c.isalpha())

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"nocs_mesh_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    mask, threshold = compute_mask(
        data, 2)
    sdf = compute_sdf(mask, args.sdf_sigma)
    mesh = extract_surface(sdf, args.upsample)
    mesh = smooth_mesh(mesh, args.smooth_method, args.smooth_iter)
    mesh = apply_affine(mesh, affine)

    nocs_colors = compute_nocs(mesh.points.copy())
    mesh.point_data["nocs_r"] = nocs_colors[:, 0]
    mesh.point_data["nocs_g"] = nocs_colors[:, 1]
    mesh.point_data["nocs_b"] = nocs_colors[:, 2]
    rgb_u8 = (np.clip(nocs_colors, 0, 1) * 255).astype(np.uint8)
    mesh.point_data["RGBA"] = np.column_stack(
        [rgb_u8, np.full(len(rgb_u8), 255, np.uint8)])

    base = out_dir / f"mesh_{stamp}_{orientation}"
    mesh.save(f"{base}.ply")
    mesh.save(f"{base}.vtk")  # VTK keeps all point-data arrays
    write_ply(out_dir / f"mesh_nocs_blender_{stamp}.ply",
              mesh, nocs_colors, args.color_format)

    bbox_min, bbox_max = mesh.points.min(0), mesh.points.max(0)
    metadata = {
        "description": "vertex = nocs * bbox_extent + bbox_min",
        "coordinate_space": "NIfTI world (mm)",
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_extent": (bbox_max - bbox_min).tolist(),
        "affine": affine.tolist(),
        "source_nifti": args.input,
        "threshold": threshold,
    }
    with open(out_dir / f"nocs_metadata_{stamp}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Mask voxels: {int(mask.sum())}")
    print(f"Mesh: {mesh.n_points} verts, {mesh.n_cells} faces")
    print(f"World bounds: {tuple(round(b, 1) for b in mesh.bounds)}")
    print(f"Saved to: {out_dir}")

    if not args.no_show:
        try:
            visualize(mesh, nocs_colors)
        except Exception as e:
            print(f"Visualization skipped ({e}). Re-run with --no-show on headless.")


if __name__ == "__main__":
    main()