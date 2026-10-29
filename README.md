# Nmatik Labs Mesh Editor

Load a NIfTI volume (`.nii` / `.nii.gz`), turn it into a surface mesh, trim it with X, Y and Z planes, and export a closed (watertight) PLY.

## Run

```bash
npm install
npm run dev
# open http://localhost:3000
```

## Use

1. Load or drop a `.nii` / `.nii.gz` file. The volume is converted to RAS if needed, thresholded (Otsu) into a binary mask, turned into a signed distance field and meshed; vertices are mapped to world millimetres with the NIfTI affine (see below).
2. A cube appears around the model. Drag the arrows on its faces (red X, green Y, blue Z) to trim the model: everything outside the cube is removed. **Reset cube** puts it back around the whole model.
3. Rotate with left-drag, pan with right-drag or Shift-drag, zoom with the wheel. The **X / Y / Z** buttons look along an axis, and **Recenter** (or double-click) returns to the initial view.
4. **Slice** bakes the result so you can cut again. The ventricles are segmented and hollowed out at load time (an internal cavity), so cut faces show a hole wherever a plane crosses them. Each Slice also keeps the pieces it cut away as separate closed parts, which can be put back and exported as their own files. The cut faces show the MRI at that plane, with the ventricles cross-hatched in blue. **Reset** returns to the loaded mesh.
5. **Export PLY** saves `<name>_cut.ply`. Every cut carries the Nmatik Labs lockup (pose-axes mark above NMATIKLABS) as raised lettering near the bottom-right corner of the cut, in the exported file as well as on screen. A mesh with no cut has no cut surface to put it on, so it is saved unchanged.

Coordinates are NIfTI world space in millimetres (RAS+: X right, Y anterior, Z superior).

## How the volume becomes a mesh (`lib/nifti.ts`, `lib/niftiMesh.ts`, run in `lib/niftiWorker.ts`)

- NIfTI-1 only. The first volume is read and scaled by `scl_slope` / `scl_inter`; the affine is the sform, else the qform, else the voxel sizes.
- If the voxel axes are not already R, A, S they are permuted and flipped (and the affine updated), so the viewer always works in RAS. The panel shows the original orientation code.
- Mask: Otsu threshold over positive voxels, closing (2 iterations), holes filled in 3D and again slice by slice along each axis (so a scan cut open at the neck still gives only the outer skin), largest component kept. The volume is padded so the surface is always closed.
- Surface: signed distance field (negative inside), Gaussian-smoothed (sigma 1.5 voxels), meshed at the zero level with surface nets, then cleaned: degenerate triangles and small stray pieces are dropped and the surface gets 10 passes of Taubin (volume-preserving) smoothing.
- Vertices go through the affine to world mm; if the affine mirrors, the triangle winding is reversed so faces point outward.

## How the cut works (`lib/planarCut.ts`)

- Vertices are welded on load, so the outline of the cut joins into closed loops.
- Flipped (inside-out) triangles are turned to face outward on load, so the mesh renders solid and slicers see a consistent shell.
- The cube is six axis-aligned planes applied one after another, each clipping and capping the mesh. Faces still at the model's edge cut nothing.
- Each triangle is clipped against the plane; the new edge points are shared between neighbouring triangles.
- Vertices within 1e-5 of the model size of the plane are snapped exactly onto it, and points that land on top of each other are welded, so repeated cuts stay clean.
- The open edges on the plane are chained into outlines, nested outlines become holes (e.g. a hollow skull), and the outlines are filled with triangles (earcut) to form a flat cap facing outward.
- The cap uses the same vertices as the cut edge, so a closed input mesh always gives a closed output mesh.

The **Result** box says *Watertight* when the exported mesh has no open edges (checked on the result itself). If your scan has holes, close them first (MeshLab: *Close Holes* or *Screened Poisson*).

## Notes

- Vertex colours from the PLY are not kept.
- The exact cut runs in a Web Worker (`lib/cutWorker.ts`), so the page stays responsive on large scans. While you drag a cube face, a GPU-clipped copy of the mesh shows the trim instantly; the exact result with caps appears when you release. Decimate very large scans first if the release step feels slow.
