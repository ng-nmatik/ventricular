import { distanceTo, flood, type Dims } from './niftiMesh';
import type { RasVolume } from './nifti';

const MIN_DEPTH_MM = 28; // ventricles sit deeper than this below the skin; the skull and surface CSF do not
// The search region is an ellipsoid around the middle of the brain, found from the top of the head (Z is up):
// its centre is this far below the vertex, and these are its semi-axes (R, A, S), all in mm.
const CENTRE_BELOW_VERTEX_MM = 66;
const REGION_RADII_MM = [45, 70, 28];
const KEEP_RATIO = 0.15; // keep every dark piece at least this fraction of the largest one
const CSF_FRACTION = 0.65; // CSF is darker than this fraction of the median brain intensity on T1

/**
 * Ventricular CSF on a T1 volume: voxels darker than the CSF/tissue threshold (a fraction of the median deep-brain intensity),
 * lying well inside the head, in the largest connected dark pieces. 1 where ventricle, 0 elsewhere.
 */
export function segmentVentricles(vol: RasVolume, head: Uint8Array): { mask: Uint8Array; threshold: number; volumeMl: number } {
  const dims = vol.dims as Dims;
  const n = vol.data.length;
  const mm = (vol.spacing[0] + vol.spacing[1] + vol.spacing[2]) / 3;

  // Vertex: the highest slice with a real head section (hair and noise give thin slivers above it).
  const [nx, ny, nz] = dims;
  const area = new Float64Array(nz);
  for (let z = 0; z < nz; z++) for (let i = nx * ny * z; i < nx * ny * (z + 1); i++) area[z] += head[i];
  const widest = Math.max(...area);
  let top = nz - 1;
  while (top > 0 && area[top] < 0.05 * widest) top--;
  const cz = top - CENTRE_BELOW_VERTEX_MM / vol.spacing[2];
  const slice = Math.max(0, Math.min(nz - 1, Math.round(cz)));
  let cx = 0, cy = 0, cells = 0;
  for (let y = 0; y < ny; y++)
    for (let x = 0; x < nx; x++)
      if (head[x + nx * (y + ny * slice)]) {
        cx += x;
        cy += y;
        cells++;
      }
  if (cells === 0) return { mask: new Uint8Array(n), threshold: 0, volumeMl: 0 };
  cx /= cells;
  cy /= cells;
  const inRegion = (x: number, y: number, z: number) =>
    ((x - cx) * vol.spacing[0]) ** 2 / REGION_RADII_MM[0] ** 2 +
      ((y - cy) * vol.spacing[1]) ** 2 / REGION_RADII_MM[1] ** 2 +
      ((z - cz) * vol.spacing[2]) ** 2 / REGION_RADII_MM[2] ** 2 <= 1;

  const depth = distanceTo(head, dims, 0); // voxels from the skin, inside the head
  const deep = new Uint8Array(n);
  const sample: number[] = [];
  const reach = MIN_DEPTH_MM / mm;
  for (let z = 0; z < nz; z++)
    for (let y = 0; y < ny; y++)
      for (let x = 0; x < nx; x++) {
        const i = x + nx * (y + ny * z);
        if (head[i] && depth[i] >= reach && inRegion(x, y, z)) {
          deep[i] = 1;
          if (i % 3 === 0) sample.push(vol.data[i]); // a subsample is plenty for a histogram
        }
      }
  if (sample.length < 500) return { mask: new Uint8Array(n), threshold: 0, volumeMl: 0 };
  sample.sort((a, b) => a - b);
  const median = sample[sample.length >> 1];
  const threshold = CSF_FRACTION * median;

  const dark = new Uint8Array(n);
  for (let i = 0; i < n; i++) dark[i] = deep[i] && vol.data[i] < threshold ? 1 : 0;

  const labels = new Int32Array(n);
  const stack = new Int32Array(n);
  const sizes: number[] = [0];
  for (let i = 0; i < n; i++) if (dark[i] && labels[i] === 0) sizes.push(flood(dark, dims, i, 1, labels, sizes.length, stack));
  const largest = Math.max(...sizes);
  const mask = new Uint8Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) {
    const size = sizes[labels[i]];
    if (labels[i] > 0 && size >= KEEP_RATIO * largest) {
      mask[i] = 1;
      count++;
    }
  }
  return { mask, threshold, volumeMl: (count * vol.spacing[0] * vol.spacing[1] * vol.spacing[2]) / 1000 };
}

/** Display intensities: the 1st to 99.5th percentile of the head mapped onto 0..255. */
export function windowIntensities(vol: RasVolume, head: Uint8Array): Uint8Array {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < vol.data.length; i += 5) if (head[i]) {
    if (vol.data[i] < lo) lo = vol.data[i];
    if (vol.data[i] > hi) hi = vol.data[i];
  }
  const bins = 1024;
  const hist = new Float64Array(bins);
  const step = (hi - lo) / bins || 1;
  let total = 0;
  for (let i = 0; i < vol.data.length; i += 5)
    if (head[i]) {
      hist[Math.min(bins - 1, Math.floor((vol.data[i] - lo) / step))]++;
      total++;
    }
  let acc = 0, a = 0, b = bins - 1;
  for (let i = 0; i < bins; i++) {
    acc += hist[i];
    if (acc < 0.01 * total) a = i + 1;
    if (acc < 0.995 * total) b = i + 1;
  }
  const from = lo + a * step, span = Math.max(lo + b * step - from, 1e-6);
  const out = new Uint8Array(vol.data.length);
  for (let i = 0; i < out.length; i++) out[i] = Math.max(0, Math.min(255, Math.round(((vol.data[i] - from) / span) * 255)));
  return out;
}
