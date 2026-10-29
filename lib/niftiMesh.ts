import type { RasVolume } from './nifti';
import type { IndexedMesh } from './planarCut';

const CLOSING_ITERATIONS = 2;
const PAD = 4; // empty voxels added on every side: closing does not eat the border and the surface always closes
const SDF_SIGMA = 1.5; // voxels: Gaussian smoothing of the signed distance field
const TAUBIN_ITERATIONS = 10; // volume-preserving smoothing of the surface
const TAUBIN_LAMBDA = 0.5;
const TAUBIN_MU = -0.53;

export type Dims = [number, number, number];

/** Otsu's threshold over the positive voxels. */
function otsu(data: Float32Array): number {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of data) if (v > 0) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!(hi > lo)) throw new Error('The volume has no positive voxels to threshold.');
  const bins = 256;
  const hist = new Float64Array(bins);
  const step = (hi - lo) / bins;
  for (const v of data) if (v > 0) hist[Math.min(bins - 1, Math.floor((v - lo) / step))]++;
  let total = 0, sumAll = 0;
  for (let i = 0; i < bins; i++) {
    total += hist[i];
    sumAll += hist[i] * i;
  }
  let w0 = 0, sum0 = 0, bestVar = -1, bestBin = 0;
  for (let i = 0; i < bins - 1; i++) {
    w0 += hist[i];
    sum0 += hist[i] * i;
    const w1 = total - w0;
    if (w0 === 0 || w1 === 0) continue;
    const m0 = sum0 / w0, m1 = (sumAll - sum0) / w1;
    const between = w0 * w1 * (m0 - m1) * (m0 - m1);
    if (between > bestVar) {
      bestVar = between;
      bestBin = i;
    }
  }
  return lo + (bestBin + 1) * step;
}

/** Marks voxels that have a 6-neighbour set (dilate) or, with `erode`, voxels with a 6-neighbour clear. */
function morph(src: Uint8Array, dims: Dims, erode: boolean): Uint8Array {
  const [nx, ny, nz] = dims;
  const out = new Uint8Array(src.length);
  const target = erode ? 0 : 1;
  for (let z = 0; z < nz; z++)
    for (let y = 0; y < ny; y++)
      for (let x = 0; x < nx; x++) {
        const i = x + nx * (y + ny * z);
        let hit = src[i] === target;
        if (!hit && x > 0) hit = src[i - 1] === target;
        if (!hit && x < nx - 1) hit = src[i + 1] === target;
        if (!hit && y > 0) hit = src[i - nx] === target;
        if (!hit && y < ny - 1) hit = src[i + nx] === target;
        if (!hit && z > 0) hit = src[i - nx * ny] === target;
        if (!hit && z < nz - 1) hit = src[i + nx * ny] === target;
        out[i] = erode ? (hit ? 0 : 1) : hit ? 1 : 0;
      }
  return out;
}

/** Visits every voxel connected (6-neighbourhood) to `start` that has `value`, calling `visit` on each. */
export function flood(src: Uint8Array, dims: Dims, start: number, value: number, seen: Uint8Array | Int32Array, mark: number, stack: Int32Array): number {
  const [nx, ny, nz] = dims;
  const sx = nx * ny;
  let top = 0;
  let count = 0;
  stack[top++] = start;
  seen[start] = mark;
  while (top > 0) {
    const i = stack[--top];
    count++;
    const x = i % nx, y = ((i / nx) | 0) % ny, z = (i / sx) | 0;
    const push = (j: number) => {
      if (src[j] === value && seen[j] === 0) {
        seen[j] = mark;
        stack[top++] = j;
      }
    };
    if (x > 0) push(i - 1);
    if (x < nx - 1) push(i + 1);
    if (y > 0) push(i - nx);
    if (y < ny - 1) push(i + nx);
    if (z > 0) push(i - sx);
    if (z < nz - 1) push(i + sx);
  }
  return count;
}

/**
 * Fills holes slice by slice along each axis. Where the scan is cut open (a neck, say) the inside connects to the
 * outside in 3D, but every slice across the cut still shows a closed ring, so this leaves only the outer skin.
 */
function fillSlices(mask: Uint8Array, dims: Dims) {
  const [nx, ny, nz] = dims;
  const stride = [1, nx, nx * ny];
  const filled = new Uint8Array(mask.length);
  for (let a = 0; a < 3; a++) {
    const u = (a + 1) % 3, v = (a + 2) % 3;
    const du = dims[u], dv = dims[v];
    const seen = new Uint8Array(du * dv);
    const stack = new Int32Array(du * dv);
    for (let s = 0; s < dims[a]; s++) {
      const origin = s * stride[a];
      const at = (p: number) => origin + (p % du) * stride[u] + ((p / du) | 0) * stride[v];
      seen.fill(0);
      let top = 0;
      for (let p = 0; p < du * dv; p++) {
        const x = p % du, y = (p / du) | 0;
        if ((x === 0 || y === 0 || x === du - 1 || y === dv - 1) && mask[at(p)] === 0) {
          seen[p] = 1;
          stack[top++] = p;
        }
      }
      while (top > 0) {
        const p = stack[--top];
        const x = p % du, y = (p / du) | 0;
        const visit = (q: number) => {
          if (!seen[q] && mask[at(q)] === 0) {
            seen[q] = 1;
            stack[top++] = q;
          }
        };
        if (x > 0) visit(p - 1);
        if (x < du - 1) visit(p + 1);
        if (y > 0) visit(p - du);
        if (y < dv - 1) visit(p + du);
      }
      for (let p = 0; p < du * dv; p++) if (!seen[p]) filled[at(p)] = 1;
    }
  }
  for (let i = 0; i < mask.length; i++) if (filled[i]) mask[i] = 1;
}

/** Threshold, close, fill the holes and keep the largest piece. `dims` includes the padding. */
function binaryMask(data: Float32Array, dims: Dims, threshold: number): Uint8Array {
  let mask: Uint8Array = new Uint8Array(data.length);
  for (let i = 0; i < data.length; i++) mask[i] = data[i] > threshold ? 1 : 0;
  for (let i = 0; i < CLOSING_ITERATIONS; i++) mask = morph(mask, dims, false);
  for (let i = 0; i < CLOSING_ITERATIONS; i++) mask = morph(mask, dims, true); // closing: seals cracks and small openings

  const stack = new Int32Array(mask.length);
  // Holes: whatever background the outside cannot reach (the padding is all background).
  const outside = new Uint8Array(mask.length);
  flood(mask, dims, 0, 0, outside, 1, stack);
  for (let i = 0; i < mask.length; i++) mask[i] = outside[i] ? 0 : 1;
  fillSlices(mask, dims);

  // Largest connected component.
  const labels = new Int32Array(mask.length);
  let bestLabel = 0, bestSize = 0, next = 1;
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] === 1 && labels[i] === 0) {
      const size = flood(mask, dims, i, 1, labels, next, stack);
      if (size > bestSize) {
        bestSize = size;
        bestLabel = next;
      }
      next++;
    }
  }
  if (bestSize === 0) throw new Error('Thresholding left an empty mask.');
  for (let i = 0; i < mask.length; i++) mask[i] = labels[i] === bestLabel ? 1 : 0;
  return mask;
}

/** Felzenszwalb squared distance transform of one line, in place through `line`. */
function edtLine(line: Float32Array, n: number, v: Int32Array, z: Float64Array, out: Float32Array, w = 1) {
  let k = 0;
  v[0] = 0;
  z[0] = -Infinity;
  z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s: number;
    for (;;) {
      const p = v[k];
      s = (line[q] + q * q * w * w - (line[p] + p * p * w * w)) / (2 * w * (q - p));
      if (s <= z[k]) k--;
      else break;
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q * w) k++;
    out[q] = (q - v[k]) * (q - v[k]) * w * w + line[v[k]];
  }
}

/** Distance from every voxel to the nearest voxel whose mask value equals `zeroValue`: in voxels, or in mm given the voxel `spacing`. */
export function distanceTo(mask: Uint8Array, dims: Dims, zeroValue: number, spacing: [number, number, number] = [1, 1, 1]): Float32Array {
  const [nx, ny, nz] = dims;
  const f = new Float32Array(mask.length);
  for (let i = 0; i < f.length; i++) f[i] = mask[i] === zeroValue ? 0 : 1e20;
  const longest = Math.max(nx, ny, nz);
  const line = new Float32Array(longest), out = new Float32Array(longest);
  const v = new Int32Array(longest), z = new Float64Array(longest + 1);
  const axes: [number, number, number][] = [[1, ny * nz, nx], [nx, nx * nz, ny], [nx * ny, nx * ny, nz]];
  // For each axis: stride between samples, number of lines, line length.
  const lineStart = (axis: number, l: number) => {
    if (axis === 0) return l * nx;
    if (axis === 1) return (l % nx) + nx * ny * ((l / nx) | 0);
    return l;
  };
  axes.forEach(([stride, lines, n], axis) => {
    for (let l = 0; l < lines; l++) {
      const start = lineStart(axis, l);
      for (let q = 0; q < n; q++) line[q] = f[start + q * stride];
      edtLine(line, n, v, z, out, spacing[axis]);
      for (let q = 0; q < n; q++) f[start + q * stride] = out[q];
    }
  });
  for (let i = 0; i < f.length; i++) f[i] = Math.sqrt(f[i]);
  return f;
}

/** Signed distance field (voxel units, or mm given `spacing`), negative inside, then Gaussian smoothed. */
function signedDistance(mask: Uint8Array, dims: Dims, sigma: number, spacing?: [number, number, number]): Float32Array {
  const sdf = distanceTo(mask, dims, 1, spacing); // outside voxels: distance to the object
  const inside = distanceTo(mask, dims, 0, spacing); // inside voxels: distance to the background
  for (let i = 0; i < sdf.length; i++) sdf[i] -= inside[i];
  if (sigma > 0) gaussian(sdf, dims, sigma);
  return sdf;
}

function gaussian(a: Float32Array, dims: Dims, sigma: number) {
  const [nx, ny, nz] = dims;
  const radius = Math.ceil(4 * sigma);
  const kernel = new Float32Array(2 * radius + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) sum += kernel[i + radius] = Math.exp(-(i * i) / (2 * sigma * sigma));
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;

  const longest = Math.max(nx, ny, nz);
  const line = new Float32Array(longest);
  const axes: [number, number, number][] = [[1, ny * nz, nx], [nx, nx * nz, ny], [nx * ny, nx * ny, nz]];
  axes.forEach(([stride, lines, n], axis) => {
    for (let l = 0; l < lines; l++) {
      const start = axis === 0 ? l * nx : axis === 1 ? (l % nx) + nx * ny * ((l / nx) | 0) : l;
      for (let q = 0; q < n; q++) line[q] = a[start + q * stride];
      for (let q = 0; q < n; q++) {
        let acc = 0;
        for (let t = -radius; t <= radius; t++) acc += kernel[t + radius] * line[Math.min(n - 1, Math.max(0, q + t))];
        a[start + q * stride] = acc;
      }
    }
  });
}

// Cube corners are numbered x + 2y + 4z. The 12 edges: four along x, four along y, four along z.
const CORNER = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
const EDGES = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
const FACES = [[0, 2, 6, 4], [1, 3, 7, 5], [0, 1, 5, 4], [2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6]]; // corners in a cycle
const EDGE_AXIS = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

let sheetTable: { comp: Int8Array; count: Uint8Array; edgeAt: number[][] } | null = null;

/**
 * For every sign pattern of a cell, which sheet of the surface each crossed edge belongs to. Giving each sheet its
 * own vertex keeps the mesh manifold where two parts of the surface pinch together inside one cell. On a face with
 * a saddle the inside corners are kept apart, the same way in both cells that share the face.
 */
function getSheetTable() {
  if (sheetTable) return sheetTable;
  const edgeOf = new Map<number, number>();
  EDGES.forEach(([a, b], e) => edgeOf.set(a * 8 + b, e));
  const edgeAt = [0, 1, 2].map(() => new Array<number>(8).fill(-1)); // [axis][lower corner] -> edge
  EDGES.forEach(([a], e) => (edgeAt[EDGE_AXIS[e]][a] = e));

  const comp = new Int8Array(256 * 12).fill(-1);
  const count = new Uint8Array(256);
  for (let mask = 0; mask < 256; mask++) {
    const neg = (c: number) => (mask >> c) & 1;
    const parent = Array.from({ length: 12 }, (_, i) => i);
    const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
    const crossed = EDGES.map(([a, b]) => neg(a) !== neg(b));
    for (const face of FACES) {
      const around = face.map((c, i) => edgeOf.get(Math.min(c, face[(i + 1) % 4]) * 8 + Math.max(c, face[(i + 1) % 4]))!);
      const hit = around.filter((e) => crossed[e]);
      if (hit.length === 2) parent[find(hit[0])] = find(hit[1]);
      else if (hit.length === 4)
        face.forEach((c, i) => {
          if (neg(c)) parent[find(around[(i + 3) % 4])] = find(around[i]);
        });
    }
    const rank = new Map<number, number>();
    for (let e = 0; e < 12; e++) {
      if (!crossed[e]) continue;
      const root = find(e);
      if (!rank.has(root)) rank.set(root, rank.size);
      comp[mask * 12 + e] = rank.get(root)!;
    }
    count[mask] = rank.size;
  }
  return (sheetTable = { comp, count, edgeAt });
}

/** Surface nets on the zero level of `f`: one vertex per surface sheet in a cell, one quad per crossed grid edge. */
function surfaceNets(f: Float32Array, dims: Dims): { positions: number[]; indices: number[] } {
  const [nx, ny, nz] = dims;
  const { comp, count, edgeAt } = getSheetTable();
  const cellStart = new Int32Array(nx * ny * nz).fill(-1);
  const cellMask = new Uint8Array(nx * ny * nz);
  const positions: number[] = [];
  const indices: number[] = [];
  const idx = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  const values = new Float32Array(8);
  const sum = new Float64Array(4 * 4); // up to 4 sheets: x, y, z, crossings

  for (let z = 0; z < nz - 1; z++)
    for (let y = 0; y < ny - 1; y++)
      for (let x = 0; x < nx - 1; x++) {
        let mask = 0;
        for (let c = 0; c < 8; c++) {
          const v = f[idx(x + CORNER[c][0], y + CORNER[c][1], z + CORNER[c][2])];
          values[c] = v;
          if (v < 0) mask |= 1 << c;
        }
        if (mask === 0 || mask === 255) continue;
        sum.fill(0);
        for (let e = 0; e < 12; e++) {
          const k = comp[mask * 12 + e];
          if (k < 0) continue;
          const [a, b] = EDGES[e];
          const t = values[a] / (values[a] - values[b]);
          sum[4 * k] += CORNER[a][0] + t * (CORNER[b][0] - CORNER[a][0]);
          sum[4 * k + 1] += CORNER[a][1] + t * (CORNER[b][1] - CORNER[a][1]);
          sum[4 * k + 2] += CORNER[a][2] + t * (CORNER[b][2] - CORNER[a][2]);
          sum[4 * k + 3]++;
        }
        const cell = idx(x, y, z);
        cellMask[cell] = mask;
        cellStart[cell] = positions.length / 3;
        for (let k = 0; k < count[mask]; k++) positions.push(x + sum[4 * k] / sum[4 * k + 3], y + sum[4 * k + 1] / sum[4 * k + 3], z + sum[4 * k + 2] / sum[4 * k + 3]);
      }

  // Axes (a, b, c) are cyclic, so cells ordered round the edge in the (b, c) plane face +a.
  const next = [[1, 2], [2, 0], [0, 1]];
  const p = [0, 0, 0];
  const rel = [0, 0, 0];
  const quad = [0, 0, 0, 0];
  const around = [[-1, -1], [0, -1], [0, 0], [-1, 0]];
  for (let z = 1; z < nz - 1; z++)
    for (let y = 1; y < ny - 1; y++)
      for (let x = 1; x < nx - 1; x++) {
        const here = f[idx(x, y, z)] < 0;
        for (let a = 0; a < 3; a++) {
          p[0] = x; p[1] = y; p[2] = z;
          p[a]++;
          if (f[idx(p[0], p[1], p[2])] < 0 === here) continue;
          const [b, c] = next[a];
          let ok = true;
          for (let i = 0; i < 4; i++) {
            const [db, dc] = around[i];
            const q = [x, y, z];
            q[b] += db;
            q[c] += dc;
            const cell = idx(q[0], q[1], q[2]);
            if (cellStart[cell] < 0) {
              ok = false;
              break;
            }
            // The crossed edge as seen from this cell picks the cell's vertex for its sheet.
            rel[a] = 0;
            rel[b] = -db;
            rel[c] = -dc;
            const e = edgeAt[a][rel[0] + 2 * rel[1] + 4 * rel[2]];
            quad[i] = cellStart[cell] + comp[cellMask[cell] * 12 + e];
          }
          if (!ok) continue;
          if (here) indices.push(quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]);
          else indices.push(quad[0], quad[2], quad[1], quad[0], quad[3], quad[2]);
        }
      }
  return { positions, indices };
}

/**
 * Splits vertices where two sheets of the surface touch along an edge (more than two triangles on one edge): the
 * triangles around such a vertex are grouped by the manifold edges they share, and each group gets its own copy.
 */
function splitPinchedVertices(positions: number[], indices: number[]) {
  const nv = positions.length / 3;
  const nt = indices.length / 3;
  const original = indices.slice(); // edges are looked up by the numbering before any split
  const key = (a: number, b: number) => (a < b ? a * nv + b : b * nv + a);
  const edgeTris = new Map<number, number[]>();
  for (let t = 0; t < nt; t++)
    for (let k = 0; k < 3; k++) {
      const e = key(original[3 * t + k], original[3 * t + ((k + 1) % 3)]);
      const list = edgeTris.get(e);
      if (list) list.push(t);
      else edgeTris.set(e, [t]);
    }

  const pinched = new Set<number>();
  for (const [e, list] of edgeTris) if (list.length > 2) pinched.add(Math.floor(e / nv)).add(e % nv);
  if (pinched.size === 0) return;

  const incident = new Map<number, number[]>();
  for (const v of pinched) incident.set(v, []);
  for (let t = 0; t < nt; t++) for (let k = 0; k < 3; k++) incident.get(original[3 * t + k])?.push(t);

  for (const [v, tris] of incident) {
    const parent = new Map<number, number>(tris.map((t) => [t, t]));
    const find = (t: number): number => {
      const p = parent.get(t)!;
      if (p === t) return t;
      const r = find(p);
      parent.set(t, r);
      return r;
    };
    for (const t of tris)
      for (let k = 0; k < 3; k++) {
        const a = original[3 * t + k], b = original[3 * t + ((k + 1) % 3)];
        if (a !== v && b !== v) continue;
        const list = edgeTris.get(key(a, b))!;
        if (list.length === 2) parent.set(find(list[0]), find(list[1]));
      }
    const copy = new Map<number, number>();
    copy.set(find(tris[0]), v); // the first group keeps the original vertex
    for (const t of tris) {
      const root = find(t);
      if (!copy.has(root)) {
        copy.set(root, positions.length / 3);
        positions.push(positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]);
      }
      for (let k = 0; k < 3; k++) if (original[3 * t + k] === v) indices[3 * t + k] = copy.get(root)!;
    }
  }
}

/** Drops triangles with repeated vertices and all but the largest connected piece, and renumbers the vertices. */
function cleanSurface(positions: number[], indices: number[], keepAll = false): { positions: Float32Array; indices: Uint32Array } {
  const nv = positions.length / 3;
  const tris: number[] = [];
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t], b = indices[t + 1], c = indices[t + 2];
    if (a === b || b === c || a === c) continue;
    tris.push(a, b, c);
  }

  // Connected pieces through shared vertices (union-find).
  const parent = new Int32Array(nv).map((_, i) => i);
  const find = (i: number) => {
    while (parent[i] !== i) i = parent[i] = parent[parent[i]];
    return i;
  };
  for (let t = 0; t < tris.length; t += 3) {
    parent[find(tris[t + 1])] = find(tris[t]);
    parent[find(tris[t + 2])] = find(tris[t]);
  }
  const size = new Map<number, number>();
  for (let t = 0; t < tris.length; t += 3) size.set(find(tris[t]), (size.get(find(tris[t])) ?? 0) + 1);
  let keep = -1, most = 0;
  for (const [root, n] of size) if (n > most) [keep, most] = [root, n];

  const remap = new Int32Array(nv).fill(-1);
  const outPositions: number[] = [];
  const out: number[] = [];
  for (let t = 0; t < tris.length; t += 3) {
    if (!keepAll && find(tris[t]) !== keep) continue;
    for (let k = 0; k < 3; k++) {
      const v = tris[t + k];
      if (remap[v] < 0) {
        remap[v] = outPositions.length / 3;
        outPositions.push(positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]);
      }
      out.push(remap[v]);
    }
  }
  return { positions: Float32Array.from(outPositions), indices: Uint32Array.from(out) };
}

/** Taubin smoothing (a shrinking step, then an inflating one) so the surface relaxes without losing volume. */
function taubin(P: Float32Array, I: Uint32Array, iterations = TAUBIN_ITERATIONS) {
  const nv = P.length / 3;
  const neighbours: Set<number>[] = Array.from({ length: nv }, () => new Set<number>());
  for (let t = 0; t < I.length; t += 3)
    for (let k = 0; k < 3; k++) {
      neighbours[I[t + k]].add(I[t + ((k + 1) % 3)]);
      neighbours[I[t + k]].add(I[t + ((k + 2) % 3)]);
    }
  const lists = neighbours.map((s) => Int32Array.from(s));
  const next = new Float32Array(P.length);
  const step = (factor: number) => {
    for (let v = 0; v < nv; v++) {
      const ring = lists[v];
      if (ring.length === 0) {
        next.set(P.subarray(3 * v, 3 * v + 3), 3 * v);
        continue;
      }
      for (let d = 0; d < 3; d++) {
        let mean = 0;
        for (const n of ring) mean += P[3 * n + d];
        next[3 * v + d] = P[3 * v + d] + (factor * (mean / ring.length - P[3 * v + d]));
      }
    }
    P.set(next);
  };
  for (let i = 0; i < iterations; i++) {
    step(TAUBIN_LAMBDA);
    step(TAUBIN_MU);
  }
}

/** Surface of the signed distance field `sdf` at level `iso`, cleaned and smoothed, in grid units (padding included). */
function gridSurface(sdf: Float32Array, dims: Dims, iso: number, keepAll: boolean, iterations: number) {
  const level = iso === 0 ? sdf : sdf.map((v) => v - iso);
  const raw = surfaceNets(level, dims);
  if (raw.indices.length === 0) throw new Error('No surface found in this volume.');
  splitPinchedVertices(raw.positions, raw.indices);
  const surface = cleanSurface(raw.positions, raw.indices, keepAll);
  taubin(surface.positions, surface.indices, iterations); // in voxel units, so the smoothing is isotropic in voxels
  return surface;
}

/** Grid coordinates (minus the padding) -> world mm through the affine; reverses the winding if the affine mirrors. */
function toWorld(grid: Float32Array, tris: Uint32Array, A: number[]): IndexedMesh {
  const positions = new Float32Array(grid.length);
  for (let i = 0; i < grid.length; i += 3) {
    const x = grid[i] - PAD, y = grid[i + 1] - PAD, z = grid[i + 2] - PAD;
    positions[i] = A[0] * x + A[1] * y + A[2] * z + A[3];
    positions[i + 1] = A[4] * x + A[5] * y + A[6] * z + A[7];
    positions[i + 2] = A[8] * x + A[9] * y + A[10] * z + A[11];
  }
  const det =
    A[0] * (A[5] * A[10] - A[6] * A[9]) - A[1] * (A[4] * A[10] - A[6] * A[8]) + A[2] * (A[4] * A[9] - A[5] * A[8]);
  if (det < 0) for (let t = 0; t < tris.length; t += 3) [tris[t + 1], tris[t + 2]] = [tris[t + 2], tris[t + 1]];
  return { positions, indices: tris };
}

/** Padded mask -> signed distance field -> surface -> cleaned and smoothed -> world mm. Faces point out of the mask. */
function surfaceFromMask(
  mask: Uint8Array,
  dims: Dims,
  A: number[],
  spacing: [number, number, number],
  shellMm: number,
  progress: (stage: string) => void = () => {},
): { mesh: IndexedMesh; shell: IndexedMesh | null } {
  progress('Computing the signed distance field');
  const sdf = signedDistance(mask, dims, SDF_SIGMA, spacing); // mm
  progress('Extracting the surface');
  const { positions, indices } = gridSurface(sdf, dims, 0, false, TAUBIN_ITERATIONS);
  // The surface `shellMm` below the skin, turned to face the core: with the skin it bounds a shell of that thickness.
  // A head too small to have a core is solid all through, so it has no inner surface.
  let shell: IndexedMesh | null = null;
  try {
    const inner = gridSurface(sdf, dims, -shellMm, true, TAUBIN_ITERATIONS);
    shell = toWorld(inner.positions, inner.indices, A);
    const I = shell.indices;
    for (let t = 0; t < I.length; t += 3) [I[t + 1], I[t + 2]] = [I[t + 2], I[t + 1]];
  } catch {
    shell = null;
  }
  progress('Mapping to world coordinates');
  return { mesh: toWorld(positions, indices, A), shell };
}

const CAVITY_SIGMA = 0.5; // voxels: the ventricles have thin parts that heavier smoothing would erase
const CAVITY_TAUBIN_ITERATIONS = 3;
const CAVITY_PAD = 6;

/** Volume enclosed by a closed triangle mesh, in cubic grid units. */
function enclosedVolume(P: Float32Array, I: Uint32Array): number {
  let v = 0;
  for (let t = 0; t < I.length; t += 3) {
    const a = 3 * I[t], b = 3 * I[t + 1], c = 3 * I[t + 2];
    v += P[a] * (P[b + 1] * P[c + 2] - P[b + 2] * P[c + 1]) - P[a + 1] * (P[b] * P[c + 2] - P[b + 2] * P[c]) + P[a + 2] * (P[b] * P[c + 1] - P[b + 1] * P[c]);
  }
  return Math.abs(v) / 6;
}

/**
 * The surface of a binary mask given in RAS voxel order, in world mm, with every face turned to point INTO the mask.
 * Added to a solid mesh, that is a cavity: the walls face the empty space. Smoothing pulls thin parts inward, so the
 * surface level is tuned until the mesh encloses the same volume as the mask.
 */
export function cavityFromMask(mask: Uint8Array, dims: Dims, affine: number[]): IndexedMesh {
  const [nx, ny, nz] = dims;
  const lo = [nx, ny, nz], hi = [-1, -1, -1];
  let count = 0;
  for (let z = 0; z < nz; z++)
    for (let y = 0; y < ny; y++)
      for (let x = 0; x < nx; x++)
        if (mask[x + nx * (y + ny * z)]) {
          count++;
          lo[0] = Math.min(lo[0], x); hi[0] = Math.max(hi[0], x);
          lo[1] = Math.min(lo[1], y); hi[1] = Math.max(hi[1], y);
          lo[2] = Math.min(lo[2], z); hi[2] = Math.max(hi[2], z);
        }
  if (count === 0) throw new Error('The ventricle mask is empty.');

  // Work on the ventricles' own box (with room for the smoothing) instead of the whole volume.
  const origin = lo.map((v) => v - CAVITY_PAD); // box index i is volume voxel i + origin
  const size = [0, 1, 2].map((a) => hi[a] - lo[a] + 1 + 2 * CAVITY_PAD) as Dims;
  const box = new Uint8Array(size[0] * size[1] * size[2]);
  for (let z = 0; z < size[2]; z++)
    for (let y = 0; y < size[1]; y++)
      for (let x = 0; x < size[0]; x++) {
        const sx = x + origin[0], sy = y + origin[1], sz = z + origin[2];
        if (sx >= 0 && sy >= 0 && sz >= 0 && sx < nx && sy < ny && sz < nz) box[x + size[0] * (y + size[1] * z)] = mask[sx + nx * (sy + ny * sz)];
      }
  // toWorld takes the grid index minus PAD as the voxel index, so fold the difference into the affine.
  const shift = origin.map((v) => v + PAD);
  const A = affine.slice();
  for (let r = 0; r < 3; r++) A[4 * r + 3] += A[4 * r] * shift[0] + A[4 * r + 1] * shift[1] + A[4 * r + 2] * shift[2];

  const sdf = signedDistance(box, size, CAVITY_SIGMA);
  // Bisect the level (voxels) until the smoothed mesh holds as many voxels as the mask.
  let lowIso = -1, highIso = 2;
  let best = gridSurface(sdf, size, 0, true, 0);
  for (let i = 0; i < 9; i++) {
    const iso = (lowIso + highIso) / 2;
    const surface = gridSurface(sdf, size, iso, true, CAVITY_TAUBIN_ITERATIONS);
    if (enclosedVolume(surface.positions, surface.indices) < count) lowIso = iso;
    else highIso = iso;
    best = surface;
  }
  const mesh = toWorld(best.positions, best.indices, A);
  const I = mesh.indices;
  for (let t = 0; t < I.length; t += 3) [I[t + 1], I[t + 2]] = [I[t + 2], I[t + 1]];
  return mesh;
}

/** NIfTI volume (already RAS) -> binary mask -> signed distance field -> surface -> vertices in world mm. */
export function volumeToMesh(
  vol: RasVolume,
  shellMm: number,
  progress: (stage: string) => void,
): { mesh: IndexedMesh; shell: IndexedMesh | null; threshold: number; head: Uint8Array } {
  progress('Thresholding');
  const threshold = otsu(vol.data);

  // Pad, so the object never touches the border.
  const [sx, sy, sz] = vol.dims;
  const dims: Dims = [sx + 2 * PAD, sy + 2 * PAD, sz + 2 * PAD];
  const padded = new Float32Array(dims[0] * dims[1] * dims[2]);
  for (let z = 0; z < sz; z++)
    for (let y = 0; y < sy; y++) {
      const from = sx * (y + sy * z);
      padded.set(vol.data.subarray(from, from + sx), PAD + dims[0] * (y + PAD + dims[1] * (z + PAD)));
    }

  progress('Building the binary mask');
  const mask = binaryMask(padded, dims, threshold);
  progress('Computing the signed distance field');
  const { mesh, shell } = surfaceFromMask(mask, dims, vol.affine, vol.spacing, shellMm, progress);
  // The head mask without its padding, for the tissue segmentation.
  const head = new Uint8Array(vol.data.length);
  for (let z = 0; z < sz; z++)
    for (let y = 0; y < sy; y++) {
      const from = PAD + dims[0] * (y + PAD + dims[1] * (z + PAD));
      head.set(mask.subarray(from, from + sx), sx * (y + sy * z));
    }
  return { mesh, shell, threshold, head };
}
