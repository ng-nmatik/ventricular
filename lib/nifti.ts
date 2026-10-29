/** A scalar volume with the affine that maps its voxel indices (i, j, k) to world millimetres. */
export interface Volume {
  data: Float32Array; // i fastest, then j, then k
  dims: [number, number, number];
  affine: number[]; // 4x4, row-major
}

export interface RasVolume extends Volume {
  sourceCode: string; // axis codes of the file as stored, e.g. "LAS"
  reoriented: boolean; // false when the file already was RAS
  spacing: [number, number, number]; // mm per voxel along each RAS axis
}

async function gunzip(buffer: ArrayBuffer): Promise<ArrayBuffer> {
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

/** Reads a NIfTI-1 file (.nii or .nii.gz): the first volume, scaled to real values, with its voxel-to-world affine. */
export async function readNifti(input: ArrayBuffer): Promise<Volume> {
  let buffer = input;
  const head = new Uint8Array(buffer, 0, Math.min(2, buffer.byteLength));
  if (head[0] === 0x1f && head[1] === 0x8b) buffer = await gunzip(buffer);
  if (buffer.byteLength < 348) throw new Error('The file is too short to be a NIfTI image.');

  const view = new DataView(buffer);
  let le = true;
  if (view.getInt32(0, true) !== 348) {
    if (view.getInt32(0, false) === 348) le = false;
    else if (view.getInt32(0, true) === 540 || view.getInt32(0, false) === 540) throw new Error('NIfTI-2 files are not supported; convert to NIfTI-1.');
    else throw new Error('This is not a NIfTI-1 file.');
  }
  const i16 = (o: number) => view.getInt16(o, le);
  const f32 = (o: number) => view.getFloat32(o, le);

  const ndim = i16(40);
  if (ndim < 3) throw new Error('The NIfTI image is not a 3D volume.');
  const dims: [number, number, number] = [i16(42), i16(44), i16(46)];
  const datatype = i16(70);
  const pixdim = [0, 1, 2, 3].map((i) => f32(76 + 4 * i));
  const voxOffset = Math.round(f32(108));
  const slope = f32(112) || 1;
  const inter = f32(116) || 0;
  const qformCode = i16(252);
  const sformCode = i16(254);

  const bytes: Record<number, number> = { 2: 1, 4: 2, 8: 4, 16: 4, 64: 8, 256: 1, 512: 2, 768: 4 };
  const size = bytes[datatype];
  if (!size) throw new Error(`Unsupported NIfTI data type ${datatype}.`);
  const count = dims[0] * dims[1] * dims[2];
  if (voxOffset + count * size > buffer.byteLength) throw new Error('The NIfTI file is truncated.');

  const read: (o: number) => number = {
    2: (o: number) => view.getUint8(o),
    4: (o: number) => view.getInt16(o, le),
    8: (o: number) => view.getInt32(o, le),
    16: (o: number) => view.getFloat32(o, le),
    64: (o: number) => view.getFloat64(o, le),
    256: (o: number) => view.getInt8(o),
    512: (o: number) => view.getUint16(o, le),
    768: (o: number) => view.getUint32(o, le),
  }[datatype]!;
  const data = new Float32Array(count);
  for (let i = 0, o = voxOffset; i < count; i++, o += size) data[i] = read(o) * slope + inter;

  let affine: number[];
  if (sformCode > 0) {
    affine = [...[280, 296, 312].flatMap((o) => [0, 1, 2, 3].map((i) => f32(o + 4 * i))), 0, 0, 0, 1];
  } else if (qformCode > 0) {
    const b = f32(256), c = f32(260), d = f32(264);
    const a = Math.sqrt(Math.max(0, 1 - (b * b + c * c + d * d)));
    const qfac = pixdim[0] < 0 ? -1 : 1;
    const R = [
      [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
      [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
      [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b],
    ];
    const scale = [pixdim[1], pixdim[2], pixdim[3] * qfac];
    const q = [268, 272, 276].map(f32);
    affine = [...R.flatMap((row, r) => [row[0] * scale[0], row[1] * scale[1], row[2] * scale[2], q[r]]), 0, 0, 0, 1];
  } else {
    affine = [pixdim[1], 0, 0, 0, 0, pixdim[2], 0, 0, 0, 0, pixdim[3], 0, 0, 0, 0, 1];
  }
  return { data, dims, affine };
}

const PERMUTATIONS = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
const CODES = [['L', 'R'], ['P', 'A'], ['I', 'S']]; // [negative, positive] along each world axis

/**
 * Reorders and flips the voxel axes so that i runs toward Right, j toward Anterior and k toward Superior
 * (the affine is updated to match, so every voxel keeps its world position).
 */
export function toRas(vol: Volume): RasVolume {
  const A = vol.affine;
  const [nx, ny, nz] = vol.dims;

  // Each voxel axis goes to the world axis it mostly points along.
  let perm = PERMUTATIONS[0];
  let best = -1;
  for (const p of PERMUTATIONS) {
    const score = Math.abs(A[p[0] * 4]) + Math.abs(A[p[1] * 4 + 1]) + Math.abs(A[p[2] * 4 + 2]);
    if (score > best) {
      best = score;
      perm = p;
    }
  }
  const flip = [0, 1, 2].map((c) => A[perm[c] * 4 + c] < 0);
  const sourceCode = [0, 1, 2].map((c) => CODES[perm[c]][flip[c] ? 0 : 1]).join('');
  const reoriented = perm.some((w, c) => w !== c) || flip.some(Boolean);

  const inv = [0, 0, 0];
  perm.forEach((w, c) => (inv[w] = c)); // world axis -> voxel axis
  const stride = [1, nx, nx * ny];
  const dims: [number, number, number] = [vol.dims[inv[0]], vol.dims[inv[1]], vol.dims[inv[2]]];

  // old = T * new, so the new affine is A * T.
  const T = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1].map(() => 0);
  T[15] = 1;
  for (let w = 0; w < 3; w++) {
    const c = inv[w];
    T[c * 4 + w] = flip[c] ? -1 : 1;
    T[c * 4 + 3] = flip[c] ? vol.dims[c] - 1 : 0;
  }
  const affine = new Array(16).fill(0);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) for (let k = 0; k < 4; k++) affine[r * 4 + c] += A[r * 4 + k] * T[k * 4 + c];

  const spacing = [0, 1, 2].map((w) => Math.hypot(affine[w], affine[4 + w], affine[8 + w])) as [number, number, number];
  if (!reoriented) return { ...vol, sourceCode, reoriented, spacing };

  const offsets = [0, 1, 2].map((w) => {
    const c = inv[w];
    const n = dims[w];
    const table = new Int32Array(n);
    for (let i = 0; i < n; i++) table[i] = (flip[c] ? n - 1 - i : i) * stride[c];
    return table;
  });
  const data = new Float32Array(vol.data.length);
  let o = 0;
  for (let z = 0; z < dims[2]; z++)
    for (let y = 0; y < dims[1]; y++) {
      const base = offsets[2][z] + offsets[1][y];
      for (let x = 0; x < dims[0]; x++) data[o++] = vol.data[base + offsets[0][x]];
    }
  return { data, dims, affine, sourceCode, reoriented, spacing };
}
