import { readNifti, toRas } from './nifti';
import { cavityFromMask, volumeToMesh } from './niftiMesh';
import { segmentVentricles, windowIntensities } from './ventricles';
import type { FromNiftiWorker, ToNiftiWorker } from './niftiProtocol';

const ctx = self as unknown as {
  onmessage: ((e: MessageEvent<ToNiftiWorker>) => void) | null;
  postMessage(message: FromNiftiWorker, transfer?: Transferable[]): void;
};

const SHELL_MM = 5; // thickness of the surface shell

function concat<T extends Float32Array | Uint32Array>(a: T, b: T): T {
  const out = new (a.constructor as new (n: number) => T)(a.length + b.length);
  out.set(a);
  out.set(b, a.length);
  return out;
}

ctx.onmessage = async (e) => {
  try {
    ctx.postMessage({ type: 'progress', stage: 'Reading the NIfTI file' });
    const raw = await readNifti(e.data.buffer);
    ctx.postMessage({ type: 'progress', stage: 'Converting to RAS' });
    const vol = toRas(raw);
    const { mesh, shell: inner, threshold, head } = volumeToMesh(vol, SHELL_MM, (stage) => ctx.postMessage({ type: 'progress', stage }));
    ctx.postMessage({ type: 'progress', stage: 'Finding the ventricles' });
    const ventricles = segmentVentricles(vol, head);
    const cavity = ventricles.volumeMl > 0 ? cavityFromMask(ventricles.mask, vol.dims, vol.affine) : null;
    // The cavity shell (faces pointing into the void) joins the mesh, so the part is hollow from the start.
    const hollow = cavity
      ? {
          positions: concat(mesh.positions, cavity.positions),
          indices: concat(mesh.indices, cavity.indices.map((i) => i + mesh.positions.length / 3)),
        }
      : mesh;
    // The same shell turned inside out is the ventricles as a solid.
    const solidVentricles = cavity
      ? {
          positions: cavity.positions.slice(),
          indices: cavity.indices.map((_, i, a) => a[i - (i % 3) + [0, 2, 1][i % 3]]),
        }
      : null;
    // Skin plus the inner surface: a solid only SHELL_MM thick. Without an inner surface the head is solid, so the skin will do.
    const shell = inner
      ? {
          positions: concat(mesh.positions, inner.positions),
          indices: concat(mesh.indices, inner.indices.map((i) => i + mesh.positions.length / 3)),
        }
      : mesh;
    const gray = windowIntensities(vol, head);
    const texels = new Uint8Array(2 * gray.length);
    for (let i = 0; i < gray.length; i++) {
      texels[2 * i] = gray[i];
      texels[2 * i + 1] = ventricles.mask[i] ? 255 : 0;
    }
    ctx.postMessage(
      {
        type: 'done',
        positions: hollow.positions,
        indices: hollow.indices,
        shell,
        ventricles: solidVentricles,
        info: { sourceCode: vol.sourceCode, reoriented: vol.reoriented, dims: vol.dims, spacing: vol.spacing, threshold, ventricleMl: ventricles.volumeMl },
        slices: { texels, dims: vol.dims, affine: vol.affine },
      },
      [hollow.positions.buffer, hollow.indices.buffer, texels.buffer, ...(shell === mesh ? [] : [shell.positions.buffer, shell.indices.buffer]), ...(solidVentricles ? [solidVentricles.positions.buffer, solidVentricles.indices.buffer] : [])],
    );
  } catch (err) {
    ctx.postMessage({ type: 'error', message: (err as Error).message });
  }
};
