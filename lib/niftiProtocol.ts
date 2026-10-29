export interface ToNiftiWorker {
  buffer: ArrayBuffer;
}

/** What the loader reports about the volume the mesh came from. */
export interface NiftiInfo {
  sourceCode: string; // axis codes of the file as stored, e.g. "LAS"
  reoriented: boolean; // true when the volume was converted to RAS
  dims: [number, number, number];
  spacing: [number, number, number]; // mm per voxel along R, A, S
  threshold: number;
  ventricleMl: number; // volume of the segmented ventricles
}

/** The volume for the slice overlay: per voxel, windowed intensity then ventricle flag, in RAS voxel order. */
export interface SliceVolume {
  texels: Uint8Array; // 2 bytes per voxel: intensity 0..255, ventricle 0 or 255
  dims: [number, number, number];
  affine: number[]; // 4x4 row-major, voxel index -> world mm
}

export type FromNiftiWorker =
  | { type: 'progress'; stage: string }
  | {
      type: 'done';
      positions: Float32Array;
      indices: Uint32Array;
      shell: { positions: Float32Array; indices: Uint32Array } | null; // the skin with the surface 5 mm below it, as a thin solid
      ventricles: { positions: Float32Array; indices: Uint32Array } | null; // the ventricles as a solid, faces pointing out
      info: NiftiInfo;
      slices: SliceVolume;
    }
  | { type: 'error'; message: string };
