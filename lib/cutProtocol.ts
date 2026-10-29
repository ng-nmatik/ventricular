import type { CutPlane, CutResult } from './planarCut';

/** Messages the main thread sends to the cut worker. */
export type ToWorker =
  | { type: 'mesh'; version: number; positions: Float32Array; indices: Uint32Array }
  | { type: 'cut'; seq: number; version: number; planes: CutPlane[] };

/** Everything the viewer needs for one finished cut, computed off the main thread. */
export interface FromWorker extends CutResult {
  seq: number;
  version: number;
  bodyNormals: Float32Array;
  capNormals: Float32Array;
  openEdges: number; // 0 means the cut mesh is closed
  planes: CutPlane[]; // the planes this result was cut with
  capBounds: Float32Array; // per plane: min xyz then max xyz of its cap, or min > max when it has none
  capAnchors: Float32Array; // per plane: the cap point nearest its bottom-right corner (see labelFrame)
}
