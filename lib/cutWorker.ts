import { countOpenEdges, labelFrame, mergeResult, planarCutPlanes, vertexNormals, type CutPlane, type IndexedMesh } from './planarCut';
import type { FromWorker, ToWorker } from './cutProtocol';

const ctx = self as unknown as {
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null;
  postMessage(message: FromWorker, transfer: Transferable[]): void;
};

let mesh: IndexedMesh | null = null;

/**
 * For each plane, the bounding box of its cap triangles and the cap vertex nearest the cap's
 * bottom-right corner (as seen from outside), so a label can sit inside the cap near that corner.
 */
function capLayout(positions: Float32Array, capIndices: Uint32Array, planes: CutPlane[]) {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < positions.length; i++) {
    lo = Math.min(lo, positions[i]);
    hi = Math.max(hi, positions[i]);
  }
  const tol = (hi - lo) * 1e-4;
  const axisIndex = { x: 0, y: 1, z: 2 };
  const bounds = new Float32Array(planes.length * 6);
  for (let i = 0; i < planes.length; i++) bounds.set([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity], i * 6);
  const onPlane: Set<number>[] = planes.map(() => new Set());

  for (let t = 0; t < capIndices.length; t += 3) {
    const corners = [capIndices[t], capIndices[t + 1], capIndices[t + 2]];
    const i = planes.findIndex((p) => {
      const k = axisIndex[p.axis];
      return corners.every((v) => Math.abs(positions[3 * v + k] - p.offset) <= tol);
    });
    if (i < 0) continue;
    for (const v of corners) {
      onPlane[i].add(v);
      for (let k = 0; k < 3; k++) {
        bounds[i * 6 + k] = Math.min(bounds[i * 6 + k], positions[3 * v + k]);
        bounds[i * 6 + 3 + k] = Math.max(bounds[i * 6 + 3 + k], positions[3 * v + k]);
      }
    }
  }

  const anchors = new Float32Array(planes.length * 3);
  planes.forEach((plane, i) => {
    const { right, up } = labelFrame(plane);
    const along = (v: number, dir: number[]) =>
      positions[3 * v] * dir[0] + positions[3 * v + 1] * dir[1] + positions[3 * v + 2] * dir[2];
    let maxRight = -Infinity;
    let minUp = Infinity;
    for (const v of onPlane[i]) {
      maxRight = Math.max(maxRight, along(v, right));
      minUp = Math.min(minUp, along(v, up));
    }
    let best = -1;
    let bestDist = Infinity;
    for (const v of onPlane[i]) {
      const dist = (along(v, right) - maxRight) ** 2 + (along(v, up) - minUp) ** 2;
      if (dist < bestDist) {
        bestDist = dist;
        best = v;
      }
    }
    if (best >= 0) anchors.set([positions[3 * best], positions[3 * best + 1], positions[3 * best + 2]], i * 3);
  });
  return { bounds, anchors };
}

ctx.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'mesh') {
    mesh = { positions: m.positions, indices: m.indices };
    return;
  }
  if (!mesh) return;

  let r = planarCutPlanes(mesh, m.planes);
  // With no planes the result points at the stored mesh; copy it so transferring it back doesn't detach ours.
  if (r.positions === mesh.positions) r = { ...r, positions: r.positions.slice(), bodyIndices: r.bodyIndices.slice() };

  const merged = mergeResult(r);
  const layout = capLayout(r.positions, r.capIndices, m.planes);
  const out: FromWorker = {
    ...r,
    seq: m.seq,
    version: m.version,
    bodyNormals: vertexNormals(r.positions, r.bodyIndices),
    capNormals: vertexNormals(r.positions, r.capIndices),
    openEdges: r.bodyIndices.length === 0 ? 0 : countOpenEdges(merged.indices, r.positions.length / 3),
    planes: m.planes,
    capBounds: layout.bounds,
    capAnchors: layout.anchors,
  };
  const buffers = [out.positions, out.bodyIndices, out.capIndices, out.sectionSegments, out.bodyNormals, out.capNormals, out.capBounds, out.capAnchors];
  ctx.postMessage(out, buffers.map((b) => b.buffer as ArrayBuffer));
};
