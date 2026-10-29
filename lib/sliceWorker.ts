import { countOpenEdges, mergeResult, planarCutPlanes, vertexNormals, type CutPlane, type IndexedMesh } from './planarCut';

export interface ToSliceWorker {
  mesh: IndexedMesh; // the part with its ventricle cavity
  ventricles: IndexedMesh | null; // the ventricles as a solid; null when there are none
  shell: IndexedMesh | null; // the skin and its inner surface: a thin solid shell
  planes: CutPlane[];
}

/** A closed mesh with the normals to draw it. */
export interface Piece {
  mesh: IndexedMesh;
  normals: Float32Array;
}

/** A closed piece that a slice cut away. `ventricles` is the part of the ventricles inside that piece, as a solid (null if none). */
export interface TrimmedPart {
  side: string; // the cube face it lay beyond, e.g. "xmin"
  hollow: Piece;
  ventricles: Piece | null;
  shell: Piece | null; // the part of it that lies in the surface shell
}

export type FromSliceWorker =
  | { type: 'done'; mesh: IndexedMesh; ventricles: IndexedMesh | null; shell: IndexedMesh | null; openEdges: number; trimmed: TrimmedPart[] }
  | { type: 'error'; message: string };

const ctx = self as unknown as {
  onmessage: ((e: MessageEvent<ToSliceWorker>) => void) | null;
  postMessage(message: FromSliceWorker, transfer?: Transferable[]): void;
};

/** The slab beyond plane i and inside the planes before it, or null when the cut leaves nothing. */
function slab(whole: IndexedMesh, planes: CutPlane[], i: number): Piece | null {
  const r = planarCutPlanes(whole, [...planes.slice(0, i), { ...planes[i], keepHigh: !planes[i].keepHigh }]);
  if (r.bodyIndices.length === 0) return null;
  // Body and cap share the vertex array but not the normals: a vertex takes the cap's where a cap triangle uses it.
  const normals = vertexNormals(r.positions, r.bodyIndices);
  const capNormals = vertexNormals(r.positions, r.capIndices);
  for (const v of r.capIndices) {
    normals[3 * v] = capNormals[3 * v];
    normals[3 * v + 1] = capNormals[3 * v + 1];
    normals[3 * v + 2] = capNormals[3 * v + 2];
  }
  return { mesh: mergeResult(r), normals };
}

/**
 * Cuts the mesh (ventricle cavity included) with the planes and merges the kept part; the ventricle solid is cut
 * the same way so later slices can still return it. The pieces that fall outside the cube come back as separate
 * closed parts: one per cube face that cut something, each the slab beyond that face and inside the faces before it.
 */
ctx.onmessage = (e) => {
  try {
    const { mesh, ventricles, shell, planes } = e.data;
    const merged = mergeResult(planarCutPlanes(mesh, planes));
    const openEdges = countOpenEdges(merged.indices, merged.positions.length / 3);
    const shellKept = shell ? mergeResult(planarCutPlanes(shell, planes)) : null;
    const ventriclesKept = ventricles ? mergeResult(planarCutPlanes(ventricles, planes)) : null;

    const trimmed: TrimmedPart[] = [];
    planes.forEach((plane, i) => {
      const hollow = slab(mesh, planes, i);
      if (!hollow) return;
      trimmed.push({ side: `${plane.axis}${plane.keepHigh ? 'min' : 'max'}`, hollow, ventricles: ventricles ? slab(ventricles, planes, i) : null, shell: shell ? slab(shell, planes, i) : null });
    });

    const buffers = (p: Piece | null) => (p ? [p.mesh.positions.buffer, p.mesh.indices.buffer, p.normals.buffer] : []);
    const transfer = [
      merged.positions.buffer,
      merged.indices.buffer,
      ...(ventriclesKept ? [ventriclesKept.positions.buffer, ventriclesKept.indices.buffer] : []),
      ...(shellKept ? [shellKept.positions.buffer, shellKept.indices.buffer] : []),
      ...trimmed.flatMap((t) => [...buffers(t.hollow), ...buffers(t.ventricles), ...buffers(t.shell)]),
    ];
    ctx.postMessage({ type: 'done', mesh: merged, ventricles: ventriclesKept, shell: shellKept, openEdges, trimmed }, transfer as ArrayBuffer[]);
  } catch (err) {
    ctx.postMessage({ type: 'error', message: (err as Error).message });
  }
};
