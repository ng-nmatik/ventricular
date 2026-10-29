import { BufferGeometry, ShapeUtils, Vector2, Vector3 } from 'three';

/** A triangle mesh with shared (welded) vertices. */
export interface IndexedMesh {
  positions: Float32Array; // xyz per vertex
  indices: Uint32Array; // 3 per triangle
}

export interface CutResult {
  positions: Float32Array; // every vertex used by body or cap
  bodyIndices: Uint32Array; // the kept part of the original surface
  capIndices: Uint32Array; // the flat surface that closes the cut
  sectionSegments: Uint32Array; // pairs of vertex ids along the cut outline
  loops: number; // number of closed outlines that were capped
  openLoops: number; // outlines that did not close (input has holes at the cut)
}

export type Axis = 'x' | 'y' | 'z';

/**
 * Turns any loaded geometry into an indexed mesh whose vertices are welded by
 * exact position. PLY files often store duplicate vertices; welding them is
 * what lets the cut outline join up into closed loops.
 */
export function weldGeometry(geometry: BufferGeometry): IndexedMesh {
  const pos = geometry.getAttribute('position');
  const index = geometry.getIndex();
  const vertexCount = pos.count;
  const triCount = Math.floor((index ? index.count : vertexCount) / 3);

  // Vertices closer than a millionth of the model size are treated as one.
  geometry.computeBoundingBox();
  const size = geometry.boundingBox!.getSize(new Vector3()).length() || 1;
  const tol = size * 1e-6;

  const lookup = new Map<string, number>();
  const remap = new Uint32Array(vertexCount);
  const out: number[] = [];
  for (let i = 0; i < vertexCount; i++) {
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const key = `${Math.round(x / tol)},${Math.round(y / tol)},${Math.round(z / tol)}`;
    let id = lookup.get(key);
    if (id === undefined) {
      id = out.length / 3;
      out.push(x, y, z);
      lookup.set(key, id);
    }
    remap[i] = id;
  }

  const indices = new Uint32Array(triCount * 3);
  let n = 0;
  for (let t = 0; t < triCount; t++) {
    const a = remap[index ? index.getX(t * 3) : t * 3];
    const b = remap[index ? index.getX(t * 3 + 1) : t * 3 + 1];
    const c = remap[index ? index.getX(t * 3 + 2) : t * 3 + 2];
    if (a === b || b === c || a === c) continue; // drop degenerate triangles
    indices[n++] = a;
    indices[n++] = b;
    indices[n++] = c;
  }

  return { positions: new Float32Array(out), indices: indices.slice(0, n) };
}

/** Plane normal from a base axis and two tilt angles (degrees). */
export function planeNormal(axis: Axis, tiltA: number, tiltB: number): Vector3 {
  const X = new Vector3(1, 0, 0);
  const Y = new Vector3(0, 1, 0);
  const Z = new Vector3(0, 0, 1);
  const [base, p1, p2] = axis === 'x' ? [X, Y, Z] : axis === 'y' ? [Y, Z, X] : [Z, X, Y];
  return base
    .clone()
    .applyAxisAngle(p1, (tiltA * Math.PI) / 180)
    .applyAxisAngle(p2, (tiltB * Math.PI) / 180)
    .normalize();
}

/** Min and max of dot(n, p) over all vertices, padded so the ends keep or remove everything. */
export function projectRange(mesh: IndexedMesh, n: Vector3): { min: number; max: number } {
  const P = mesh.positions;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < P.length; i += 3) {
    const d = n.x * P[i] + n.y * P[i + 1] + n.z * P[i + 2];
    if (d < min) min = d;
    if (d > max) max = d;
  }
  const pad = (max - min) * 1e-3 || 1e-6;
  return { min: min - pad, max: max + pad };
}

/**
 * Cuts the mesh with the plane dot(n, p) = offset. Everything on the side
 * where dot(n, p) < offset is kept, and the opening is closed with a flat cap
 * whose outward normal is +n. Cap and body share the exact same vertices on
 * the cut outline, so a watertight input gives a watertight output.
 */
export function planarCut(mesh: IndexedMesh, normal: Vector3, offset: number): CutResult {
  const P = mesh.positions;
  const I = mesh.indices;
  const nV = P.length / 3;
  const { x: nx, y: ny, z: nz } = normal;

  // Signed distance of every vertex to the plane.
  const dist = new Float64Array(nV);
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < nV; i++) {
    const d = nx * P[3 * i] + ny * P[3 * i + 1] + nz * P[3 * i + 2] - offset;
    dist[i] = d;
    if (d < lo) lo = d;
    if (d > hi) hi = d;
  }
  // Vertices within this distance count as lying on the plane: they are kept
  // and snapped exactly onto it, instead of spawning near-duplicate points.
  const eps = Math.max((hi - lo) * 1e-5, 1e-12);
  const side = new Int8Array(nV); // -1 kept, 0 on the plane, 1 removed
  for (let i = 0; i < nV; i++) side[i] = dist[i] < -eps ? -1 : dist[i] > eps ? 1 : 0;

  const outPos: number[] = [];
  const keptId = new Int32Array(nV).fill(-1);
  const edgeId = new Map<number, number>();
  const body: number[] = [];
  const rimIds: number[] = []; // vertices that lie on the plane in the output

  const keep = (i: number): number => {
    let id = keptId[i];
    if (id < 0) {
      id = outPos.length / 3;
      if (side[i] === 0) {
        const d = dist[i];
        outPos.push(P[3 * i] - d * nx, P[3 * i + 1] - d * ny, P[3 * i + 2] - d * nz);
        rimIds.push(id);
      } else {
        outPos.push(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
      }
      keptId[i] = id;
    }
    return id;
  };

  // One new vertex per crossed edge, shared by both triangles on that edge.
  const crossing = (i: number, j: number): number => {
    const a = Math.min(i, j);
    const b = Math.max(i, j);
    const key = a * nV + b;
    let id = edgeId.get(key);
    if (id === undefined) {
      const t = dist[a] / (dist[a] - dist[b]);
      id = outPos.length / 3;
      outPos.push(
        P[3 * a] + t * (P[3 * b] - P[3 * a]),
        P[3 * a + 1] + t * (P[3 * b + 1] - P[3 * a + 1]),
        P[3 * a + 2] + t * (P[3 * b + 2] - P[3 * a + 2]),
      );
      edgeId.set(key, id);
      rimIds.push(id);
    }
    return id;
  };

  const tri = [0, 0, 0];
  const poly: number[] = [];
  for (let t = 0; t < I.length; t += 3) {
    tri[0] = I[t];
    tri[1] = I[t + 1];
    tri[2] = I[t + 2];
    const s0 = side[tri[0]];
    const s1 = side[tri[1]];
    const s2 = side[tri[2]];
    const hasIn = s0 < 0 || s1 < 0 || s2 < 0;
    const hasOut = s0 > 0 || s1 > 0 || s2 > 0;

    if (!hasOut) {
      if (!hasIn) {
        // Lies flat in the plane (e.g. an earlier cap): keep it only if it faces out of the kept part.
        const a = 3 * tri[0], b = 3 * tri[1], c = 3 * tri[2];
        const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
        const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
        const facing = nx * (uy * vz - uz * vy) + ny * (uz * vx - ux * vz) + nz * (ux * vy - uy * vx);
        if (facing <= 0) continue;
      }
      body.push(keep(tri[0]), keep(tri[1]), keep(tri[2]));
      continue;
    }
    if (!hasIn) continue;

    // Clip the triangle (Sutherland–Hodgman); keeps the original winding.
    poly.length = 0;
    for (let k = 0; k < 3; k++) {
      const i = tri[k];
      const j = tri[(k + 1) % 3];
      if (side[i] <= 0) poly.push(keep(i));
      if (side[i] * side[j] === -1) poly.push(crossing(i, j));
    }
    for (let k = 1; k < poly.length - 1; k++) body.push(poly[0], poly[k], poly[k + 1]);
  }

  // Weld rim vertices that landed on top of each other (happens when the
  // plane crosses very thin triangles), then drop triangles that collapse.
  const nOut = outPos.length / 3;
  const remap = new Int32Array(nOut);
  for (let i = 0; i < nOut; i++) remap[i] = i;
  const grid = new Map<string, number[]>();
  for (const id of rimIds) {
    const x = outPos[3 * id], y = outPos[3 * id + 1], z = outPos[3 * id + 2];
    const cx = Math.floor(x / eps), cy = Math.floor(y / eps), cz = Math.floor(z / eps);
    let match = -1;
    search: for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dz = -1; dz <= 1; dz++) {
          const cellIds = grid.get(`${cx + dx},${cy + dy},${cz + dz}`);
          if (!cellIds) continue;
          for (const o of cellIds) {
            const ex = outPos[3 * o] - x, ey = outPos[3 * o + 1] - y, ez = outPos[3 * o + 2] - z;
            if (ex * ex + ey * ey + ez * ez <= eps * eps) {
              match = o;
              break search;
            }
          }
        }
    if (match >= 0) remap[id] = match;
    else {
      const key = `${cx},${cy},${cz}`;
      (grid.get(key) ?? grid.set(key, []).get(key)!).push(id);
    }
  }
  let w = 0;
  for (let t = 0; t < body.length; t += 3) {
    const a = remap[body[t]], b = remap[body[t + 1]], c = remap[body[t + 2]];
    if (a === b || b === c || a === c) continue;
    body[w++] = a;
    body[w++] = b;
    body[w++] = c;
  }
  body.length = w;

  // The cut outline is every edge on the plane that only one kept triangle uses.
  const onRim = new Uint8Array(nOut);
  for (const id of rimIds) onRim[remap[id]] = 1;
  const rimEdges = new Map<number, number>();
  for (let t = 0; t < body.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      const a = body[t + k];
      const b = body[t + ((k + 1) % 3)];
      if (!onRim[a] || !onRim[b]) continue;
      const key = Math.min(a, b) * nOut + Math.max(a, b);
      rimEdges.set(key, (rimEdges.get(key) ?? 0) + 1);
    }
  }
  const adjacency = new Map<number, number[]>();
  const link = (a: number, b: number) => {
    (adjacency.get(a) ?? adjacency.set(a, []).get(a)!).push(b);
    (adjacency.get(b) ?? adjacency.set(b, []).get(b)!).push(a);
  };
  for (const [key, count] of rimEdges) {
    if (count === 1) link(Math.floor(key / nOut), key % nOut);
  }

  // Chain the cut segments into outlines. Walking edges (not vertices) and
  // splitting off a loop whenever the path meets itself handles outlines that
  // touch at a single point, e.g. when the plane is tangent to the surface.
  const usedEdge = new Set<number>();
  const loops: { ids: number[]; closed: boolean }[] = [];
  const nextEdge = (v: number): number => {
    for (const nb of adjacency.get(v)!) {
      const key = Math.min(v, nb) * nOut + Math.max(v, nb);
      if (!usedEdge.has(key)) {
        usedEdge.add(key);
        return nb;
      }
    }
    return -1;
  };
  const walk = (start: number) => {
    const path = [start];
    const at = new Map<number, number>([[start, 0]]);
    let cur = start;
    for (;;) {
      const nb = nextEdge(cur);
      if (nb < 0) break;
      const k = at.get(nb);
      if (k !== undefined) {
        loops.push({ ids: path.slice(k), closed: true });
        for (const v of path.slice(k + 1)) at.delete(v);
        path.length = k + 1;
      } else {
        at.set(nb, path.length);
        path.push(nb);
      }
      cur = nb;
    }
    if (path.length >= 2) loops.push({ ids: path, closed: false });
  };
  const hasFreeEdge = (v: number) =>
    adjacency.get(v)!.some((nb) => !usedEdge.has(Math.min(v, nb) * nOut + Math.max(v, nb)));
  for (const [v, nbs] of adjacency) if (nbs.length % 2 === 1 && hasFreeEdge(v)) walk(v); // open chains first
  for (const v of adjacency.keys()) while (hasFreeEdge(v)) walk(v);

  const capIndices = triangulateCap(outPos, loops.filter((l) => l.ids.length >= 3), normal);

  const segments: number[] = [];
  for (const [a, nbs] of adjacency) for (const b of nbs) if (a < b) segments.push(a, b);

  // Drop vertices nothing uses any more and renumber.
  const used = new Int32Array(outPos.length / 3).fill(-1);
  let count = 0;
  const renumber = (list: number[]) => {
    const outList = new Uint32Array(list.length);
    for (let i = 0; i < list.length; i++) {
      const id = list[i];
      if (used[id] < 0) used[id] = count++;
      outList[i] = used[id];
    }
    return outList;
  };
  const bodyIndices = renumber(body);
  const capOut = renumber(capIndices);
  const sectionSegments = renumber(segments);
  const positions = new Float32Array(count * 3);
  for (let id = 0; id < used.length; id++) {
    const j = used[id];
    if (j < 0) continue;
    positions[3 * j] = outPos[3 * id];
    positions[3 * j + 1] = outPos[3 * id + 1];
    positions[3 * j + 2] = outPos[3 * id + 2];
  }

  return {
    positions,
    bodyIndices,
    capIndices: capOut,
    sectionSegments,
    loops: loops.filter((l) => l.closed).length,
    openLoops: loops.filter((l) => !l.closed && l.ids.length >= 2).length,
  };
}

const AXIS_NORMAL: Record<Axis, [number, number, number]> = {
  x: [1, 0, 0],
  y: [0, 1, 0],
  z: [0, 0, 1],
};

/**
 * Directions for text lying on a cut, as seen from outside the part: `normal` points out of it,
 * `up` is the model's +Z (or +Y for cuts across Z) and `right` completes a right-handed frame.
 */
export function labelFrame(plane: { axis: Axis; keepHigh: boolean }) {
  const sign = plane.keepHigh ? -1 : 1;
  const [nx, ny, nz] = AXIS_NORMAL[plane.axis].map((c) => c * sign);
  const [ux, uy, uz] = plane.axis === 'z' ? [0, 1, 0] : [0, 0, 1];
  return {
    normal: [nx, ny, nz],
    up: [ux, uy, uz],
    right: [uy * nz - uz * ny, uz * nx - ux * nz, ux * ny - uy * nx], // up × normal
  };
}

/** Keeps the side of the plane `axis = offset` where the coordinate is higher (or lower). */
export interface CutPlane {
  axis: Axis;
  keepHigh: boolean;
  offset: number;
}

/**
 * Cuts with a list of axis-aligned planes in turn, so the result is the
 * intersection of the kept sides (e.g. the inside of a box). Each step caps its
 * own cut; afterwards every triangle lying flat on one of the planes is
 * reported as cap, so caps from earlier planes keep their colour.
 */
export function planarCutPlanes(mesh: IndexedMesh, planes: CutPlane[]): CutResult {
  let current = mesh;
  let last: CutResult | null = null;
  let loops = 0;
  let openLoops = 0;
  for (const { axis, keepHigh, offset } of planes) {
    const sign = keepHigh ? -1 : 1;
    const [x, y, z] = AXIS_NORMAL[axis];
    last = planarCut(current, new Vector3(x * sign, y * sign, z * sign), offset * sign);
    loops += last.loops;
    openLoops += last.openLoops;
    if (last.bodyIndices.length === 0) break;
    current = mergeResult(last);
  }
  if (!last) {
    return {
      positions: mesh.positions,
      bodyIndices: mesh.indices,
      capIndices: new Uint32Array(0),
      sectionSegments: new Uint32Array(0),
      loops: 0,
      openLoops: 0,
    };
  }
  if (last.bodyIndices.length === 0) return { ...last, loops, openLoops };

  const { positions } = last;
  const all = mergeResult(last).indices;
  const nV = positions.length / 3;

  // Flat tolerance: a hair above the snap distance planarCut uses.
  const probe = projectRange({ positions, indices: all }, new Vector3(1, 1, 1).normalize());
  const tol = (probe.max - probe.min) * 1e-4; // ~1e-4 of the model size
  const axisIndex: Record<Axis, number> = { x: 0, y: 1, z: 2 };

  // Bit i of a vertex's mask is set when it lies on plane i.
  const mask = new Uint8Array(nV);
  for (let v = 0; v < nV; v++) {
    for (let i = 0; i < planes.length; i++) {
      if (Math.abs(positions[3 * v + axisIndex[planes[i].axis]] - planes[i].offset) <= tol) mask[v] |= 1 << i;
    }
  }
  const body: number[] = [];
  const cap: number[] = [];
  for (let t = 0; t < all.length; t += 3) {
    const a = all[t], b = all[t + 1], c = all[t + 2];
    // A cap triangle has all three corners on the same plane.
    (mask[a] & mask[b] & mask[c] ? cap : body).push(a, b, c);
  }

  // Outline: edges shared by a cap triangle and a body triangle.
  const edgeKinds = new Map<number, number>();
  const note = (list: number[], bit: number) => {
    for (let t = 0; t < list.length; t += 3) {
      for (let k = 0; k < 3; k++) {
        const u = list[t + k];
        const w = list[t + ((k + 1) % 3)];
        const key = Math.min(u, w) * nV + Math.max(u, w);
        edgeKinds.set(key, (edgeKinds.get(key) ?? 0) | bit);
      }
    }
  };
  note(cap, 1);
  note(body, 2);
  const segments: number[] = [];
  for (const [key, kinds] of edgeKinds) if (kinds === 3) segments.push(Math.floor(key / nV), key % nV);

  return {
    positions,
    bodyIndices: Uint32Array.from(body),
    capIndices: Uint32Array.from(cap),
    sectionSegments: Uint32Array.from(segments),
    loops,
    openLoops,
  };
}

/**
 * Fills the outlines with triangles. Outlines nested inside other outlines
 * (an inner cavity, for instance) become holes; nesting is decided by
 * containment, so it works even if the input's winding is inconsistent.
 */
function triangulateCap(
  pos: number[],
  loops: { ids: number[] }[],
  n: Vector3,
): number[] {
  if (loops.length === 0) return [];

  // 2D basis on the plane with u × v = n.
  const helper = Math.abs(n.x) < 0.9 ? new Vector3(1, 0, 0) : new Vector3(0, 1, 0);
  const u = new Vector3().crossVectors(helper, n).normalize();
  const v = new Vector3().crossVectors(n, u);
  const to2D = (id: number) =>
    new Vector2(
      u.x * pos[3 * id] + u.y * pos[3 * id + 1] + u.z * pos[3 * id + 2],
      v.x * pos[3 * id] + v.y * pos[3 * id + 1] + v.z * pos[3 * id + 2],
    );

  const rings = loops
    .map((l) => {
      const ids = l.ids.slice();
      const pts = ids.map(to2D);
      if (pts.length > 3 && pts[0].equals(pts[pts.length - 1])) {
        pts.pop();
        ids.pop();
      }
      return { ids, pts, area: Math.abs(ShapeUtils.area(pts)), depth: 0, parent: -1 };
    })
    .sort((a, b) => b.area - a.area);

  for (let i = 0; i < rings.length; i++) {
    for (let j = 0; j < i; j++) {
      if (pointInPolygon(rings[i].pts[0], rings[j].pts)) {
        rings[i].depth++;
        rings[i].parent = j; // larger rings come first, so the last hit is the tightest
      }
    }
  }

  const out: number[] = [];
  rings.forEach((outer, oi) => {
    if (outer.depth % 2 !== 0) return; // this ring is a hole
    const holes = rings.filter((r) => r.parent === oi && r.depth === outer.depth + 1);
    const allIds = [...outer.ids, ...holes.flatMap((h) => h.ids)];
    const allPts = [...outer.pts, ...holes.flatMap((h) => h.pts)];
    const faces = ShapeUtils.triangulateShape(
      outer.pts.slice(),
      holes.map((h) => h.pts.slice()),
    );

    // Earcut skips points that are exactly collinear with their neighbours
    // (common when a second cut crosses an earlier flat cap). Find them so
    // they can be stitched back in; otherwise the mesh would have T-junctions.
    const used = new Set<number>();
    for (const f of faces) for (const i of f) used.add(allIds[i]);
    const skipped = new Map<string, number[]>();
    for (const ring of [outer, ...holes]) {
      const ids = ring.ids;
      const first = ids.findIndex((id) => used.has(id));
      if (first < 0) continue;
      let from = ids[first];
      let between: number[] = [];
      for (let s = 1; s <= ids.length; s++) {
        const id = ids[(first + s) % ids.length];
        if (!used.has(id)) {
          between.push(id);
          continue;
        }
        if (between.length) {
          skipped.set(`${from},${id}`, between);
          skipped.set(`${id},${from}`, between.slice().reverse());
        }
        from = id;
        between = [];
      }
    }

    for (const [ia, ib, ic] of faces) {
      const pa = allPts[ia];
      const pb = allPts[ib];
      const pc = allPts[ic];
      const area2 = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x);
      // Positive area in (u, v) means the face normal is +n, pointing out of the kept part.
      const tri = area2 >= 0 ? [allIds[ia], allIds[ib], allIds[ic]] : [allIds[ia], allIds[ic], allIds[ib]];

      const poly: number[] = [];
      for (let k = 0; k < 3; k++) {
        poly.push(tri[k]);
        const extra = skipped.get(`${tri[k]},${tri[(k + 1) % 3]}`);
        if (extra) poly.push(...extra);
      }
      if (poly.length === 3) {
        out.push(tri[0], tri[1], tri[2]);
        continue;
      }
      // Fan the triangle around its centroid so every outline point is a corner.
      const centre = pos.length / 3;
      pos.push(
        (pos[3 * tri[0]] + pos[3 * tri[1]] + pos[3 * tri[2]]) / 3,
        (pos[3 * tri[0] + 1] + pos[3 * tri[1] + 1] + pos[3 * tri[2] + 1]) / 3,
        (pos[3 * tri[0] + 2] + pos[3 * tri[1] + 2] + pos[3 * tri[2] + 2]) / 3,
      );
      for (let k = 0; k < poly.length; k++) out.push(poly[k], poly[(k + 1) % poly.length], centre);
    }
  });
  return out;
}

function pointInPolygon(p: Vector2, poly: Vector2[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i];
    const b = poly[j];
    if (a.y > p.y !== b.y > p.y && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) {
      inside = !inside;
    }
  }
  return inside;
}

/** Counts edges not shared by exactly two triangles. Zero means the mesh is closed. */
export function countOpenEdges(indices: Uint32Array, vertexCount: number): number {
  const counts = new Map<number, number>();
  for (let t = 0; t < indices.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      const a = indices[t + k];
      const b = indices[t + ((k + 1) % 3)];
      const key = Math.min(a, b) * vertexCount + Math.max(a, b);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
  let open = 0;
  for (const c of counts.values()) if (c !== 2) open++;
  return open;
}

/** Area-weighted vertex normals, the same result as BufferGeometry.computeVertexNormals. */
export function vertexNormals(P: Float32Array, I: Uint32Array): Float32Array {
  const N = new Float32Array(P.length);
  for (let t = 0; t < I.length; t += 3) {
    const a = 3 * I[t], b = 3 * I[t + 1], c = 3 * I[t + 2];
    const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
    const vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    for (const i of [a, b, c]) {
      N[i] += nx;
      N[i + 1] += ny;
      N[i + 2] += nz;
    }
  }
  for (let i = 0; i < N.length; i += 3) {
    const len = Math.hypot(N[i], N[i + 1], N[i + 2]) || 1;
    N[i] /= len;
    N[i + 1] /= len;
    N[i + 2] /= len;
  }
  return N;
}

/** Merges a cut result into one indexed mesh (for export or further cuts). */
export function mergeResult(r: CutResult): IndexedMesh {
  const indices = new Uint32Array(r.bodyIndices.length + r.capIndices.length);
  indices.set(r.bodyIndices, 0);
  indices.set(r.capIndices, r.bodyIndices.length);
  return { positions: r.positions, indices };
}

/**
 * Makes every triangle face outward. Scans often contain flipped triangles,
 * which render as see-through holes and confuse slicers. Neighbouring faces
 * are made consistent by walking across shared edges, then any piece whose
 * volume comes out negative (inside-out) is flipped as a whole.
 */
export function orientOutward(mesh: IndexedMesh): { mesh: IndexedMesh; flipped: number } {
  const P = mesh.positions;
  const I = mesh.indices.slice();
  const nT = I.length / 3;
  const nV = P.length / 3;
  const original = mesh.indices;

  // Neighbour across each edge slot (edge k of triangle t joins corners k and k+1).
  const nb = new Int32Array(nT * 3).fill(-1);
  const open = new Map<number, number>();
  for (let t = 0; t < nT; t++) {
    for (let k = 0; k < 3; k++) {
      const a = I[3 * t + k];
      const b = I[3 * t + ((k + 1) % 3)];
      const key = Math.min(a, b) * nV + Math.max(a, b);
      const other = open.get(key);
      if (other === undefined) open.set(key, 3 * t + k);
      else {
        nb[3 * t + k] = Math.floor(other / 3);
        nb[other] = t;
        open.delete(key);
      }
    }
  }

  const flip = (t: number) => {
    const tmp = I[3 * t + 1];
    I[3 * t + 1] = I[3 * t + 2];
    I[3 * t + 2] = tmp;
    // Corners (v0, v2, v1): edge slots 0 and 2 swap places.
    const s = nb[3 * t];
    nb[3 * t] = nb[3 * t + 2];
    nb[3 * t + 2] = s;
  };

  const seen = new Uint8Array(nT);
  const queue = new Int32Array(nT);
  for (let seed = 0; seed < nT; seed++) {
    if (seen[seed]) continue;
    seen[seed] = 1;
    let head = 0;
    let tail = 0;
    queue[tail++] = seed;
    while (head < tail) {
      const t = queue[head++];
      for (let k = 0; k < 3; k++) {
        const n = nb[3 * t + k];
        if (n < 0 || seen[n]) continue;
        const a = I[3 * t + k];
        const b = I[3 * t + ((k + 1) % 3)];
        // A consistent neighbour runs the shared edge the other way (b → a).
        for (let j = 0; j < 3; j++) {
          if (I[3 * n + j] === a && I[3 * n + ((j + 1) % 3)] === b) {
            flip(n);
            break;
          }
        }
        seen[n] = 1;
        queue[tail++] = n;
      }
    }

    // Signed volume of this connected piece, measured from its centroid.
    let cx = 0, cy = 0, cz = 0;
    for (let q = 0; q < tail; q++) {
      const t = queue[q];
      for (let k = 0; k < 3; k++) {
        const v = I[3 * t + k];
        cx += P[3 * v];
        cy += P[3 * v + 1];
        cz += P[3 * v + 2];
      }
    }
    const m = tail * 3;
    cx /= m;
    cy /= m;
    cz /= m;
    let vol = 0;
    for (let q = 0; q < tail; q++) {
      const t = queue[q];
      const a = 3 * I[3 * t], b = 3 * I[3 * t + 1], c = 3 * I[3 * t + 2];
      const ax = P[a] - cx, ay = P[a + 1] - cy, az = P[a + 2] - cz;
      const bx = P[b] - cx, by = P[b + 1] - cy, bz = P[b + 2] - cz;
      const qx = P[c] - cx, qy = P[c + 1] - cy, qz = P[c + 2] - cz;
      vol += ax * (by * qz - bz * qy) - ay * (bx * qz - bz * qx) + az * (bx * qy - by * qx);
    }
    if (vol < 0) for (let q = 0; q < tail; q++) flip(queue[q]);
  }

  let flipped = 0;
  for (let t = 0; t < nT; t++) if (I[3 * t + 1] !== original[3 * t + 1]) flipped++;
  return { mesh: { positions: P, indices: I }, flipped };
}
