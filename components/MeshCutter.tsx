'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { ArcballControls } from 'three/examples/jsm/controls/ArcballControls.js';
import { mergeGeometries, mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { Font } from 'three/examples/jsm/loaders/FontLoader.js';
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry.js';
import barlowSemiBold from '@/lib/fonts/barlow-semibold.typeface.json';
import barlowLight from '@/lib/fonts/barlow-light.typeface.json';
import { PLYExporter } from 'three/examples/jsm/exporters/PLYExporter.js';
import Mark from './Mark';
import {
  type Axis,
  type CutPlane,
  type CutResult,
  type IndexedMesh,
  labelFrame,
  countOpenEdges,
  mergeResult,
  projectRange,
} from '@/lib/planarCut';
import type { FromWorker, ToWorker } from '@/lib/cutProtocol';
import type { FromSliceWorker, Piece, TrimmedPart } from '@/lib/sliceWorker';
import type { FromNiftiWorker, NiftiInfo, SliceVolume } from '@/lib/niftiProtocol';

const BODY_COLOR = 0xd9d3c7;
const CAP_COLOR = 0x5b9bd5; // accent blue: the cube
const CUT_COLOR = 0x8a8d92; // the cut surface
const OUTLINE_COLOR = 0x2b2e33;
const LABEL_TEXT = 'NMATIKLABS';
const LABEL_DEPTH = 0.06; // extrusion of the label geometry, in its own units (one unit = type size)
const LABEL_COLOR = 0x15171a; // ink: reads on both the gray cut and the beige of an applied cut

interface View {
  direction: THREE.Vector3; // from the model toward the camera
  up: THREE.Vector3; // which model axis points up on screen
}

// Scans are Z-up, so the side views keep +Z up; the Z view looks down from above with +Y up.
const AXIS_VIEW: Record<Axis, View> = {
  x: { direction: new THREE.Vector3(1, 0, 0), up: new THREE.Vector3(0, 0, 1) },
  y: { direction: new THREE.Vector3(0, 1, 0), up: new THREE.Vector3(0, 0, 1) },
  z: { direction: new THREE.Vector3(0, 0, 1), up: new THREE.Vector3(0, 1, 0) },
};
// The initial pose (and Recenter) puts the +Y axis facing the camera.
const DEFAULT_VIEW = AXIS_VIEW.y;


// Robotics convention from the brand guidelines: X red, Y green, Z blue (dark-ground values).
const AXIS_COLOR: Record<Axis, number> = { x: 0xf0565b, y: 0x46c27a, z: 0x5b9be6 };
const GIZMO_SIZE = 112; // px, bottom-right corner of the viewport
const GIZMO_MARGIN = 16;

/** Small X/Y/Z triad that is drawn in a corner and turns with the camera. */
function makeAxesGizmo() {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
  const up = new THREE.Vector3(0, 1, 0);
  for (const a of AXES) {
    const dir = AXIS_VECTOR[a];
    const material = new THREE.MeshBasicMaterial({ color: AXIS_COLOR[a] });
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.85, 12), material);
    shaft.position.y = 0.425;
    const head = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.28, 16), material);
    head.position.y = 0.99;
    const arrow = new THREE.Group();
    arrow.add(shaft, head);
    arrow.quaternion.setFromUnitVectors(up, dir);

    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 64;
    const ctx = canvas.getContext('2d')!;
    ctx.font = '600 44px Barlow, "Helvetica Neue", Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = `#${AXIS_COLOR[a].toString(16).padStart(6, '0')}`;
    ctx.fillText(a.toUpperCase(), 32, 34);
    const label = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false, transparent: true }),
    );
    label.position.copy(dir).multiplyScalar(1.5);
    label.scale.setScalar(0.5);
    label.renderOrder = 1;
    scene.add(arrow, label);
  }
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(0.08, 12, 12), new THREE.MeshBasicMaterial({ color: 0xa7a9ac })));
  return { scene, camera };
}

const AXES: Axis[] = ['x', 'y', 'z'];
const AXIS_VECTOR: Record<Axis, THREE.Vector3> = {
  x: new THREE.Vector3(1, 0, 0),
  y: new THREE.Vector3(0, 1, 0),
  z: new THREE.Vector3(0, 0, 1),
};

/** The cut cube: everything outside it is removed. Absolute model coordinates. */
interface Box {
  min: Record<Axis, number>;
  max: Record<Axis, number>;
}
type Side = 'min' | 'max';
type Ranges = Record<Axis, { min: number; max: number }>;

const boxFromRanges = (r: Ranges): Box => ({
  min: { x: r.x.min, y: r.y.min, z: r.z.min },
  max: { x: r.x.max, y: r.y.max, z: r.z.max },
});

const GRIP_SCALE = 0.014; // arrow half-length is 1.5x this fraction of its distance from the camera

// A glossy bead: soft diffuse light from the upper left of the view, a small highlight and a rim glow.
const GRIP_VERTEX = `
  varying vec3 vNormal;
  varying vec3 vView;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vNormal = normalize(normalMatrix * normal);
    vView = -mv.xyz;
    gl_Position = projectionMatrix * mv;
  }
`;
const GRIP_FRAGMENT = `
  uniform vec3 uColor;
  uniform float uGlow;
  varying vec3 vNormal;
  varying vec3 vView;
  void main() {
    vec3 n = normalize(vNormal);
    vec3 v = normalize(vView);
    vec3 l = normalize(vec3(-0.45, 0.6, 0.65));
    float diffuse = max(dot(n, l), 0.0);
    float specular = pow(max(dot(n, normalize(l + v)), 0.0), 56.0);
    float rim = pow(1.0 - max(dot(n, v), 0.0), 2.4);
    vec3 col = uColor * (0.32 + 0.8 * diffuse) + vec3(specular) * 0.9 + vec3(rim) * 0.4 + uColor * uGlow * 0.4;
    gl_FragColor = vec4(col, 1.0);
    #include <colorspace_fragment>
  }
`;

interface Cube {
  group: THREE.Group;
  edges: THREE.LineSegments;
  face: THREE.Mesh; // highlights the face being dragged
  grips: THREE.Group[]; // one arrow per face; userData: { axis, side, material }
  active: THREE.Group | null; // grip being dragged
  hover: THREE.Group | null; // grip under the pointer
}

function makeCube(): Cube {
  const group = new THREE.Group();
  group.visible = false;
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
    new THREE.LineBasicMaterial({ color: CAP_COLOR, transparent: true, opacity: 0.9, depthTest: false }),
  );
  edges.renderOrder = 2;
  const face = new THREE.Mesh(
    new THREE.BoxGeometry(1, 1, 1), // a thin slab scaled to the face, so no rotation is needed
    new THREE.MeshBasicMaterial({
      color: CAP_COLOR,
      transparent: true,
      opacity: 0.16,
      side: THREE.DoubleSide,
      depthWrite: false,
      depthTest: false,
    }),
  );
  face.renderOrder = 2;
  face.visible = false;
  group.add(edges, face);

  // A double-headed arrow along +Y (turned onto each face's axis below): the face can move either way.
  const arrow = mergeGeometries([
    new THREE.CylinderGeometry(0.11, 0.11, 1.8, 16),
    new THREE.ConeGeometry(0.3, 0.6, 20).translate(0, 1.2, 0),
    new THREE.ConeGeometry(0.3, 0.6, 20).rotateX(Math.PI).translate(0, -1.2, 0),
  ])!;
  const hitbox = new THREE.CylinderGeometry(0.6, 0.6, 3, 8); // generous invisible target, so thin arrows are easy to grab
  const grips: THREE.Group[] = [];
  for (const axis of AXES) {
    for (const side of ['min', 'max'] as Side[]) {
      const material = new THREE.ShaderMaterial({
        vertexShader: GRIP_VERTEX,
        fragmentShader: GRIP_FRAGMENT,
        uniforms: { uColor: { value: new THREE.Color(AXIS_COLOR[axis]) }, uGlow: { value: 0 } },
        depthTest: false,
      });
      const mesh = new THREE.Mesh(arrow, material);
      mesh.renderOrder = 3; // always on top, so it can be grabbed through the model
      const hit = new THREE.Mesh(hitbox, new THREE.MeshBasicMaterial({ visible: false }));
      const grip = new THREE.Group();
      grip.add(mesh, hit);
      grip.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), AXIS_VECTOR[axis]);
      grip.userData = { axis, side, material };
      grips.push(grip);
      group.add(grip);
    }
  }
  return { group, edges, face, grips, active: null, hover: null };
}

/** Moves the cube's wireframe, grips and highlight to match the box. */
function updateCube(cube: Cube, box: Box) {
  const centre = new THREE.Vector3();
  const size = new THREE.Vector3();
  for (const a of AXES) {
    centre[a] = (box.min[a] + box.max[a]) / 2;
    size[a] = box.max[a] - box.min[a];
  }
  cube.edges.position.copy(centre);
  cube.edges.scale.copy(size);
  for (const grip of cube.grips) {
    const { axis, side } = grip.userData as { axis: Axis; side: Side };
    grip.position.copy(centre);
    grip.position[axis] = box[side][axis];
  }
  const active = cube.active;
  cube.face.visible = !!active;
  if (active) {
    const { axis } = active.userData as { axis: Axis };
    cube.face.position.copy(active.position);
    cube.face.scale.copy(size);
    cube.face.scale[axis] = Math.max(size.x, size.y, size.z) * 0.002;
  }
}

// The bundled typings omit `target`, which the controls do have at runtime.
type Arcball = ArcballControls & { target: THREE.Vector3; minDistance: number; maxDistance: number };

let labelText: { geometry: THREE.BufferGeometry; width: number; height: number } | null = null;

/**
 * The Nmatik Labs pose-axes mark as extruded shapes, laid out on the brand's 72-unit grid (guidelines, section 02):
 * three arrows from one origin, Z up, X and Y 30 degrees below horizontal, with filled dot, heads and labels.
 * Returned centred on the origin and scaled so it is `height` tall.
 */
function makeMark(font: Font, height: number, depth: number): THREE.BufferGeometry {
  const f = height / 72;
  const extrude = (shape: THREE.Shape) => new THREE.ExtrudeGeometry(shape, { depth: depth / f, bevelEnabled: false });
  const parts: THREE.BufferGeometry[] = [];
  const origin = new THREE.Vector2(36, -40); // the grid is y-down; flip so Z points up
  const arrows = [
    { tip: new THREE.Vector2(36, -6), label: 'Z', at: new THREE.Vector2(36, 0.5) },
    { tip: new THREE.Vector2(8.3, -56), label: 'X', at: new THREE.Vector2(3.5, -63.5) },
    { tip: new THREE.Vector2(63.7, -56), label: 'Y', at: new THREE.Vector2(68.5, -63.5) },
  ];
  for (const { tip, label, at } of arrows) {
    const dir = tip.clone().sub(origin).normalize();
    const side = new THREE.Vector2(-dir.y, dir.x);
    const base = tip.clone().addScaledVector(dir, -8); // heads are 8 long and 8 wide
    // The shaft runs 2 units into the head, so the two overlap and fuse as one solid when printed.
    const shaftEnd = base.clone().addScaledVector(dir, 2);
    const shaft = new THREE.Shape([
      origin.clone().addScaledVector(side, 0.85),
      shaftEnd.clone().addScaledVector(side, 0.85),
      shaftEnd.clone().addScaledVector(side, -0.85),
      origin.clone().addScaledVector(side, -0.85),
    ]);
    const head = new THREE.Shape([tip, base.clone().addScaledVector(side, 4), base.clone().addScaledVector(side, -4)]);
    parts.push(extrude(shaft), extrude(head));

    const letter = new TextGeometry(label, { font, size: 7, depth: depth / f, curveSegments: 8, bevelEnabled: false });
    letter.computeBoundingBox();
    const lb = letter.boundingBox!;
    parts.push(letter.translate(at.x - (lb.min.x + lb.max.x) / 2, at.y - (lb.min.y + lb.max.y) / 2, 0));
  }
  parts.push(extrude(new THREE.Shape().absarc(origin.x, origin.y, 2.5, 0, Math.PI * 2, false)));

  const mark = mergeGeometries(parts)!;
  parts.forEach((g) => g.dispose());
  mark.computeBoundingBox();
  const b = mark.boundingBox!;
  mark.translate(-(b.min.x + b.max.x) / 2, -(b.min.y + b.max.y) / 2, 0).scale(f, f, f);
  return mark;
}

/**
 * The wordmark as extruded letters, one unit of type size, centred on the origin and built once:
 * NMATIK in Barlow SemiBold and LABS in Barlow Light, tracked wide as in the brand guidelines,
 * under the pose-axes mark (the guidelines' stacked lockup).
 */
function getLabelText() {
  if (labelText) return labelText;
  type FontData = ConstructorParameters<typeof Font>[0];
  const bold = new Font(barlowSemiBold as unknown as FontData);
  const light = new Font(barlowLight as unknown as FontData);
  const options = { size: 1, depth: LABEL_DEPTH, curveSegments: 8, bevelEnabled: false };

  // Barlow's cap height is 0.7 em, which gives the em to track against.
  const probe = new TextGeometry('N', { font: bold, ...options });
  probe.computeBoundingBox();
  const em = (probe.boundingBox!.max.y - probe.boundingBox!.min.y) / 0.7;
  probe.dispose();
  const tracking = 0.38 * em;

  const letters: THREE.BufferGeometry[] = [];
  let pen = 0;
  [...LABEL_TEXT].forEach((ch, i) => {
    const font = i < 6 ? bold : light;
    const data = font.data as unknown as { resolution: number; glyphs: Record<string, { ha: number }> };
    letters.push(new TextGeometry(ch, { font, ...options }).translate(pen, 0, 0));
    pen += (data.glyphs[ch].ha * options.size) / data.resolution + tracking;
  });
  const wordmark = mergeGeometries(letters)!;
  letters.forEach((g) => g.dispose());
  wordmark.computeBoundingBox();
  const wb = wordmark.boundingBox!;
  wordmark.translate(-(wb.min.x + wb.max.x) / 2, -(wb.min.y + wb.max.y) / 2, 0);
  const wordHeight = wb.max.y - wb.min.y;

  // The mark sits centred above the wordmark, a little over a third of its width tall.
  const markHeight = 0.38 * (wb.max.x - wb.min.x);
  const mark = makeMark(bold, markHeight, options.depth);
  mark.translate(0, wordHeight / 2 + 0.7 + markHeight / 2, 0);

  const geometry = mergeGeometries([wordmark, mark])!;
  wordmark.dispose();
  mark.dispose();
  geometry.computeBoundingBox();
  const b = geometry.boundingBox!;
  geometry.translate(-(b.min.x + b.max.x) / 2, -(b.min.y + b.max.y) / 2, 0);
  labelText = { geometry, width: b.max.x - b.min.x, height: b.max.y - b.min.y };
  return labelText;
}

/** Where the label goes: on the largest cap only, small, near its bottom-right corner, reading correctly from outside the part. */
function labelMatrices(result: FromWorker): THREE.Matrix4[] {
  const text = getLabelText();
  const found: { matrix: THREE.Matrix4; area: number }[] = [];
  result.planes.forEach((plane, i) => {
    const b = result.capBounds.subarray(i * 6, i * 6 + 6);
    if (b[0] > b[3] || b[1] > b[4] || b[2] > b[5]) return; // no cap on this plane

    const frame = labelFrame(plane);
    const normal = new THREE.Vector3(...frame.normal);
    const up = new THREE.Vector3(...frame.up);
    const right = new THREE.Vector3(...frame.right);
    const axisOf = (v: THREE.Vector3) => [Math.abs(v.x), Math.abs(v.y), Math.abs(v.z)].findIndex((c) => c > 0.5);
    const width = b[3 + axisOf(right)] - b[axisOf(right)];
    const height = b[3 + axisOf(up)] - b[axisOf(up)];
    if (width <= 0 || height <= 0) return;

    // Just over a quarter of the cap's width, and never taller than a fifth of its height (the lockup is stacked).
    const scale = Math.min((0.28 * width) / text.width, (0.2 * height) / text.height);
    // Raised lettering thick enough to print: about 1% of the cap, sunk 30% into it so it fuses with the part.
    const thickness = Math.max(LABEL_DEPTH * scale, 0.012 * Math.min(width, height));
    const margin = 0.03 * Math.min(width, height);
    const anchor = new THREE.Vector3().fromArray(result.capAnchors, i * 3);
    // The anchor is the cap point nearest the bottom-right corner: put the text's own corner just inside it.
    const position = anchor
      .addScaledVector(right, -(text.width * scale) / 2 - margin)
      .addScaledVector(up, (text.height * scale) / 2 + margin)
      .addScaledVector(normal, -0.3 * thickness);

    if (found.length > 0 && found[0].area >= width * height) return;
    found[0] = {
      area: width * height,
      matrix: new THREE.Matrix4()
        .makeBasis(
          right.multiplyScalar(scale),
          up.clone().multiplyScalar(scale),
          normal.clone().multiplyScalar(thickness / LABEL_DEPTH),
        )
        .setPosition(position),
    };
  });
  return found.map((f) => f.matrix);
}

let labelSolid: IndexedMesh | null = null;

/** The label as a welded triangle mesh, so each letter and the mark are closed shells in the export. */
function getLabelSolid(): IndexedMesh {
  if (!labelSolid) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', getLabelText().geometry.getAttribute('position'));
    const welded = mergeVertices(g, 1e-4);
    labelSolid = {
      positions: Float32Array.from(welded.getAttribute('position').array),
      indices: Uint32Array.from(welded.getIndex()!.array),
    };
  }
  return labelSolid;
}

/** Appends a copy of the label, placed by each matrix, to the mesh. */
function withLabels(mesh: IndexedMesh, matrices: THREE.Matrix4[]): IndexedMesh {
  if (matrices.length === 0) return mesh;
  const solid = getLabelSolid();
  const positions = new Float32Array(mesh.positions.length + matrices.length * solid.positions.length);
  const indices = new Uint32Array(mesh.indices.length + matrices.length * solid.indices.length);
  positions.set(mesh.positions);
  indices.set(mesh.indices);
  let p = mesh.positions.length;
  let t = mesh.indices.length;
  let base = mesh.positions.length / 3;
  const v = new THREE.Vector3();
  for (const m of matrices) {
    for (let i = 0; i < solid.positions.length; i += 3) {
      v.set(solid.positions[i], solid.positions[i + 1], solid.positions[i + 2]).applyMatrix4(m);
      positions[p++] = v.x;
      positions[p++] = v.y;
      positions[p++] = v.z;
    }
    for (const index of solid.indices) indices[t++] = index + base;
    base += solid.positions.length / 3;
  }
  return { positions, indices };
}

/** Replaces the group's labels with one label mesh per matrix. */
function fillLabels(labels: THREE.Group, matrices: THREE.Matrix4[]) {
  labels.clear();
  const { geometry } = getLabelText();
  const material = labels.userData.material as THREE.Material;
  for (const matrix of matrices) {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.matrixAutoUpdate = false;
    mesh.matrix.copy(matrix);
    mesh.matrixWorldNeedsUpdate = true;
    labels.add(mesh);
  }
}


/**
 * Paints the MRI onto the cut faces: each fragment's world position is mapped through the inverse affine into the
 * volume texture (R: intensity, G: ventricle). Ventricles get a blue cross-hatch instead of the grey scan.
 */
function makeSliceUniforms() {
  return {
    uVolume: { value: null as THREE.Data3DTexture | null },
    uWorldToVoxel: { value: new THREE.Matrix4() },
    uDims: { value: new THREE.Vector3(1, 1, 1) },
    uHasVolume: { value: 0 },
  };
}
type SliceUniforms = ReturnType<typeof makeSliceUniforms>;

function addSliceOverlay(material: THREE.MeshStandardMaterial, uniforms: SliceUniforms, walls = false) {
  // Both materials patch the same stock shader with the same function text; three would share one program between them.
  material.customProgramCacheKey = () => (walls ? 'slice-walls' : 'slice-cap');
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vSlicePos;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvSlicePos = position;');
    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
        varying vec3 vSlicePos;
        uniform highp sampler3D uVolume;
        uniform mat4 uWorldToVoxel;
        uniform vec3 uDims;
        uniform float uHasVolume;`,
      )
      .replace(
        '#include <opaque_fragment>',
        walls
          ? `if (uHasVolume > 0.5) {
          // The hollowed ventricle's inner walls lie on the edge of the ventricle mask: tint them blue and hatch them.
          vec3 uvw = (uWorldToVoxel * vec4(vSlicePos, 1.0)).xyz;
          uvw = (uvw + 0.5) / uDims;
          if (all(greaterThanEqual(uvw, vec3(0.0))) && all(lessThanEqual(uvw, vec3(1.0)))) {
            float inside = smoothstep(0.1, 0.35, texture(uVolume, uvw).g);
            float a = step(0.55, fract((vSlicePos.x + vSlicePos.y + vSlicePos.z) / 1.6));
            float b = step(0.55, fract((vSlicePos.x - vSlicePos.y + vSlicePos.z) / 1.6));
            vec3 hatch = pow(mix(vec3(0.02, 0.18, 0.38), vec3(0.25, 0.85, 1.0), max(a, b)), vec3(2.2));
            float light = clamp(dot(outgoingLight, vec3(0.333)), 0.35, 1.0);
            outgoingLight = mix(outgoingLight, hatch * light, inside);
          }
        }
        #include <opaque_fragment>`
          : `if (uHasVolume > 0.5) {
          vec3 uvw = (uWorldToVoxel * vec4(vSlicePos, 1.0)).xyz;
          uvw = (uvw + 0.5) / uDims;
          if (all(greaterThanEqual(uvw, vec3(0.0))) && all(lessThanEqual(uvw, vec3(1.0)))) {
            vec2 texel = texture(uVolume, uvw).rg;
            vec3 tissue = vec3(pow(texel.r, 0.85 * 2.2)); // the output stage re-encodes sRGB
            float inside = smoothstep(0.35, 0.65, texel.g);
            // Diagonal hatching in world space (1.6 mm period), the same on every cut plane.
            float a = step(0.55, fract((vSlicePos.x + vSlicePos.y + vSlicePos.z) / 1.6));
            float b = step(0.55, fract((vSlicePos.x - vSlicePos.y + vSlicePos.z) / 1.6));
            vec3 hatch = mix(vec3(0.02, 0.18, 0.38), vec3(0.25, 0.85, 1.0), max(a, b));
            outgoingLight = mix(tissue, pow(hatch, vec3(2.2)), inside);
          }
        }
        #include <opaque_fragment>`,
      );
  };
}

interface Viewer {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: Arcball; // free rotation: no pole lock, can roll
  body: THREE.Mesh;
  cap: THREE.Mesh;
  ghost: THREE.Mesh;
  parts: THREE.Group; // the restored trimmed pieces
  outline: THREE.LineSegments;
  cube: Cube;
  preview: THREE.Mesh; // the uncut mesh, clipped by the cube on the GPU while the exact cut is computed
  labels: THREE.Group; // NMATIKLABS on the caps of the live cut
  baked: THREE.Group; // NMATIKLABS on caps from cuts that were applied
  slice: SliceUniforms; // the MRI shown on the cut faces
  clip: THREE.Plane[]; // the cube's six faces, in the order min x, max x, min y, max y, min z, max z
  axes: ReturnType<typeof makeAxesGizmo> & { visible: boolean };
}

export default function MeshCutter() {
  const mountRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<Viewer | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [fileName, setFileName] = useState<string | null>(null);
  const [original, setOriginal] = useState<IndexedMesh | null>(null);
  const [base, setBase] = useState<IndexedMesh | null>(null);
  const [baseOpenEdges, setBaseOpenEdges] = useState(0);
  const [slicing, setSlicing] = useState(false);
  const [parts, setParts] = useState<(TrimmedPart & { cut: number })[]>([]); // what each slice cut away, as closed pieces
  const [ventricles, setVentricles] = useState<IndexedMesh | null>(null); // the ventricles as a solid, cut along with the mesh
  const [ventriclesOriginal, setVentriclesOriginal] = useState<IndexedMesh | null>(null);
  const [shell, setShell] = useState<IndexedMesh | null>(null); // the skin plus its 5 mm inner surface, cut along with the mesh
  const [shellOriginal, setShellOriginal] = useState<IndexedMesh | null>(null);
  // What the restored pieces show: nothing, the whole pieces, only their ventricles, or only their surface shell.
  const [view, setView] = useState<'off' | 'whole' | 'ventricles' | 'shell'>('off');
  const restored = view !== 'off'; // the pieces are back, drawn like the mesh and exported on their own
  const [showTrimmed, setShowTrimmed] = useState(false); // the removed part drawn faintly
  const [trimmedAlpha, setTrimmedAlpha] = useState(0.3);
  const [sliceVolume, setSliceVolume] = useState<SliceVolume | null>(null);
  const [niftiInfo, setNiftiInfo] = useState<NiftiInfo | null>(null);
  const [loadingStage, setLoadingStage] = useState<string | null>(null);
  const [result, setResult] = useState<FromWorker | null>(null);
  const [pending, setPending] = useState(true); // the shown result is older than the cube
  const [cutsApplied, setCutsApplied] = useState(0);

  const [box, setBox] = useState<Box | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const [bakedLabels, setBakedLabels] = useState<THREE.Matrix4[]>([]); // the label kept on an applied cut (at most one)
  const bakedLabelsRef = useRef(bakedLabels);
  bakedLabelsRef.current = bakedLabels;
  // The mesh carries a single label: a live cut gets one only while no applied cut already holds it.
  const liveLabels = (r: FromWorker) => (bakedLabelsRef.current.length > 0 ? [] : labelMatrices(r));
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  // ---------- Three.js setup ----------
  useEffect(() => {
    const mount = mountRef.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
    camera.position.set(2, 1.5, 2.5);
    scene.add(camera);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.8);
    key.position.set(1, 1.5, 2);
    camera.add(key); // light follows the view

    const controls = new ArcballControls(camera, renderer.domElement) as Arcball;
    controls.cursorZoom = true; // scroll zooms toward the pointer
    controls.setCamera(camera);

    const body = new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.MeshStandardMaterial({ color: BODY_COLOR, roughness: 0.75, metalness: 0, side: THREE.DoubleSide }),
    );
    const cap = new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.MeshStandardMaterial({ color: CUT_COLOR, roughness: 0.55, metalness: 0, side: THREE.DoubleSide }),
    );
    const slice = makeSliceUniforms();
    addSliceOverlay(cap.material as THREE.MeshStandardMaterial, slice);
    addSliceOverlay(body.material as THREE.MeshStandardMaterial, slice, true);
    const ghost = new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.MeshBasicMaterial({
        color: 0x8794a1,
        transparent: true,
        opacity: 0.12,
        depthWrite: false,
        // Pushed slightly back so it never tints the solid part it overlaps.
        polygonOffset: true,
        polygonOffsetFactor: 2,
        polygonOffsetUnits: 2,
      }),
    );
    const outline = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: OUTLINE_COLOR }),
    );
    ghost.renderOrder = 1;
    ghost.visible = false; // the removed part stays hidden until restored
    const cube = makeCube();
    // Start by keeping everything; the cube effect sets the real faces.
    const clip = Array.from({ length: 6 }, () => new THREE.Plane(new THREE.Vector3(1, 0, 0), 1e12));
    const preview = new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.MeshStandardMaterial({
        color: BODY_COLOR,
        roughness: 0.75,
        metalness: 0,
        side: THREE.DoubleSide,
        clippingPlanes: clip,
      }),
    );
    renderer.localClippingEnabled = true;
    const labelMaterial = new THREE.MeshStandardMaterial({ color: LABEL_COLOR, roughness: 0.5, metalness: 0 });
    const labels = new THREE.Group();
    const baked = new THREE.Group();
    labels.userData.material = baked.userData.material = labelMaterial;
    const parts = new THREE.Group();
    parts.userData.material = new THREE.MeshStandardMaterial({ color: BODY_COLOR, roughness: 0.75, metalness: 0, side: THREE.DoubleSide });
    parts.visible = false;
    scene.add(body, cap, outline, preview, ghost, labels, baked, parts, cube.group);

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = mount;
      renderer.setSize(w, h, false);
      camera.aspect = w / Math.max(h, 1);
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(mount);
    resize();

    const axes = { ...makeAxesGizmo(), visible: true };
    const gizmoPos = new THREE.Vector3();
    const toCamera = new THREE.Vector3();
    renderer.setAnimationLoop(() => {
      // Grips keep their size on screen, and only the ones on faces turned toward the camera are shown.
      for (const grip of cube.grips) {
        const { axis, side } = grip.userData as { axis: Axis; side: Side };
        const outward = side === 'max' ? 1 : -1;
        toCamera.copy(camera.position).sub(grip.position);
        grip.visible = grip === cube.active || toCamera[axis] * outward > 0;
        grip.scale.setScalar(toCamera.length() * GRIP_SCALE);
        (grip.userData.material as THREE.ShaderMaterial).uniforms.uGlow.value = grip === cube.active ? 1 : grip === cube.hover ? 0.5 : 0;
      }
      renderer.render(scene, camera);
      if (!axes.visible) return;
      // The triad looks at the origin from the same direction as the main camera.
      axes.camera.quaternion.copy(camera.quaternion);
      axes.camera.position.copy(gizmoPos.set(0, 0, 6).applyQuaternion(camera.quaternion));
      renderer.setViewport(mount.clientWidth - GIZMO_SIZE - GIZMO_MARGIN, GIZMO_MARGIN, GIZMO_SIZE, GIZMO_SIZE);
      renderer.setScissor(mount.clientWidth - GIZMO_SIZE - GIZMO_MARGIN, GIZMO_MARGIN, GIZMO_SIZE, GIZMO_SIZE);
      renderer.setScissorTest(true);
      renderer.autoClear = false;
      renderer.clearDepth();
      renderer.render(axes.scene, axes.camera);
      renderer.autoClear = true;
      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, mount.clientWidth, mount.clientHeight);
    });

    viewerRef.current = { renderer, scene, camera, controls, body, cap, ghost, parts, outline, cube, preview, labels, baked, clip, slice, axes };

    return () => {
      ro.disconnect();
      renderer.setAnimationLoop(null);
      controls.dispose();
      (labels.userData.material as THREE.Material).dispose();
      (parts.userData.material as THREE.Material).dispose();
      for (const root of [scene, axes.scene]) {
        root.traverse((o) => {
          const m = o as THREE.Mesh;
          m.geometry?.dispose();
          const mat = m.material as (THREE.Material & { map?: THREE.Texture | null }) | THREE.Material[] | undefined;
          if (Array.isArray(mat)) mat.forEach((x) => x.dispose());
          else {
            mat?.map?.dispose();
            mat?.dispose();
          }
        });
      }
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      viewerRef.current = null;
    };
  }, []);

  // ---------- Loading ----------
  const frameCamera = useCallback((mesh: IndexedMesh, { direction, up }: View = DEFAULT_VIEW) => {
    const v = viewerRef.current;
    if (!v) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
    g.computeBoundingSphere();
    const { center, radius } = g.boundingSphere!;
    g.dispose();
    v.controls.target.copy(center);
    v.controls.minDistance = radius * 0.05;
    v.controls.maxDistance = radius * 20;
    v.camera.near = radius / 100;
    v.camera.far = radius * 100;
    v.camera.up.copy(up);
    v.camera.position.copy(center).add(direction.clone().normalize().multiplyScalar(radius * 3));
    v.camera.updateProjectionMatrix();
    v.controls.setCamera(v.camera); // re-read the camera it controls
  }, []);

  const loadFile = useCallback(
    (file: File) => {
      setError(null);
      if (!/\.nii(\.gz)?$/i.test(file.name)) {
        setError(`${file.name} is not a NIfTI file. Choose a .nii or .nii.gz volume.`);
        return;
      }
      setLoadingStage('Reading the NIfTI file');
      const worker = new Worker(new URL('../lib/niftiWorker.ts', import.meta.url), { type: 'module' });
      const finish = () => {
        worker.terminate();
        setLoadingStage(null);
      };
      worker.onmessage = (e: MessageEvent<FromNiftiWorker>) => {
        const msg = e.data;
        if (msg.type === 'progress') {
          setLoadingStage(msg.stage);
          return;
        }
        finish();
        if (msg.type === 'error') {
          setError(`Could not read ${file.name}: ${msg.message}`);
          return;
        }
        const mesh: IndexedMesh = { positions: msg.positions, indices: msg.indices };
        setFileName(file.name.replace(/\.nii(\.gz)?$/i, ''));
        setNiftiInfo(msg.info);
        setSliceVolume(msg.slices);
        setParts([]);
        setView('off');
        setShell(msg.shell);
        setShellOriginal(msg.shell);
        setVentricles(msg.ventricles);
        setVentriclesOriginal(msg.ventricles);
        setOriginal(mesh);
        setBakedLabels([]);
        setBase(mesh);
        setBaseOpenEdges(countOpenEdges(mesh.indices, mesh.positions.length / 3));
        setCutsApplied(0);
        frameCamera(mesh);
      };
      worker.onerror = () => {
        finish();
        setError(`Could not read ${file.name}.`);
      };
      file.arrayBuffer().then(
        (buffer) => worker.postMessage({ buffer }, [buffer]),
        (err) => {
          finish();
          setError(`Could not read ${file.name}: ${(err as Error).message}`);
        },
      );
    },
    [frameCamera],
  );

  // ---------- Cut cube ----------
  const ranges = useMemo<Ranges | null>(() => {
    if (!base) return null;
    return {
      x: projectRange(base, AXIS_VECTOR.x),
      y: projectRange(base, AXIS_VECTOR.y),
      z: projectRange(base, AXIS_VECTOR.z),
    };
  }, [base]);
  const boxRef = useRef<Box | null>(null);
  const rangesRef = useRef<Ranges | null>(null);
  boxRef.current = box;
  rangesRef.current = ranges;

  // A new mesh starts with the cube around all of it.
  useEffect(() => {
    setBox(ranges ? boxFromRanges(ranges) : null);
  }, [ranges]);

  /** Faces the cube has moved in from the model's edge; faces still at the edge cut nothing. */
  const cutPlanes = useMemo<CutPlane[]>(() => {
    if (!box || !ranges) return [];
    const out: CutPlane[] = [];
    for (const axis of AXES) {
      const eps = (ranges[axis].max - ranges[axis].min) * 1e-6;
      if (box.min[axis] > ranges[axis].min + eps) out.push({ axis, keepHigh: true, offset: box.min[axis] });
      if (box.max[axis] < ranges[axis].max - eps) out.push({ axis, keepHigh: false, offset: box.max[axis] });
    }
    return out;
  }, [box, ranges]);

  // The cube follows the box instantly.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v) return;
    v.cube.group.visible = !!box;
    if (box) updateCube(v.cube, box);
  }, [box]);

  // Dragging a grip moves that face of the cube along its axis.
  useEffect(() => {
    const v = viewerRef.current;
    const mount = mountRef.current;
    if (!v || !mount) return;
    const dom = v.renderer.domElement;
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let drag: { grip: THREE.Group; axis: Axis; side: Side; startValue: number; startT: number } | null = null;

    const aim = (e: PointerEvent) => {
      const r = dom.getBoundingClientRect();
      ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ndc, v.camera);
    };
    // Raycasting ignores `visible`, so grips hidden on the far side of the cube are filtered out here.
    const pick = () =>
      v.cube.group.visible
        ? (ray.intersectObjects(v.cube.grips, true).find((h) => h.object.parent?.visible)?.object.parent as
            | THREE.Group
            | undefined)
        : undefined;

    // Where along the face's axis the pointer is: closest point between the pointer ray and the axis line.
    const along = (origin: THREE.Vector3, axis: Axis): number | null => {
      const d = AXIS_VECTOR[axis];
      const r = ray.ray.direction;
      const w = origin.clone().sub(ray.ray.origin);
      const b = d.dot(r);
      const denom = 1 - b * b;
      if (denom < 1e-6) return null; // looking straight down the axis
      return (b * r.dot(w) - d.dot(w)) / denom;
    };

    const onDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      aim(e);
      const grip = pick();
      const box = boxRef.current;
      if (!grip || !box) return;
      const { axis, side } = grip.userData as { axis: Axis; side: Side };
      const t = along(grip.position, axis);
      if (t === null) return;
      e.stopPropagation(); // keep the rotate control from starting
      e.preventDefault();
      drag = { grip, axis, side, startValue: box[side][axis], startT: t };
      v.cube.active = grip;
      v.controls.enabled = false;
      setIsDragging(true);
      dom.style.cursor = 'grabbing';
      if (boxRef.current) updateCube(v.cube, boxRef.current);
    };
    const onMove = (e: PointerEvent) => {
      aim(e);
      if (!drag) {
        v.cube.hover = pick() ?? null;
        dom.style.cursor = v.cube.hover ? 'grab' : '';
        return;
      }
      const ranges = rangesRef.current;
      if (!ranges) return;
      const { axis, side, startValue, startT, grip } = drag;
      const t = along(grip.position, axis);
      if (t === null) return;
      const r = ranges[axis];
      const gap = (r.max - r.min) * 0.02;
      setBox((b) => {
        if (!b) return b;
        const value = startValue + (t - startT);
        const next = side === 'min'
          ? Math.min(Math.max(value, r.min), b.max[axis] - gap)
          : Math.max(Math.min(value, r.max), b.min[axis] + gap);
        return { ...b, [side]: { ...b[side], [axis]: next } };
      });
    };
    const onUp = () => {
      if (!drag) return;
      drag = null;
      v.cube.active = null;
      v.controls.enabled = true;
      setIsDragging(false);
      dom.style.cursor = '';
      if (boxRef.current) updateCube(v.cube, boxRef.current);
    };

    mount.addEventListener('pointerdown', onDown, { capture: true });
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    return () => {
      mount.removeEventListener('pointerdown', onDown, { capture: true });
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
    };
  }, []);

  // ---------- Cut worker ----------
  // The exact cut runs off the main thread. While a cube face is being dragged nothing is sent: the
  // clipped preview shows the cut instantly, and the exact result (with caps) follows on release.
  const workerRef = useRef<Worker | null>(null);
  const jobRef = useRef({ inFlight: false, wantedSeq: 0, sentSeq: 0, version: 0, planes: [] as CutPlane[] });

  const pump = useCallback(() => {
    const w = workerRef.current;
    const j = jobRef.current;
    if (!w || j.inFlight || j.sentSeq === j.wantedSeq) return;
    j.inFlight = true;
    j.sentSeq = j.wantedSeq;
    w.postMessage({ type: 'cut', seq: j.sentSeq, version: j.version, planes: j.planes } satisfies ToWorker);
  }, []);

  useEffect(() => {
    const w = new Worker(new URL('../lib/cutWorker.ts', import.meta.url), { type: 'module' });
    workerRef.current = w;
    w.onmessage = (e: MessageEvent<FromWorker>) => {
      const j = jobRef.current;
      j.inFlight = false;
      const r = e.data;
      if (r.version === j.version && r.seq === j.wantedSeq) {
        setResult(r);
        setPending(false);
      }
      pump(); // a newer request may have arrived meanwhile
    };
    return () => {
      w.terminate();
      workerRef.current = null;
    };
  }, [pump]);

  // Give the worker each new mesh once; cuts then only send the planes.
  useEffect(() => {
    if (!base) return;
    const j = jobRef.current;
    j.version++;
    workerRef.current?.postMessage({
      type: 'mesh',
      version: j.version,
      positions: base.positions,
      indices: base.indices,
    } satisfies ToWorker);
  }, [base]);

  useEffect(() => {
    if (!base) return;
    const j = jobRef.current;
    j.wantedSeq++;
    j.planes = cutPlanes;
    setPending(true);
    if (!isDragging) pump();
  }, [base, cutPlanes, isDragging, pump]);

  // Show the clipped preview while the result is out of date, the exact result otherwise.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v) return;
    const exact = !pending && !!result;
    v.body.visible = v.cap.visible = v.outline.visible = exact;
    v.labels.visible = exact;
    v.baked.visible = true; // part of the mesh now, so shown even while a new cut is pending
    v.preview.visible = !exact;
  }, [pending, result]);

  // The preview's clip planes follow the cube.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v || !box) return;
    AXES.forEach((a, i) => {
      v.clip[2 * i].set(AXIS_VECTOR[a], -box.min[a]); // keeps coordinate >= min
      v.clip[2 * i + 1].set(AXIS_VECTOR[a].clone().negate(), box.max[a]); // keeps coordinate <= max
    });
  }, [box]);

  // Push the cut result into the scene.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v || !result) return;
    const position = new THREE.BufferAttribute(result.positions, 3);

    const bodyGeo = new THREE.BufferGeometry();
    bodyGeo.setAttribute('position', position);
    bodyGeo.setAttribute('normal', new THREE.BufferAttribute(result.bodyNormals, 3));
    bodyGeo.setIndex(new THREE.BufferAttribute(result.bodyIndices, 1));

    const capGeo = new THREE.BufferGeometry();
    capGeo.setAttribute('position', position);
    capGeo.setAttribute('normal', new THREE.BufferAttribute(result.capNormals, 3));
    capGeo.setIndex(new THREE.BufferAttribute(result.capIndices, 1));

    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', position);
    lineGeo.setIndex(new THREE.BufferAttribute(result.sectionSegments, 1));

    v.body.geometry.dispose();
    v.cap.geometry.dispose();
    v.outline.geometry.dispose();
    v.body.geometry = bodyGeo;
    v.cap.geometry = capGeo;
    v.outline.geometry = lineGeo;
    fillLabels(v.labels, liveLabels(result));
  }, [result, bakedLabels]);

  useEffect(() => {
    const v = viewerRef.current;
    if (v) fillLabels(v.baked, bakedLabels);
  }, [bakedLabels]);

  // Faint copy of the uncut mesh so you can see what will be removed.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v || !base) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(base.positions, 3));
    g.setIndex(new THREE.BufferAttribute(base.indices, 1));
    v.preview.geometry.dispose();
    v.preview.geometry = g;
    g.computeVertexNormals(); // for the lit preview
  }, [base]);

  // The trimmed part: the mesh as loaded, drawn faintly behind the solid one so only what was cut away shows.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v || !original) return;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(original.positions, 3));
    g.setIndex(new THREE.BufferAttribute(original.indices, 1));
    v.ghost.geometry.dispose();
    v.ghost.geometry = g;
  }, [original]);

  /** What the current view shows of a trimmed part: the surface shell also keeps the ventricles inside it, filled. */
  const piecesOf = (part: TrimmedPart): { kind: string; piece: Piece }[] => {
    const wanted: [string, Piece | null][] =
      view === 'ventricles' ? [['ventricles', part.ventricles]] : view === 'shell' ? [['shell', part.shell], ['ventricles', part.ventricles]] : [['trimmed', part.hollow]];
    return wanted.flatMap(([kind, piece]) => (piece ? [{ kind, piece }] : []));
  };
  const shownPieces = parts.flatMap((part) => piecesOf(part).map((p) => ({ ...p, part })));

  // Restored pieces: the same material as the body, so they read as part of the mesh.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v) return;
    for (const child of [...v.parts.children]) {
      v.parts.remove(child);
      (child as THREE.Mesh).geometry.dispose();
    }
    for (const { piece } of shownPieces) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(piece.mesh.positions, 3));
      g.setAttribute('normal', new THREE.BufferAttribute(piece.normals, 3));
      g.setIndex(new THREE.BufferAttribute(piece.mesh.indices, 1));
      v.parts.add(new THREE.Mesh(g, v.parts.userData.material as THREE.Material));
    }
  }, [parts, view]);

  useEffect(() => {
    const v = viewerRef.current;
    if (v) v.parts.visible = restored;
  }, [restored]);

  // The slider fades the restored pieces too; they stay the mesh's colour, only see-through.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v) return;
    const material = v.parts.userData.material as THREE.MeshStandardMaterial;
    material.opacity = trimmedAlpha;
    material.transparent = trimmedAlpha < 1;
    material.depthWrite = trimmedAlpha >= 1;
    material.needsUpdate = true;
  }, [trimmedAlpha]);

  useEffect(() => {
    const v = viewerRef.current;
    if (!v) return;
    v.ghost.visible = showTrimmed;
    (v.ghost.material as THREE.MeshBasicMaterial).opacity = trimmedAlpha;
  }, [showTrimmed, trimmedAlpha]);

  // The MRI behind the cut faces.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v || !sliceVolume) return;
    const { texels, dims, affine } = sliceVolume;
    const texture = new THREE.Data3DTexture(texels as Uint8Array<ArrayBuffer>, dims[0], dims[1], dims[2]);
    texture.format = THREE.RGFormat;
    texture.type = THREE.UnsignedByteType;
    texture.minFilter = texture.magFilter = THREE.LinearFilter;
    texture.unpackAlignment = 1;
    texture.needsUpdate = true;
    v.slice.uVolume.value?.dispose();
    v.slice.uVolume.value = texture;
    v.slice.uWorldToVoxel.value.set(...(affine as Parameters<THREE.Matrix4['set']>)).invert();
    v.slice.uDims.value.set(dims[0], dims[1], dims[2]);
    v.slice.uHasVolume.value = 1;
  }, [sliceVolume]);

  // ---------- Actions ----------
  const applyCut = () => {
    if (!result || result.bodyIndices.length === 0 || slicing || !base) return;
    // The cap is part of the mesh from now on, so its label stays. Labels whose cap this cut removed are dropped:
    // those now lie outside the cube (the small margin covers the label's lift above its cap).
    const margin = ranges ? Math.max(...AXES.map((a) => ranges[a].max - ranges[a].min)) * 0.005 : 0;
    const inside = (m: THREE.Matrix4) => {
      const at = new THREE.Vector3().setFromMatrixPosition(m);
      return !box || AXES.every((a) => at[a] >= box.min[a] - margin && at[a] <= box.max[a] + margin);
    };
    const finish = (merged: IndexedMesh, kept: IndexedMesh | null, keptShell: IndexedMesh | null, openEdges: number, trimmed: TrimmedPart[]) => {
      setVentricles(kept);
      setShell(keptShell);
      setParts((kept) => [...kept, ...trimmed.map((t) => ({ ...t, cut: cutsApplied + 1 }))]);
      setBakedLabels((kept) => {
        const stay = kept.filter(inside);
        return stay.length > 0 ? stay : labelMatrices(result);
      });
      setBase(merged);
      setBaseOpenEdges(openEdges);
      setCutsApplied((c) => c + 1);
    };

    // The worker also returns the pieces this slice cuts away, as separate closed parts.
    setSlicing(true);
    setError(null);
    const worker = new Worker(new URL('../lib/sliceWorker.ts', import.meta.url), { type: 'module' });
    const done = () => {
      worker.terminate();
      setSlicing(false);
    };
    worker.onmessage = (e: MessageEvent<FromSliceWorker>) => {
      done();
      if (e.data.type === 'error') {
        setError(`Could not slice the mesh: ${e.data.message}`);
        return;
      }
      finish(e.data.mesh, e.data.ventricles, e.data.shell, e.data.openEdges, e.data.trimmed);
    };
    worker.onerror = () => {
      done();
      setError('Could not slice the mesh.');
    };
    worker.postMessage({ mesh: base, ventricles, shell, planes: cutPlanes });
  };

  const reset = () => {
    if (!original) return;
    setBase(original);
    setParts([]);
    setView('off');
    setShell(shellOriginal);
    setVentricles(ventriclesOriginal);
    setBakedLabels([]);
    setBaseOpenEdges(countOpenEdges(original.indices, original.positions.length / 3));
    setCutsApplied(0);
  };

  const resetCube = () => ranges && setBox(boxFromRanges(ranges));

  const downloadPly = (mesh: IndexedMesh, name: string) => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
    geo.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
    new PLYExporter().parse(
      new THREE.Mesh(geo),
      (data) => {
        const url = URL.createObjectURL(new Blob([data as ArrayBuffer], { type: 'application/octet-stream' }));
        const a = document.createElement('a');
        a.href = url;
        a.download = name;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        geo.dispose();
      },
      { binary: true, excludeAttributes: ['normal', 'color', 'uv'] },
    );
  };

  const exportPly = () => {
    if (!result || !fileName) return;
    // The label goes into every export, as raised geometry on each cut: applied ones plus the live cut's.
    downloadPly(withLabels(mergeResult(result), [...bakedLabels, ...liveLabels(result)]), `${fileName}_cut.ply`);
    // Restored pieces go out as files of their own, so each one prints as its own part.
    if (restored)
      shownPieces.forEach(({ kind, piece, part }, i) =>
        setTimeout(() => downloadPly(piece.mesh, `${fileName}_${kind}_${part.cut}_${part.side}.ply`), 300 * (i + 1)),
      );
  };

  /** The cut mesh and the restored parts in one file: a single mesh made of several shells. */
  const exportFused = () => {
    if (!result || !fileName) return;
    const shells = [
      withLabels(mergeResult(result), [...bakedLabels, ...liveLabels(result)]),
      ...shownPieces.map(({ piece }) => piece.mesh),
    ];
    const positions = new Float32Array(shells.reduce((n, m) => n + m.positions.length, 0));
    const indices = new Uint32Array(shells.reduce((n, m) => n + m.indices.length, 0));
    let p = 0;
    let t = 0;
    for (const m of shells) {
      const offset = p / 3;
      positions.set(m.positions, p);
      for (let i = 0; i < m.indices.length; i++) indices[t + i] = m.indices[i] + offset;
      p += m.positions.length;
      t += m.indices.length;
    }
    downloadPly({ positions, indices }, `${fileName}_fused.ply`);
  };

  // ---------- Derived info ----------
  const triCount = result ? (result.bodyIndices.length + result.capIndices.length) / 3 : 0;
  const empty = !!result && result.bodyIndices.length === 0;
  const status = pending ? 'Updating…' : empty ? 'Empty' : result?.openEdges === 0 ? 'Watertight' : 'Has gaps';

  return (
    <div className="app">
      <aside className="panel">
        <header className="brand">
          <div className="lockup">
            <Mark />
            <span className="wordmark">
              Nmatik <span>Labs</span>
            </span>
          </div>
          <h1>Mesh Editor</h1>
        </header>

        <section className="group">
          <button className="btn btn-primary" onClick={() => fileInputRef.current?.click()} disabled={!!loadingStage}>
            {loadingStage ?? (fileName ? 'Load another .nii' : 'Load .nii volume')}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".nii,.nii.gz,.gz"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) loadFile(f);
              e.target.value = '';
            }}
          />
          {fileName && (
            <p className="meta">
              <strong>{fileName}.nii</strong>
              <br />
              {baseOpenEdges === 0
                ? 'Closed mesh, so the cut will be watertight.'
                : `${baseOpenEdges.toLocaleString()} open edges in the mesh. Close its holes first for a watertight cut.`}
            </p>
          )}
          {error && <p className="error" role="alert">{error}</p>}
        </section>

        <fieldset className="group" disabled={!base}>
          <legend>Cut cube</legend>
          <p className="hint">
            Drag the arrows on the cube to trim the model. Everything outside the cube is removed.
          </p>
          <button className="btn btn-quiet" onClick={resetCube} disabled={cutPlanes.length === 0}>
            Reset cube
          </button>
        </fieldset>

        <fieldset className="group" disabled={!base}>
          <legend>Trimmed part</legend>
          {(
            [
              ['whole', 'Put trimmed parts back', 'Remove restored parts', true],
              ['shell', 'Restore the 5 mm surface + ventricles', 'Remove the surface shell', !!shell],
              ['ventricles', 'Keep only the ventricles', 'Remove the ventricles', !!ventricles],
            ] as const
          ).map(([mode, label, activeLabel, available]) =>
            available ? (
              <button
                key={mode}
                className={mode === 'whole' ? 'btn' : 'btn btn-quiet'}
                onClick={() => {
                  if (!restored) setTrimmedAlpha(1); // back as solid as the mesh; the slider fades it from there
                  setView(view === mode ? 'off' : mode);
                }}
                disabled={parts.length === 0}
                aria-pressed={view === mode}
                title={parts.length === 0 ? 'Slice first: the pieces a slice cuts away can then be put back' : undefined}
              >
                {view === mode ? activeLabel : label}
              </button>
            ) : null,
          )}
          {restored && (
            <p className="hint">
              {shownPieces.length} separate part{shownPieces.length === 1 ? '' : 's'} restored. Export PLY saves each one as its own file.
            </p>
          )}
          <button className="btn btn-quiet" onClick={() => setShowTrimmed((s) => !s)} aria-pressed={showTrimmed}>
            {showTrimmed ? 'Hide faint trimmed part' : 'Show faint trimmed part'}
          </button>
          <label className="slider">
            <span>Alpha</span>
            <input
              type="range"
              min={0.05}
              max={1}
              step={0.05}
              value={trimmedAlpha}
              disabled={!showTrimmed && !restored}
              onChange={(e) => setTrimmedAlpha(Number(e.target.value))}
            />
            <output>{Math.round(trimmedAlpha * 100)}%</output>
          </label>
        </fieldset>

        <section className="group">
          {ranges && (
            <dl className="stats">
              <div>
                <dt>Dimensions</dt>
                <dd>{AXES.map((a) => (ranges[a].max - ranges[a].min).toFixed(2)).join(' × ')}</dd>
              </div>
              <div>
                <dt>NIfTI orientation</dt>
                <dd>
                  {niftiInfo
                    ? niftiInfo.reoriented
                      ? `RAS (converted from ${niftiInfo.sourceCode})`
                      : 'RAS (as stored)'
                    : 'RAS'}
                </dd>
              </div>
              {niftiInfo && (
                <div>
                  <dt>Ventricles</dt>
                  <dd>{niftiInfo.ventricleMl > 0 ? `${niftiInfo.ventricleMl.toFixed(1)} mL` : 'Not found'}</dd>
                </div>
              )}
              {result && (
                <div>
                  <dt>Triangles</dt>
                  <dd>{triCount.toLocaleString()}</dd>
                </div>
              )}
              {result && (
                <div>
                  <dt>Result</dt>
                  <dd className={status === 'Watertight' ? 'ok' : status === 'Has gaps' ? 'warn' : ''}>{status}</dd>
                </div>
              )}
            </dl>
          )}
          {result && result.openLoops > 0 && (
            <p className="warn-text">
              The mesh has holes where a plane crosses it, so part of the cap could not be closed.
            </p>
          )}

          <div className="row">
            <button className="btn" onClick={applyCut} disabled={!result || empty || pending || slicing}>
              {slicing ? 'Slicing…' : 'Slice'}
            </button>
            <button className="btn btn-quiet" onClick={reset} disabled={!original || cutsApplied === 0}>
              Reset
            </button>
          </div>

          <button className="btn btn-primary export" onClick={exportPly} disabled={!result || empty || pending}>
            Export PLY
          </button>
          <button className="btn export" onClick={exportFused} disabled={!result || empty || pending || !restored}
            title={restored ? 'Cut mesh and restored parts as one PLY' : 'Put the trimmed parts back first'}>
            Export fused
          </button>
        </section>
      </aside>

      <main
        className={`viewport${dragging ? ' dragging' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDoubleClick={() => base && frameCamera(base)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f) loadFile(f);
        }}
      >
        <div ref={mountRef} className="canvas" />
        {!base && (
          <div className="empty">
            <p>Drop a .nii or .nii.gz volume here</p>
            <span>or use Load .nii volume on the left</span>
          </div>
        )}
        {base && (
          <div className="view-controls" onDoubleClick={(e) => e.stopPropagation()}>
            {AXES.map((a) => (
              <button
                key={a}
                className="btn"
                onClick={() => frameCamera(base, AXIS_VIEW[a])}
                title={`Look along the ${a.toUpperCase()} axis`}
              >
                {a.toUpperCase()}
              </button>
            ))}
            <button className="btn" onClick={() => frameCamera(base)} title="Return the view to its initial pose (or double-click)">
              Recenter
            </button>
          </div>
        )}
        {base && <p className="orbit-hint">Drag to rotate, right-drag or Shift-drag to pan, scroll to zoom, double-click to recenter</p>}
      </main>
    </div>
  );
}
