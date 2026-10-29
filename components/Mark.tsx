/** Nmatik Labs pose-axes mark: three arrows from one origin, Z up (see brand guidelines, section 02). */
const ORIGIN: [number, number] = [36, 40];
const ARROWS = [
  { label: 'Z', tip: [36, 6], dir: [0, -1], text: [36, 2.5] },
  { label: 'X', tip: [8.3, 56], dir: [-Math.cos(Math.PI / 6), 0.5], text: [3.5, 64] },
  { label: 'Y', tip: [63.7, 56], dir: [Math.cos(Math.PI / 6), 0.5], text: [68.5, 64] },
] as const;

export default function Mark({ size = 44 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="-4 -6 80 80" role="img" aria-label="Nmatik Labs" fill="currentColor">
      {ARROWS.map(({ label, tip, dir, text }) => {
        const [tx, ty] = tip;
        const [dx, dy] = dir;
        const bx = tx - dx * 8;
        const by = ty - dy * 8;
        return (
          <g key={label}>
            <line x1={ORIGIN[0]} y1={ORIGIN[1]} x2={bx} y2={by} stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" />
            <polygon points={`${tx},${ty} ${bx - dy * 4},${by + dx * 4} ${bx + dy * 4},${by - dx * 4}`} />
            <text x={text[0]} y={text[1]} fontSize={10} fontWeight={600} textAnchor="middle" stroke="none">
              {label}
            </text>
          </g>
        );
      })}
      <circle cx={ORIGIN[0]} cy={ORIGIN[1]} r={2.5} />
    </svg>
  );
}
