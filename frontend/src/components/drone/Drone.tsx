/**
 * Quadcopter artwork, drawn to look like a real airframe rather than a mascot.
 * Rotors are animated with CSS (see .q-rotor / .q-blade in globals.css); colour comes from `currentColor`.
 */

/** Side view: body, gimbal camera, landing skids, two visible rotors. Nose points right. */
export function QuadSide({ width = 96, className, label }: { width?: number; className?: string; label?: string }) {
  return (
    <svg
      className={`quad quad-side ${className ?? ""}`}
      width={width}
      height={(width * 40) / 96}
      viewBox="0 0 96 40"
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      {/* rotors: blurred disc plus a blade whose apparent length sweeps as it turns */}
      {[14, 82].map((cx, i) => (
        <g key={cx}>
          <ellipse cx={cx} cy={8.5} rx={13} ry={1.9} className="q-disc" />
          <rect
            x={cx - 12.5}
            y={7.7}
            width={25}
            height={1.6}
            rx={0.8}
            className="q-blade-side"
            style={{ animationDelay: `${i * -0.031}s` }}
          />
          <rect x={cx - 1.3} y={9} width={2.6} height={5.5} rx={0.8} className="q-frame" />
          <rect x={cx - 2.6} y={13.4} width={5.2} height={3.4} rx={1.2} className="q-frame" />
        </g>
      ))}
      {/* arms */}
      <path d="M15 15.2 H81" className="q-arm" />
      {/* body and canopy */}
      <rect x={33} y={11.2} width={30} height={10.4} rx={4.6} className="q-frame" />
      <path d="M38 11.6 Q48 6.8 58 11.6 Z" className="q-frame q-soft" />
      <circle cx={60.5} cy={15.6} r={1.1} className="q-led" />
      {/* gimbal and camera */}
      <rect x={51} y={21.2} width={3} height={2.6} className="q-frame" />
      <rect x={48.6} y={23.4} width={8.4} height={5.6} rx={2.2} className="q-frame" />
      <circle cx={55.2} cy={26.2} r={1.5} className="q-lens" />
      {/* landing gear */}
      <path d="M40 21.4 L36.5 31.5 M56 21.4 L59.5 31.5" className="q-leg" />
      <path d="M31.5 31.8 H41.5 M54.5 31.8 H64.5" className="q-leg q-skid" />
    </svg>
  );
}

const ARM = [
  { x: 9, y: 9, dir: 1 },
  { x: 39, y: 9, dir: -1 },
  { x: 39, y: 39, dir: 1 },
  { x: 9, y: 39, dir: -1 },
];

/** Top-down X quad. Nose points up (-y); rotate the element to set the heading. */
export function QuadTop({ size = 24, className }: { size?: number; className?: string }) {
  return (
    <svg className={`quad quad-top ${className ?? ""}`} width={size} height={size} viewBox="0 0 48 48" aria-hidden="true">
      <path d="M9 9 L39 39 M39 9 L9 39" className="q-arm" />
      {ARM.map((m, i) => (
        <g key={i}>
          <circle cx={m.x} cy={m.y} r={8.2} className="q-disc" />
          <g className={`q-rotor ${m.dir < 0 ? "is-ccw" : ""}`} style={{ transformOrigin: `${m.x}px ${m.y}px`, animationDelay: `${i * -0.017}s` }}>
            <rect x={m.x - 7.6} y={m.y - 0.8} width={15.2} height={1.6} rx={0.8} className="q-blade" />
          </g>
          <circle cx={m.x} cy={m.y} r={1.9} className="q-frame" />
        </g>
      ))}
      <rect x={18.5} y={16} width={11} height={16} rx={4.5} className="q-frame" />
      <circle cx={24} cy={18.6} r={1.25} className="q-led" />
    </svg>
  );
}
