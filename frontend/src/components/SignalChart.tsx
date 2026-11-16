import type { ChartPack } from "@/lib/api";
import { polylineSegments } from "@/lib/format";

const W = 640;
const H = 150;
const PAD_L = 48;
const PAD_B = 22;
const COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"];

function mmss(t: number): string {
  const s = Math.max(0, t);
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(0).padStart(2, "0")}`;
}

function fmt(v: number): string {
  const a = Math.abs(v);
  return a >= 1000 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : Number(v.toPrecision(3)).toString();
}

/** One chart per signal group: lines, the finding window shaded, time as m:ss since log start. */
export function SignalChart({ chart, window: win, highlight }: {
  chart: ChartPack["charts"][number];
  window: [number, number];
  highlight?: [number, number];
}) {
  const ys = chart.series.flatMap((s) => s.y).filter((v) => Number.isFinite(v));
  if (!ys.length) return null;
  let y0 = Math.min(...ys);
  let y1 = Math.max(...ys);
  if (y0 === y1) {
    y0 -= 1;
    y1 += 1;
  }
  const pad = (y1 - y0) * 0.08;
  y0 -= pad;
  y1 += pad;
  const [x0, x1] = win;
  const pw = W - PAD_L;
  const ph = H - PAD_B;
  const xpx = (t: number) => PAD_L + ((t - x0) / (x1 - x0 || 1)) * pw;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => x0 + f * (x1 - x0));
  return (
    <figure className="sig-chart">
      <figcaption>
        <span className="mono">{chart.title}</span>
        {chart.unit ? <span className="muted"> ({chart.unit})</span> : null}
        <span className="sig-legend">
          {chart.series.map((s, i) => (
            <span key={s.label}>
              <i style={{ background: COLORS[i % COLORS.length] }} /> {s.label}
            </span>
          ))}
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${chart.title} over time`}>
        {highlight && (
          <rect
            x={xpx(Math.max(x0, highlight[0]))}
            y={0}
            width={Math.max(2, xpx(Math.min(x1, highlight[1])) - xpx(Math.max(x0, highlight[0])))}
            height={ph}
            className="sig-band"
          />
        )}
        <line x1={PAD_L} y1={ph} x2={W} y2={ph} className="sig-axis" />
        <text x={PAD_L - 6} y={10} className="sig-tick" textAnchor="end">{fmt(y1)}</text>
        <text x={PAD_L - 6} y={ph} className="sig-tick" textAnchor="end">{fmt(y0)}</text>
        {ticks.map((t) => (
          <text key={t} x={xpx(t)} y={H - 6} className="sig-tick" textAnchor="middle">{mmss(t)}</text>
        ))}
        <g transform={`translate(${PAD_L},0)`}>
          {chart.series.map((s, i) =>
            polylineSegments(s.t, s.y, x0, x1, y0, y1, pw, ph).map((pts, k) => (
              <polyline key={`${s.label}-${k}`} points={pts} fill="none" stroke={COLORS[i % COLORS.length]} strokeWidth={1.4} />
            ))
          )}
        </g>
      </svg>
    </figure>
  );
}
