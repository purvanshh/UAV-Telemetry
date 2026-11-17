"use client";

import { useMemo, useRef, useState } from "react";
import type { ChartPack } from "@/lib/api";
import { polylineSegments } from "@/lib/format";

const W = 640;
const H = 170;
const PAD_L = 46;
const PAD_B = 22;
const PAD_T = 6;
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

function nearest(ts: number[], t: number): number {
  let lo = 0;
  let hi = ts.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] < t) lo = mid;
    else hi = mid;
  }
  return Math.abs(ts[lo] - t) <= Math.abs(ts[hi] - t) ? lo : hi;
}

/** One signal group over time: the problem window shaded, a hover readout of every line at that moment. */
export function SignalChart({ chart, window: win, highlight }: {
  chart: ChartPack["charts"][number];
  window: [number, number];
  highlight?: [number, number];
}) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [x0, x1] = win;
  const pw = W - PAD_L;
  const ph = H - PAD_B - PAD_T;

  const [y0, y1] = useMemo(() => {
    const ys = chart.series.flatMap((s) => s.y).filter((v) => Number.isFinite(v));
    if (!ys.length) return [0, 1];
    let lo = Math.min(...ys);
    let hi = Math.max(...ys);
    if (lo === hi) {
      lo -= 1;
      hi += 1;
    }
    const pad = (hi - lo) * 0.08;
    return [lo - pad, hi + pad];
  }, [chart]);

  if (!chart.series.some((s) => s.y.some((v) => Number.isFinite(v)))) return null;
  const xpx = (t: number) => PAD_L + ((t - x0) / (x1 - x0 || 1)) * pw;
  const ypx = (v: number) => PAD_T + ph - ((v - y0) / (y1 - y0 || 1)) * ph;
  const xticks = [0, 0.25, 0.5, 0.75, 1].map((f) => x0 + f * (x1 - x0));
  const yticks = [0, 0.5, 1].map((f) => y0 + f * (y1 - y0));

  function move(e: React.PointerEvent<SVGSVGElement>) {
    const r = ref.current?.getBoundingClientRect();
    if (!r) return;
    const px = ((e.clientX - r.left) / r.width) * W;
    if (px < PAD_L) return setHover(null);
    setHover(x0 + ((px - PAD_L) / pw) * (x1 - x0));
  }

  const readout =
    hover == null
      ? null
      : chart.series.map((s) => {
          const i = nearest(s.t, hover);
          return { label: s.label, v: s.y[i], t: s.t[i] };
        });

  return (
    <figure className="sig-chart">
      <div className="sig-head">
        <span className="sig-title">
          {chart.title}
          {chart.unit ? <span className="muted"> · {chart.unit}</span> : null}
        </span>
        <span className="sig-legend">
          {chart.series.map((s, i) => (
            <span key={`${s.label}-${i}`}>
              <i style={{ background: COLORS[i % COLORS.length] }} />
              {s.label}
            </span>
          ))}
        </span>
      </div>
      <svg
        ref={ref}
        className="sig-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${chart.title} over time`}
        onPointerMove={move}
        onPointerLeave={() => setHover(null)}
      >
        {yticks.map((v) => (
          <g key={v}>
            <line className="grid" x1={PAD_L} x2={W} y1={ypx(v)} y2={ypx(v)} />
            <text className="tick" x={PAD_L - 6} y={ypx(v) + 3} textAnchor="end">
              {fmt(v)}
            </text>
          </g>
        ))}
        {highlight && (
          <rect
            className="band"
            x={xpx(Math.max(x0, highlight[0]))}
            y={PAD_T}
            width={Math.max(2, xpx(Math.min(x1, highlight[1])) - xpx(Math.max(x0, highlight[0])))}
            height={ph}
            rx={3}
          />
        )}
        {xticks.map((t) => (
          <text key={t} className="tick" x={xpx(t)} y={H - 6} textAnchor="middle">
            {mmss(t)}
          </text>
        ))}
        <g transform={`translate(${PAD_L},${PAD_T})`}>
          {chart.series.map((s, i) =>
            polylineSegments(s.t, s.y, x0, x1, y0, y1, pw, ph).map((pts, k) => (
              <polyline
                key={`${s.label}-${i}-${k}`}
                className="line"
                points={pts}
                pathLength={1}
                style={{ stroke: COLORS[i % COLORS.length], strokeDasharray: 1, ["--len" as string]: 1 }}
              />
            ))
          )}
        </g>
        {hover != null && (
          <g pointerEvents="none">
            <line className="cursor" x1={xpx(hover)} x2={xpx(hover)} y1={PAD_T} y2={PAD_T + ph} />
            {readout?.map((r, i) =>
              Number.isFinite(r.v) ? <circle key={`${r.label}-${i}`} cx={xpx(r.t)} cy={ypx(r.v)} r={3.5} fill={COLORS[i % COLORS.length]} stroke="var(--surface)" strokeWidth={1.5} /> : null
            )}
          </g>
        )}
      </svg>
      <div className="sig-tip" aria-live="off">
        {readout ? (
          <>
            {mmss(hover!)}{" "}
            {readout.map((r, i) => (
              <span key={`${r.label}-${i}`}>
                · {r.label} {Number.isFinite(r.v) ? fmt(r.v) : "—"}{" "}
              </span>
            ))}
          </>
        ) : (
          <span className="muted">Hover for values · shaded: when it happened</span>
        )}
      </div>
    </figure>
  );
}
