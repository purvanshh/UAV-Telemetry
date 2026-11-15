import type { DirectEventContext } from "@/lib/api";
import { polylineSegments } from "@/lib/format";

const W = 600;
const H = 84;

const fmt = (v: number | null | undefined) => {
  if (v == null || !Number.isFinite(v)) return "Unavailable";
  return Number(v.toPrecision(4)).toString();
};

function mmss(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

/** Charts show one line each, so a plural group label reads better in the singular. */
function chartLabel(label: string): string {
  return label
    .replace(/ readings$/i, " reading")
    .replace(/ inputs$/i, " input")
    .replace(/ attitudes$/i, " attitude")
    .replace(/ commands$/i, " command")
    .replace(/ outputs$/i, " output");
}

/**
 * One small chart per channel with its OWN y-axis (channels differ in scale by orders of magnitude, so they are never
 * drawn on a shared axis), a shared horizontal axis in mm:ss from the start of the flight when the time unit is known,
 * and three shaded regions: earlier baseline, the event interval, and the period after it.
 */
export function DirectEventChart({
  ctx,
  maxChannels,
  technical = false,
}: {
  ctx: DirectEventContext;
  maxChannels?: number;
  technical?: boolean;
}) {
  if (!ctx.channels.length) return <p className="muted">No telemetry is available around this event.</p>;
  const axis = ctx.time_axis ?? null;
  const pilotAxis = !technical && axis != null;
  const conv = (v: number) => (pilotAxis && axis ? (v - axis.offset) * axis.scale : v);
  const label = (v: number) => (pilotAxis ? mmss(v) : fmt(v));

  const x0 = conv(ctx.window_start);
  const x1 = conv(ctx.window_end);
  const span = x1 - x0 || 1;
  const px = (x: number) => ((x - x0) / span) * W;
  const pct = (x: number) => `${(((x - x0) / span) * 100).toFixed(2)}%`;
  const { baseline, event, post } = ctx.regions;
  const bx = [px(conv(baseline.start)), px(conv(event.start))];
  const ex = [px(conv(event.start)), Math.max(px(conv(event.end)), px(conv(event.start)) + 2)];
  const ax = [px(conv(event.end)), px(conv(post.end))];
  const channels = maxChannels ? ctx.channels.slice(0, maxChannels) : ctx.channels;
  const labelCounts = new Map<string, number>();
  for (const c of channels) labelCounts.set(c.label, (labelCounts.get(c.label) ?? 0) + 1);
  const labelSeen = new Map<string, number>();

  return (
    <div className="chart-stack dchart">
      <div className="region-strip" aria-hidden="true">
        <span style={{ left: 0, width: pct(conv(event.start)) }} className="rs-base">
          {baseline.available ? "Before" : "Before: unavailable"}
        </span>
        <span
          style={{ left: pct(conv(event.start)), width: `calc(${pct(conv(event.end))} - ${pct(conv(event.start))})`, minWidth: "3.2rem" }}
          className="rs-event"
        >
          During
        </span>
        <span style={{ left: pct(conv(event.end)), right: 0 }} className="rs-post">
          {post.available ? "After" : "After: unavailable"}
        </span>
      </div>
      {channels.map((c) => {
        const finite = c.values.filter((v): v is number => v != null && Number.isFinite(v));
        if (!finite.length) return null;
        const lo = Math.min(...finite);
        const hi = Math.max(...finite);
        const pad = (hi - lo || Math.abs(hi) * 0.05 || 1) * 0.1;
        const y0 = lo - pad;
        const y1 = hi + pad;
        const yy = (v: number) => H - ((v - y0) / (y1 - y0 || 1)) * H;
        const times = pilotAxis ? c.time.map(conv) : c.time;
        const segs = polylineSegments(times, c.values, x0, x1, y0, y1, W, H);
        const lines: { x: [number, number]; v: number | null; k: string }[] = [
          { x: [bx[0], bx[1]] as [number, number], v: c.before.available ? c.before.median : null, k: "b" },
          { x: [ex[0], ex[1]] as [number, number], v: c.during.median, k: "d" },
          { x: [ax[0], ax[1]] as [number, number], v: c.after.available ? c.after.median : null, k: "a" },
        ];
        const base = chartLabel(c.label);
        const total = labelCounts.get(c.label) ?? 1;
        const n = (labelSeen.get(c.label) ?? 0) + 1;
        labelSeen.set(c.label, n);
        const display = total > 1 ? `${base} ${n}` : base;
        return (
          <div className="chart-row" key={c.signal}>
            <div className="chart-title">
              <span>
                <strong className={technical ? "mono" : undefined}>
                  {technical ? c.technical_name ?? c.signal : display}
                </strong>
                {technical && <span className="muted"> · {display}</span>}
              </span>
              <span className="mono">
                {fmt(lo)} to {fmt(hi)}
              </span>
            </div>
            <svg
              viewBox={`0 0 ${W} ${H}`}
              preserveAspectRatio="none"
              role="img"
              aria-label={`${display}, raw values ${fmt(lo)} to ${fmt(hi)}. Median before: ${fmt(
                c.before.available ? c.before.median : null
              )}, during: ${fmt(c.during.median)}, after: ${fmt(c.after.available ? c.after.median : null)}.`}
            >
              {baseline.available && <rect x={bx[0]} y={0} width={Math.max(bx[1] - bx[0], 0)} height={H} fill="var(--chart-base)" />}
              <rect x={ex[0]} y={0} width={ex[1] - ex[0]} height={H} fill="var(--chart-event)" />
              {post.available && <rect x={ax[0]} y={0} width={Math.max(ax[1] - ax[0], 0)} height={H} fill="var(--chart-post)" />}
              {lines.map(
                (l) =>
                  l.v != null &&
                  Number.isFinite(l.v) && (
                    <line key={l.k} x1={l.x[0]} x2={l.x[1]} y1={yy(l.v)} y2={yy(l.v)} stroke="var(--muted)" strokeDasharray={l.k === "d" ? "2 3" : "5 4"} strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
                  )
              )}
              {segs.map((p, i) => (
                <polyline key={i} points={p} fill="none" stroke="var(--chart-line)" strokeWidth={1.6} vectorEffect="non-scaling-stroke" />
              ))}
            </svg>
          </div>
        );
      })}
      <div className="chart-axis" aria-hidden="true">
        <span>{label(x0)}</span>
        <span>
          during {label(conv(event.start))}–{label(conv(event.end))}
        </span>
        <span>{label(x1)}</span>
      </div>
      <p className="muted small">
        {technical
          ? `Horizontal axis: ${ctx.axis}. Each chart has its own vertical scale in raw values. Dashed lines: median before, during and after the event (shown only when available).`
          : pilotAxis
            ? `Each chart has its own vertical scale. The horizontal axis is time from the start of the flight${axis?.verified ? "" : " (approximate; time unit not verified)"}. Dashed lines show the typical value before, during and after the period.`
            : "Each chart has its own vertical scale. Dashed lines show the typical value before, during and after the period. The file has no usable time column, so the axis is raw row or tick values."}
        {ctx.segment_limited
          ? " The recording starts or ends too close to the event for a full before or after comparison."
          : ""}
      </p>
    </div>
  );
}
