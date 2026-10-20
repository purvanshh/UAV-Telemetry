import type { EventContext } from "@/lib/api";
import { clock, num, polylineSegments } from "@/lib/format";

const W = 600;
const H = 96;

/** One small chart per channel, shared time axis, event span shaded.
 *  A dashed line marks the median of the preceding recording for comparison. */
export function EventChart({
  ctx,
  before,
}: {
  ctx: EventContext;
  before?: Record<string, number | null | undefined>;
}) {
  if (!ctx.time.length || !ctx.channels.length) {
    return <p className="muted">No telemetry is available around this event.</p>;
  }
  const x0 = ctx.time[0];
  const x1 = ctx.time[ctx.time.length - 1];
  const ex0 = ((ctx.start_rel_s - x0) / (x1 - x0 || 1)) * W;
  const ex1 = ((ctx.end_rel_s - x0) / (x1 - x0 || 1)) * W;
  return (
    <div className="chart-stack">
      {ctx.channels.map((c) => {
        const finite = c.values.filter((v): v is number => v != null && Number.isFinite(v));
        const lo = Math.min(...finite);
        const hi = Math.max(...finite);
        const pad = (hi - lo || 1) * 0.08;
        const y0 = lo - pad;
        const y1 = hi + pad;
        const segs = polylineSegments(ctx.time, c.values, x0, x1, y0, y1, W, H);
        const med = before?.[c.signal];
        const medY = med != null ? H - ((med - y0) / (y1 - y0 || 1)) * H : null;
        return (
          <div className="chart-row" key={c.signal}>
            <div className="chart-title">
              <span>{c.label}</span>
              <span className="mono">
                {num(lo)} to {num(hi)} {c.unit ?? ""}
              </span>
            </div>
            <svg
              viewBox={`0 0 ${W} ${H}`}
              preserveAspectRatio="none"
              role="img"
              aria-label={`${c.label} from ${clock(x0)} to ${clock(x1)}; the event runs ${clock(
                ctx.start_rel_s
              )} to ${clock(ctx.end_rel_s)}. Values range ${num(lo)} to ${num(hi)} ${c.unit ?? ""}.`}
            >
              <rect x={ex0} y={0} width={Math.max(ex1 - ex0, 2)} height={H} fill="var(--chart-event)" />
              {medY != null && medY >= 0 && medY <= H && (
                <line x1={0} x2={W} y1={medY} y2={medY} stroke="var(--muted)" strokeDasharray="4 4" strokeWidth={1} />
              )}
              {segs.map((p, i) => (
                <polyline
                  key={i}
                  points={p}
                  fill="none"
                  stroke="var(--chart-line)"
                  strokeWidth={1.6}
                  vectorEffect="non-scaling-stroke"
                />
              ))}
            </svg>
          </div>
        );
      })}
      <div className="chart-axis" aria-hidden="true">
        <span>{clock(x0)}</span>
        <span>
          event {clock(ctx.start_rel_s)}–{clock(ctx.end_rel_s)}
        </span>
        <span>{clock(x1)}</span>
      </div>
      <div className="chart-legend">
        <span>
          <span className="swatch" style={{ background: "var(--chart-event)", border: "1px solid var(--attention)" }} />
          flagged period
        </span>
        <span>
          <span className="swatch" style={{ borderTop: "2px dashed var(--muted)", height: 0, width: "1rem" }} />
          typical value in the preceding recording
        </span>
      </div>
      {ctx.segment_limited && (
        <p className="muted small">
          Context is cut short where the recording has a gap or begins/ends, so there is less to compare
          against on that side.
        </p>
      )}
    </div>
  );
}
