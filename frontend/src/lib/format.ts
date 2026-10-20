import type { VerdictState } from "./api";

/** mm:ss (or h:mm:ss) for seconds into the recording. */
export function clock(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec)) return "—";
  const s = Math.max(0, Math.round(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  const mm = String(m).padStart(2, "0");
  const ss = String(r).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

export function duration(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec)) return "—";
  if (sec < 90) return `${sec.toFixed(sec < 10 ? 1 : 0)} s`;
  const m = Math.floor(sec / 60);
  return `${m} min ${Math.round(sec - m * 60)} s`;
}

export function percent(x: number | null | undefined): string {
  return x == null || !Number.isFinite(x) ? "—" : `${Math.round(x * 100)}%`;
}

export function num(x: number | null | undefined, digits = 3): string {
  if (x == null || !Number.isFinite(x)) return "—";
  return Number(x.toPrecision(digits)).toString();
}

/** Short label used in lists. Never says "normal" or "safe". */
export function verdictLabel(state: VerdictState | null | undefined): string {
  switch (state) {
    case "anomalies_detected":
      return "Unusual telemetry detected";
    case "no_alerts_adequate_coverage":
      return "No alerts detected";
    case "no_alerts_low_coverage":
      return "No alerts, limited coverage";
    case "analysis_failed":
      return "Analysis failed";
    case "insufficient_data":
      return "Not enough data";
    default:
      return "Not analysed";
  }
}

export function verdictTone(state: VerdictState | null | undefined): "attention" | "caution" | "error" | "neutral" {
  switch (state) {
    case "anomalies_detected":
      return "attention";
    case "no_alerts_low_coverage":
    case "insufficient_data":
      return "caution";
    case "analysis_failed":
      return "error";
    default:
      return "neutral";
  }
}

/** Map a series to SVG polyline segments, breaking the line at missing values. */
export function polylineSegments(
  xs: number[],
  ys: (number | null)[],
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  w: number,
  h: number
): string[] {
  const segs: string[] = [];
  let cur: string[] = [];
  const dx = x1 - x0 || 1;
  const dy = y1 - y0 || 1;
  for (let i = 0; i < xs.length; i++) {
    const y = ys[i];
    if (y == null || !Number.isFinite(y)) {
      if (cur.length) segs.push(cur.join(" "));
      cur = [];
      continue;
    }
    const px = ((xs[i] - x0) / dx) * w;
    const py = h - ((y - y0) / dy) * h;
    cur.push(`${px.toFixed(1)},${py.toFixed(1)}`);
  }
  if (cur.length) segs.push(cur.join(" "));
  return segs;
}
