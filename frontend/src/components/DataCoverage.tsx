"use client";

type Dq = {
  coverage_status?: string;
  durations?: {
    flight_duration_s?: number | null;
    aligned_duration_s?: number | null;
    analysed_duration_s?: number | null;
    analysed_fraction_of_aligned?: number | null;
  };
  windows?: {
    candidate?: number;
    scored?: number;
    discarded?: number;
    discarded_pct?: number | null;
  };
  streams?: Record<
    string,
    {
      rows_input?: number;
      rows_after_cleaning?: number;
      rows_duplicate_timestamp?: number;
      rows_missing_timestamp?: number;
      gap_count?: number;
      gap_total_s?: number;
      timestamp_unit?: string | null;
    }
  >;
  features?: { missing?: string[]; excluded_from_model?: string[] };
  excluded_interval_count?: number;
  warnings?: string[];
};

const fmt = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined ? "—" : v.toFixed(d);

export function DataCoverage({ dq }: { dq?: Record<string, unknown> | null }) {
  if (!dq) return null;
  const d = dq as Dq;
  const status = d.coverage_status || "unknown";
  const bad = status !== "ok";
  const frac = d.durations?.analysed_fraction_of_aligned;
  return (
    <section className="panel">
      <h2>Data coverage</h2>
      <p className="muted" style={{ marginTop: 0 }}>
        How much of the uploaded flight was actually scored.{" "}
        <span className="chip">{status}</span>
      </p>
      {bad && (
        <div className="alert alert-warn">
          {status === "insufficient_data"
            ? "Not enough usable data to score this flight."
            : status === "low_coverage"
              ? "Only part of this flight was scored — a quiet result does not cover the rest."
              : "Coverage could not be fully determined."}
        </div>
      )}
      {d.windows && (
        <div className="grid-metrics">
          <div className="metric"><div className="label">Flight length</div><div className="value">{fmt(d.durations?.flight_duration_s)} s</div></div>
          <div className="metric"><div className="label">Scored</div><div className="value">{fmt(d.durations?.analysed_duration_s)} s{frac != null ? ` (${(frac * 100).toFixed(0)}%)` : ""}</div></div>
          <div className="metric"><div className="label">Windows scored</div><div className="value">{d.windows.scored ?? 0} / {d.windows.candidate ?? 0}</div></div>
          <div className="metric"><div className="label">Windows discarded</div><div className="value">{d.windows.discarded ?? 0}{d.windows.discarded_pct != null ? ` (${d.windows.discarded_pct.toFixed(1)}%)` : ""}</div></div>
        </div>
      )}
      {d.streams && (
        <div className="table-wrap" style={{ marginTop: "0.85rem" }}>
          <table className="data">
            <thead>
              <tr><th>Stream</th><th>Rows</th><th>Kept</th><th>Dup. time</th><th>Bad time</th><th>Gaps</th><th>Gap time</th><th>Time unit</th></tr>
            </thead>
            <tbody>
              {Object.entries(d.streams).map(([name, s]) => (
                <tr key={name}>
                  <td>{name}</td>
                  <td>{s.rows_input}</td>
                  <td>{s.rows_after_cleaning}</td>
                  <td>{s.rows_duplicate_timestamp}</td>
                  <td>{s.rows_missing_timestamp}</td>
                  <td>{s.gap_count}</td>
                  <td>{fmt(s.gap_total_s)} s</td>
                  <td>{s.timestamp_unit ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {d.features?.missing?.length ? (
        <p className="muted">Missing model features: {d.features.missing.join(", ")}</p>
      ) : null}
      {d.warnings?.length ? (
        <div className="alert alert-warn" style={{ marginTop: "0.85rem" }}>
          {d.warnings.map((w) => (<div key={w}>{w}</div>))}
        </div>
      ) : null}
    </section>
  );
}
