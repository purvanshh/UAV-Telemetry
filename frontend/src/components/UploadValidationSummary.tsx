"use client";

import { UploadSummary, UploadSummaryStream } from "@/lib/api";

function FileSize({ bytes }: { bytes: number | null }) {
  if (!bytes) return null;
  if (bytes < 1024) return <span>{bytes} B</span>;
  if (bytes < 1024 * 1024) return <span>{(bytes / 1024).toFixed(1)} KB</span>;
  return <span>{(bytes / 1024 / 1024).toFixed(1)} MB</span>;
}

function StreamCard({ s }: { s: UploadSummaryStream }) {
  const bad = !s.ok || s.errors.length > 0;
  const warn = !bad && (s.warnings.length > 0 || s.missing_required.length > 0 || (s.timestamp?.duplicate_timestamps ?? 0) > 0);
  const borderColor = bad ? "var(--error)" : warn ? "var(--caution)" : "var(--ok)";
  const iconChar = bad ? "✕" : warn ? "!" : "✓";
  const iconColor = bad ? "var(--error)" : warn ? "var(--caution)" : "var(--ok)";

  return (
    <div
      style={{
        border: `1px solid ${borderColor}`,
        borderLeft: `4px solid ${borderColor}`,
        borderRadius: "var(--radius)",
        background: "var(--surface)",
        padding: "0.85rem 1rem",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: "0.5rem", flexWrap: "wrap" }}>
        <div>
          <span
            style={{
              fontWeight: 700,
              fontSize: "0.88rem",
              color: iconColor,
              marginRight: "0.45rem",
            }}
            aria-hidden="true"
          >
            {iconChar}
          </span>
          <strong>{s.stream}</strong>
          {s.filename && (
            <span className="muted small" style={{ marginLeft: "0.4rem" }}>
              {s.filename}
            </span>
          )}
        </div>
        <span className="muted small">
          {s.rows.toLocaleString()} rows · {s.columns} cols
          {s.file_size_bytes != null && (
            <> · <FileSize bytes={s.file_size_bytes} /></>
          )}
        </span>
      </div>

      <p className="muted small" style={{ margin: "0.2rem 0 0.5rem" }}>
        {s.role}
      </p>

      {s.timestamp && (
        <dl
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))",
            gap: "0.2rem 1rem",
            margin: "0 0 0.5rem",
            fontSize: "0.86rem",
          }}
        >
          <div>
            <dt className="muted" style={{ fontSize: "0.76rem", textTransform: "uppercase", letterSpacing: "0.04em" }}>
              Sample rate
            </dt>
            <dd style={{ margin: 0 }}>
              {s.timestamp.est_hz != null ? `~${s.timestamp.est_hz} Hz` : "—"}
            </dd>
          </div>
          <div>
            <dt className="muted" style={{ fontSize: "0.76rem", textTransform: "uppercase", letterSpacing: "0.04em" }}>
              Duration
            </dt>
            <dd style={{ margin: 0 }}>
              {s.timestamp.duration_s != null ? `${s.timestamp.duration_s.toFixed(1)} s` : "—"}
            </dd>
          </div>
          <div>
            <dt className="muted" style={{ fontSize: "0.76rem", textTransform: "uppercase", letterSpacing: "0.04em" }}>
              Timestamp unit
            </dt>
            <dd style={{ margin: 0 }}>
              {s.timestamp.unit ?? "—"}
              {s.timestamp.unit_source && s.timestamp.unit_source !== "schema" && (
                <span className="muted" style={{ marginLeft: "0.3rem" }}>
                  ({s.timestamp.unit_source})
                </span>
              )}
            </dd>
          </div>
          {(s.timestamp.duplicate_timestamps > 0 || s.timestamp.non_monotonic_timestamps > 0) && (
            <div>
              <dt className="muted" style={{ fontSize: "0.76rem", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                Timestamp issues
              </dt>
              <dd style={{ margin: 0, color: "var(--caution)" }}>
                {[
                  s.timestamp.duplicate_timestamps > 0 && `${s.timestamp.duplicate_timestamps} duplicates`,
                  s.timestamp.non_monotonic_timestamps > 0 &&
                    `${s.timestamp.non_monotonic_timestamps} non-monotonic`,
                ]
                  .filter(Boolean)
                  .join(", ")}
              </dd>
            </div>
          )}
        </dl>
      )}

      {s.missing_required.length > 0 && (
        <div className="alert alert-error" style={{ margin: "0.3rem 0", padding: "0.4rem 0.7rem", fontSize: "0.86rem" }}>
          <strong>Missing required columns:</strong> {s.missing_required.join(", ")}
        </div>
      )}

      {s.range_flags.length > 0 && (
        <ul
          style={{
            margin: "0.3rem 0",
            padding: "0 0 0 1.1rem",
            fontSize: "0.86rem",
            color: "var(--caution)",
          }}
        >
          {s.range_flags.map((f) => (
            <li key={f}>{f}</li>
          ))}
        </ul>
      )}

      {s.errors.map((e) => (
        <div
          key={e}
          style={{ fontSize: "0.86rem", color: "var(--error)", marginTop: "0.2rem" }}
        >
          {e}
        </div>
      ))}

      {s.warnings.map((w) => (
        <div
          key={w}
          style={{ fontSize: "0.82rem", color: "var(--caution)", marginTop: "0.15rem" }}
        >
          {w}
        </div>
      ))}
    </div>
  );
}

/** Operator-facing pre-analysis validation summary for Deep Analysis uploads. */
export function UploadValidationSummary({
  summary,
}: {
  summary: UploadSummary;
}) {
  const ready = summary.ready_for_analysis;
  const hasErrors = summary.all_errors.length > 0;

  return (
    <div style={{ marginTop: "1rem" }}>
      <div className="row between" style={{ marginBottom: "0.6rem" }}>
        <h3 style={{ margin: 0, fontSize: "1rem" }}>File validation</h3>
        <span
          className={`chip ${ready ? "s-ok" : "s-error"}`}
        >
          {ready ? "Ready for analysis" : "Not ready — see errors below"}
        </span>
      </div>

      {/* Per-stream cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: "0.75rem",
          marginBottom: "0.75rem",
        }}
      >
        {summary.streams.map((s) => (
          <StreamCard key={s.stream} s={s} />
        ))}
      </div>

      {/* Top-level errors (cross-stream) */}
      {hasErrors && (
        <div className="alert alert-error">
          {summary.all_errors.map((e) => (
            <div key={e}>{e}</div>
          ))}
        </div>
      )}

      {/* Upload warnings */}
      {summary.upload_warnings.length > 0 && (
        <div className="alert alert-warn">
          {summary.upload_warnings.map((w) => (
            <div key={w}>{w}</div>
          ))}
        </div>
      )}

      {/* Hints */}
      {summary.hints.map((h) => (
        <p key={h} className={`small ${ready ? "muted" : ""}`} style={{ margin: "0.4rem 0 0" }}>
          {h}
        </p>
      ))}

      <p className="muted small" style={{ marginTop: "0.4rem" }}>
        {summary.analysis_note}
      </p>
    </div>
  );
}
