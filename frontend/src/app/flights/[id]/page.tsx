"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { AppNav } from "@/components/AppNav";
import { FlightTabs } from "@/components/FlightTabs";
import { VerdictBanner } from "@/components/VerdictBanner";
import { api, ApiError, FlightDetail, Overview, setLastFlightId } from "@/lib/api";
import { clock, duration, percent } from "@/lib/format";

const SOURCE_LABEL: Record<string, string> = {
  individual: "Three separate files",
  zip: "ZIP archive",
  combined: "One combined CSV",
};

export default function FlightOverviewPage() {
  const flightId = String(useParams().id || "");
  const [flight, setFlight] = useState<FlightDetail | null>(null);
  const [ov, setOv] = useState<Overview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLastFlightId(flightId);
    const [f, o] = await Promise.all([api.getFlight(flightId), api.overview(flightId)]);
    setFlight(f);
    setOv(o);
  }, [flightId]);

  useEffect(() => {
    load().catch((e) => setError(e instanceof Error ? e.message : "Failed to load flight"));
  }, [load]);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      await api.analyze(flightId);
      await load();
    } catch (e) {
      setError(e instanceof ApiError || e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  }

  const analysed = ov && ov.verdict.state !== "not_analyzed";
  const cov = ov?.coverage;
  const files = ov ? Object.values(ov.filenames) : [];
  const uniqueFiles = Array.from(new Set(files));

  return (
    <>
      <AppNav />
      <main>
        <div className="page-head">
          <h1>Flight {flightId}</h1>
          <p className="sub">
            {ov?.source ? SOURCE_LABEL[ov.source] ?? ov.source : ""}
            {uniqueFiles.length ? ` · ${uniqueFiles.join(", ")}` : ""}
          </p>
        </div>
        <FlightTabs flightId={flightId} />

        {error && <div className="alert alert-error">{error}</div>}
        {!ov && !error && <p className="muted">Loading…</p>}

        {ov && (
          <>
            <VerdictBanner verdict={ov.verdict} />

            <div className="row between" style={{ marginBottom: "1.25rem" }}>
              <p className="muted" style={{ margin: 0 }}>
                {analysed && ov.analyzed_at
                  ? `Analysed ${new Date(ov.analyzed_at).toLocaleString()}`
                  : "This flight has not been analysed."}
              </p>
              <button className="btn btn-primary" onClick={run} disabled={busy}>
                {busy ? "Analysing…" : analysed ? "Re-run analysis" : "Run analysis"}
              </button>
            </div>

            {analysed && (
              <>
                <dl className="facts">
                  <div className="fact">
                    <dt>Events flagged</dt>
                    <dd>{ov.events.length}</dd>
                  </div>
                  <div className="fact">
                    <dt>Data assessed</dt>
                    <dd>
                      {cov?.known ? percent(cov.analysed_fraction) : "Not recorded"}
                      <small>
                        {cov?.known
                          ? `${duration(ov.duration.analysed_s)} of ${duration(ov.duration.aligned_s)} usable`
                          : "Re-run the analysis to record coverage."}
                      </small>
                    </dd>
                  </div>
                  <div className="fact">
                    <dt>Recording length</dt>
                    <dd>{ov.duration.flight_s != null ? duration(ov.duration.flight_s) : "Not recorded"}</dd>
                  </div>
                  <div className="fact">
                    <dt>Unassessed periods</dt>
                    <dd>{cov?.known ? ov.unassessed.length : "Unknown"}</dd>
                  </div>
                </dl>

                {(ov.warnings.length > 0 || ov.upload_warnings.length > 0) && (
                  <div className="alert alert-warn" role="alert">
                    <strong>Please note</strong>
                    <ul style={{ margin: "0.3rem 0 0", paddingLeft: "1.1rem" }}>
                      {Array.from(new Set([...ov.warnings, ...ov.upload_warnings])).slice(0, 8).map((w) => (
                        <li key={w}>{w}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <section className="section">
                  <h2>Summary</h2>
                  <p>{ov.summary_text}</p>
                  {cov?.missing_signals?.length ? (
                    <p className="muted">Signals missing from this flight: {cov.missing_signals.join(", ")}.</p>
                  ) : null}
                </section>

                <section className="section">
                  <h2>Events in time order</h2>
                  {ov.events.length === 0 ? (
                    <p className="muted">
                      No events were flagged in the assessed part of the flight. That is not a confirmation that
                      the aircraft behaved well or was safe.
                    </p>
                  ) : (
                    <ol className="event-list">
                      {ov.events.map((e) => (
                        <li key={e.event_id}>
                          <Link
                            className="event-row"
                            href={`/flights/${encodeURIComponent(flightId)}/timeline?event=${encodeURIComponent(e.event_id)}`}
                          >
                            <span className="when">
                              {clock(e.start_rel_s)} – {clock(e.end_rel_s)}
                            </span>
                            <span className="what">
                              {e.headline}
                              <small>{e.channels.slice(0, 3).join(" · ") || "No per-signal detail"}</small>
                            </span>
                            <span className="len">{duration(e.duration_s)}</span>
                          </Link>
                        </li>
                      ))}
                    </ol>
                  )}
                </section>

                {ov.unassessed.length > 0 && (
                  <section className="section">
                    <h2>Periods that could not be assessed</h2>
                    <ul className="event-list">
                      {ov.unassessed.slice(0, 12).map((u, i) => (
                        <li key={i}>
                          <div className="event-row gap">
                            <span className="when">
                              {clock(u.start_rel_s)} – {clock(u.end_rel_s)}
                            </span>
                            <span className="what">
                              <span className="gap-mark" />
                              {u.text}
                            </span>
                            <span className="len">{duration(u.duration_s)}</span>
                          </div>
                        </li>
                      ))}
                    </ul>
                    {ov.unassessed.length > 12 && (
                      <p className="muted small">{ov.unassessed.length - 12} more in the report.</p>
                    )}
                  </section>
                )}

                {ov.input && (
                  <details className="appendix">
                    <summary>How this file was read</summary>
                    <div>
                      <p className="small muted">
                        {ov.input.filename} · {ov.input.rows?.toLocaleString()} rows
                        {ov.input.profile_id ? ` · layout profile ${ov.input.profile_id}` : ""} · mapping v
                        {ov.input.mapping_version}
                        {ov.input.llm?.status === "ok" ? " · AI helped name some columns" : ""}
                      </p>
                      <div className="table-wrap">
                        <table className="data">
                          <thead><tr><th>Column</th><th>Meaning</th><th>Unit</th><th>Mapped by</th></tr></thead>
                          <tbody>
                            {ov.input.columns.filter((c) => c.canonical || c.ignored).map((c) => (
                              <tr key={c.source}>
                                <td className="mono">{c.source}</td>
                                <td>{c.ignored ? "not used" : c.canonical}</td>
                                <td>
                                  {c.ignored ? "—" : `${c.unit ?? "?"}${c.converted ? ` → ${c.expected_unit}` : ""}`}
                                </td>
                                <td className="small">{c.method?.replace("_", " ")}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </details>
                )}

                <div className="row">
                  <Link className="btn" href={`/flights/${encodeURIComponent(flightId)}/timeline`}>
                    Open timeline
                  </Link>
                  <Link className="btn" href={`/flights/${encodeURIComponent(flightId)}/reports`}>
                    Read report
                  </Link>
                </div>
              </>
            )}
            {flight && !analysed && flight.warnings.length > 0 && (
              <div className="alert alert-warn">
                {flight.warnings.slice(0, 6).map((w) => (
                  <div key={w}>{w}</div>
                ))}
              </div>
            )}
          </>
        )}
      </main>
    </>
  );
}
