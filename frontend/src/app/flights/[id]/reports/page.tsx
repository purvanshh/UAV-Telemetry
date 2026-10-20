"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { AppNav } from "@/components/AppNav";
import { EventDetailPanel } from "@/components/EventDetailPanel";
import { FlightTabs } from "@/components/FlightTabs";
import { VerdictBanner } from "@/components/VerdictBanner";
import { api, exportUrl, OperatorReport, setLastFlightId } from "@/lib/api";
import { clock, duration, num, percent } from "@/lib/format";

type Appendix = {
  detector?: Record<string, unknown> & { features?: string[]; threshold?: number };
  scores?: Record<string, number | null>;
  events?: { event_id: string; peak_score: number | null; mean_flagged_score: number | null; peak_over_threshold: number | null; window_indices: number[] }[];
  data_quality?: unknown;
  provenance?: unknown;
  grouping_policy?: unknown;
  not_implemented_checks?: Record<string, string>;
  report_schema_version?: string;
};

export default function OperatorReportPage() {
  const flightId = String(useParams().id || "");
  const [rep, setRep] = useState<OperatorReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [trace, setTrace] = useState("");

  useEffect(() => {
    setLastFlightId(flightId);
    api.operatorReport(flightId).then(setRep).catch((e) => setError(e instanceof Error ? e.message : "Failed to load report"));
    api.evidenceTrace(flightId).then((t) => setTrace(t.formatted)).catch(() => undefined);
  }, [flightId]);

  const ov = rep?.overview;
  const ap = (rep?.technical_appendix ?? {}) as Appendix;
  const cov = ov?.coverage;

  return (
    <>
      <AppNav />
      <main>
        <div className="page-head row between">
          <div>
            <h1>Flight report</h1>
            <p className="sub mono">{flightId}</p>
          </div>
          <button className="btn no-print" onClick={() => window.print()} disabled={!rep}>
            Print or save as PDF
          </button>
        </div>
        <FlightTabs flightId={flightId} />
        {error && (
          <div className="alert alert-info">
            {error}. Run the analysis from the Overview first.
          </div>
        )}
        {!rep && !error && <p className="muted">Loading…</p>}

        {rep && ov && (
          <>
            <section className="section">
              <h2>1. Flight summary</h2>
              <VerdictBanner verdict={ov.verdict} />
              <p>{ov.summary_text}</p>
              <p className="muted small">
                Recording: {ov.duration.flight_s != null ? duration(ov.duration.flight_s) : "length not recorded"} ·
                Events flagged: {ov.events.length}
                {ov.analyzed_at ? ` · Analysed ${new Date(ov.analyzed_at).toLocaleString()}` : ""}
              </p>
            </section>

            <section className="section">
              <h2>2. Analysis coverage and limitations</h2>
              <ul>
                <li>
                  Data assessed:{" "}
                  {cov?.known ? `${percent(cov.analysed_fraction)} (${cov.scored_windows ?? "?"} of ${cov.candidate_windows ?? "?"} analysis windows)` : "not recorded for this analysis"}.
                </li>
                <li>Only 12 attitude and inertial signals are scored; altitude, pressure, motors and radio inputs are not.</li>
                <li>The detector reports unusual telemetry, not faults, and gives no cause.</li>
                {cov?.missing_signals?.length ? <li>Missing signals: {cov.missing_signals.join(", ")}.</li> : null}
                {ov.warnings.slice(0, 6).map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            </section>

            <section className="section">
              <h2>3. Detected events</h2>
              {rep.events.length === 0 && <p className="muted">No events were flagged in the assessed part of the flight.</p>}
              {rep.events.map((e, i) => (
                <article key={e.event_id} style={{ marginBottom: "1.75rem" }}>
                  <h3 style={{ fontSize: "1.05rem" }}>
                    Event {i + 1} — {e.start_label} to {e.end_label}
                  </h3>
                  <EventDetailPanel flightId={flightId} detail={e} showAi={false} />
                </article>
              ))}
            </section>

            <section className="section">
              <h2>4. Periods that could not be assessed</h2>
              {rep.unassessed.length === 0 ? (
                <p className="muted">{cov?.known ? "None recorded." : "Not recorded for this analysis."}</p>
              ) : (
                <ul>
                  {rep.unassessed.map((u, i) => (
                    <li key={i}>
                      {clock(u.start_rel_s)}–{clock(u.end_rel_s)} ({duration(u.duration_s)}): {u.text}
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section className="section">
              <h2>5. Suggested review actions</h2>
              <ul>
                {rep.events.map((e) => (
                  <li key={e.event_id}>
                    {e.start_label}–{e.end_label}: {e.suggested_action}
                  </li>
                ))}
                {rep.unassessed.length > 0 && (
                  <li>Treat the unassessed periods as unknown; review them from the raw log if they matter.</li>
                )}
                {rep.events.length === 0 && <li>Review the flight as you normally would; this report adds no findings.</li>}
              </ul>
            </section>

            <section className="section">
              <h2>6. Method and model version</h2>
              <p>{rep.methodology}</p>
              <p className="muted small">
                Model: {String(rep.model_version.model)} ({String(rep.model_version.variant)}), feature schema{" "}
                {String(rep.model_version.feature_schema_version ?? "n/a")}. {rep.disclaimer}
              </p>
            </section>

            <details className="appendix">
              <summary>Technical appendix</summary>
              <div>
                <p className="muted small">Model configuration, scores and preprocessing for engineers.</p>
                <h3>Detector</h3>
                <p className="small">
                  {String(ap.detector?.model)} · variant {String(ap.detector?.variant)} · window{" "}
                  {String(ap.detector?.window_length)} samples, stride {String(ap.detector?.stride)} · alert limit{" "}
                  <span className="mono">{String(ap.detector?.threshold)}</span>
                </p>
                <p className="small">Features: <span className="mono">{ap.detector?.features?.join(", ")}</span></p>
                <h3>Score summary</h3>
                <p className="small mono">{JSON.stringify(ap.scores)}</p>
                {ap.events && ap.events.length > 0 && (
                  <div className="table-wrap">
                    <table className="data">
                      <thead>
                        <tr><th>Event</th><th>Peak score</th><th>Mean flagged</th><th>Peak ÷ limit</th><th>Windows</th></tr>
                      </thead>
                      <tbody>
                        {ap.events.map((e) => (
                          <tr key={e.event_id}>
                            <td className="mono">{e.event_id}</td>
                            <td className="num">{num(e.peak_score)}</td>
                            <td className="num">{num(e.mean_flagged_score)}</td>
                            <td className="num">{num(e.peak_over_threshold)}</td>
                            <td className="num">{e.window_indices.join(", ")}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                <h3>Not implemented</h3>
                <ul className="small">
                  {Object.values(ap.not_implemented_checks ?? {}).map((t) => (
                    <li key={t}>{t}</li>
                  ))}
                </ul>
                <h3>Provenance and data quality</h3>
                <pre className="trace">{JSON.stringify({ provenance: ap.provenance, data_quality: ap.data_quality, grouping_policy: ap.grouping_policy }, null, 2)}</pre>
                {trace && (
                  <>
                    <h3>Evidence pipeline trace</h3>
                    <pre className="trace">{trace}</pre>
                  </>
                )}
                <h3>Downloads</h3>
                <div className="row no-print">
                  <a className="btn" href={exportUrl(flightId, "json")}>Report JSON</a>
                  <a className="btn" href={exportUrl(flightId, "windows.csv")}>Windows CSV</a>
                  <a className="btn" href={exportUrl(flightId, "events.csv")}>Events CSV</a>
                </div>
              </div>
            </details>
          </>
        )}
      </main>
    </>
  );
}
