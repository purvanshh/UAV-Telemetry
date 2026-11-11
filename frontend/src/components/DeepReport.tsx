"use client";

import { useEffect, useMemo, useState } from "react";
import { EventChart } from "@/components/EventChart";
import { EventDetailPanel } from "@/components/EventDetailPanel";
import { PilotBriefing } from "@/components/PilotBriefing";
import { VerdictBanner } from "@/components/VerdictBanner";
import { api, EventContext, exportUrl, OperatorReport, setLastFlightId } from "@/lib/api";
import { clock, duration, num, percent } from "@/lib/format";

type Appendix = {
  detector?: Record<string, unknown> & { features?: string[]; threshold?: number };
  scores?: Record<string, number | null>;
  events?: {
    event_id: string;
    peak_score: number | null;
    mean_flagged_score: number | null;
    peak_over_threshold: number | null;
    window_indices: number[];
    flagged_window_count?: number;
    affected_signals?: string[];
  }[];
  signal_reference?: { feature: string; pilot_label: string; technical_label: string; unit: string | null }[];
  data_quality?: unknown;
  provenance?: unknown;
  grouping_policy?: unknown;
  not_implemented_checks?: Record<string, string>;
  report_schema_version?: string;
};

export function DeepReport({ flightId }: { flightId: string }) {
  const [rep, setRep] = useState<OperatorReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [trace, setTrace] = useState("");
  const [contexts, setContexts] = useState<Record<string, EventContext | null>>({});

  useEffect(() => {
    setLastFlightId(flightId);
    api
      .operatorReport(flightId)
      .then(setRep)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load report"));
    api.evidenceTrace(flightId).then((t) => setTrace(t.formatted)).catch(() => undefined);
  }, [flightId]);

  useEffect(() => {
    if (!rep) return;
    let live = true;
    Promise.all(
      rep.events.map(async (e) => {
        try {
          const ctx = await api.eventContext(flightId, e.event_id);
          return [e.event_id, ctx] as const;
        } catch {
          return [e.event_id, null] as const;
        }
      })
    ).then((entries) => {
      if (live) setContexts(Object.fromEntries(entries));
    });
    return () => {
      live = false;
    };
  }, [rep, flightId]);

  const ov = rep?.overview;
  const ap = (rep?.technical_appendix ?? {}) as Appendix;
  const cov = ov?.coverage;

  const beforeByEvent = useMemo(() => {
    const out: Record<string, Record<string, number | null>> = {};
    for (const e of rep?.events ?? []) {
      out[e.event_id] = Object.fromEntries(e.channels.map((c) => [c.signal, c.before?.median ?? null]));
    }
    return out;
  }, [rep]);

  const facts = ov
    ? [
        ...(ov.duration.flight_s != null
          ? [{ label: "Recording", value: duration(ov.duration.flight_s) }]
          : []),
        ...(cov?.known
          ? [
              {
                label: "Assessed",
                value: percent(cov.analysed_fraction),
                note: `${cov.scored_windows ?? "?"} of ${cov.candidate_windows ?? "?"} windows`,
              },
            ]
          : []),
        { label: "Periods flagged", value: String(ov.events.length) },
      ]
    : [];

  return (
    <>
        {error && (
          <div className="alert alert-info">
            {error}. If this flight has not been analysed yet, run the analysis first.
          </div>
        )}
        {!rep && !error && <p className="muted">Loading…</p>}

        {rep && ov && rep.pilot_report && (
          <PilotBriefing
            report={rep.pilot_report}
            events={Object.values(rep.pilot_events ?? {})}
            facts={facts}
            charts={(eventId) => {
              const ctx = contexts[eventId];
              if (ctx === undefined) return <p className="muted small">Loading chart…</p>;
              if (ctx === null) return null;
              return <EventChart ctx={ctx} before={beforeByEvent[eventId]} maxChannels={3} />;
            }}
            technical={
              <>
                <h3>Analysis coverage and limitations</h3>
                <ul>
                  <li>
                    Data assessed:{" "}
                    {cov?.known
                      ? `${percent(cov.analysed_fraction)} (${cov.scored_windows ?? "?"} of ${cov.candidate_windows ?? "?"} analysis windows)`
                      : "not recorded for this analysis"}
                    .
                  </li>
                  <li>
                    Only the attitude and inertial signals used by the detector are scored; altitude, pressure, motors
                    and radio inputs are not.
                  </li>
                  <li>The detector reports unusual telemetry, not faults, and gives no cause.</li>
                  {cov?.missing_signals?.length ? <li>Missing signals: {cov.missing_signals.join(", ")}.</li> : null}
                  {Array.from(new Set([...ov.warnings, ...ov.upload_warnings])).slice(0, 8).map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
                <VerdictBanner verdict={ov.verdict} />

                <h3>Periods that could not be assessed</h3>
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

                <h3>Detected events — engineering evidence</h3>
                {rep.events.length === 0 && (
                  <p className="muted">No events were flagged in the assessed part of the flight.</p>
                )}
                {rep.events.map((e) => {
                  const card = rep.pilot_events?.[e.event_id];
                  return (
                    <article key={e.event_id} className="tech-event">
                      <h4>
                        {e.event_id} — {e.start_label} to {e.end_label}
                      </h4>
                      {card && <p className="small muted">Briefing title: “{card.title}”</p>}
                      <EventDetailPanel flightId={flightId} detail={e} showAi={false} />
                      {contexts[e.event_id] && (
                        <>
                          <h3>Signal context — all affected channels</h3>
                          <EventChart ctx={contexts[e.event_id]!} before={beforeByEvent[e.event_id]} />
                        </>
                      )}
                    </article>
                  );
                })}

                <h3>Method and model version</h3>
                <p>{rep.methodology}</p>
                <p className="muted small">
                  Model: {String(rep.model_version.model)} ({String(rep.model_version.variant)}), feature schema{" "}
                  {String(rep.model_version.feature_schema_version ?? "n/a")}. {rep.disclaimer}
                </p>

                <h3>Signal reference</h3>
                {ap.signal_reference?.length ? (
                  <div className="table-wrap">
                    <table className="data">
                      <thead>
                        <tr>
                          <th>Signal (engineering)</th>
                          <th>Engineering label</th>
                          <th>Pilot briefing label</th>
                          <th>Unit</th>
                        </tr>
                      </thead>
                      <tbody>
                        {ap.signal_reference.map((s) => (
                          <tr key={s.feature}>
                            <td className="mono">{s.feature}</td>
                            <td>{s.technical_label}</td>
                            <td>{s.pilot_label}</td>
                            <td>{s.unit ?? "not established"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="muted small">Not recorded.</p>
                )}

                <h3>Detector</h3>
                <p className="small">
                  {String(ap.detector?.model)} · variant {String(ap.detector?.variant)} · window{" "}
                  {String(ap.detector?.window_length)} samples, stride {String(ap.detector?.stride)} · alert limit{" "}
                  <span className="mono">{String(ap.detector?.threshold)}</span>
                </p>
                <p className="small">
                  Features: <span className="mono">{ap.detector?.features?.join(", ")}</span>
                </p>

                <h3>Score summary</h3>
                <p className="small mono">{JSON.stringify(ap.scores)}</p>
                {ap.events && ap.events.length > 0 && (
                  <div className="table-wrap">
                    <table className="data">
                      <thead>
                        <tr>
                          <th>Event</th>
                          <th>Peak score</th>
                          <th>Mean flagged</th>
                          <th>Peak ÷ limit</th>
                          <th>Flagged windows</th>
                          <th>Windows</th>
                        </tr>
                      </thead>
                      <tbody>
                        {ap.events.map((e) => (
                          <tr key={e.event_id}>
                            <td className="mono">{e.event_id}</td>
                            <td className="num">{num(e.peak_score)}</td>
                            <td className="num">{num(e.mean_flagged_score)}</td>
                            <td className="num">{num(e.peak_over_threshold)}</td>
                            <td className="num">{e.flagged_window_count ?? "—"}</td>
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
                <pre className="trace">
                  {JSON.stringify(
                    { provenance: ap.provenance, data_quality: ap.data_quality, grouping_policy: ap.grouping_policy },
                    null,
                    2
                  )}
                </pre>
                {trace && (
                  <>
                    <h3>Evidence pipeline trace</h3>
                    <pre className="trace">{trace}</pre>
                  </>
                )}

                {ov.input && (
                  <>
                    <h3>How this file was read</h3>
                    <p className="small muted">
                      {ov.input.filename} · {ov.input.rows?.toLocaleString()} rows
                      {ov.input.profile_id ? ` · layout profile ${ov.input.profile_id}` : ""} · mapping v
                      {ov.input.mapping_version}
                      {ov.input.llm?.status === "ok" ? " · AI helped name some columns" : ""}
                    </p>
                    <div className="table-wrap">
                      <table className="data">
                        <thead>
                          <tr>
                            <th>Column</th>
                            <th>Meaning</th>
                            <th>Unit</th>
                            <th>Mapped by</th>
                          </tr>
                        </thead>
                        <tbody>
                          {ov.input.columns
                            .filter((c) => c.canonical || c.ignored)
                            .map((c) => (
                              <tr key={c.source}>
                                <td className="mono">{c.source}</td>
                                <td>{c.ignored ? "not used" : c.canonical}</td>
                                <td>{c.ignored ? "—" : `${c.unit ?? "?"}${c.converted ? ` → ${c.expected_unit}` : ""}`}</td>
                                <td className="small">{c.method?.replace("_", " ")}</td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}

                <h3>Downloads</h3>
                <div className="row no-print">
                  <a className="btn" href={exportUrl(flightId, "json")}>
                    Report JSON
                  </a>
                  <a className="btn" href={exportUrl(flightId, "windows.csv")}>
                    Windows CSV
                  </a>
                  <a className="btn" href={exportUrl(flightId, "events.csv")}>
                    Events CSV
                  </a>
                </div>
              </>
            }
          />
        )}

        {rep && !rep.pilot_report && (
          <div className="alert alert-info">
            This report was created by an older version of the analyser and has no pilot briefing. Re-run the analysis
            to regenerate it.
          </div>
        )}
    </>
  );
}
