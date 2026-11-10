"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { AppNav } from "@/components/AppNav";
import { DirectEventChart } from "@/components/DirectEventChart";
import { PilotBriefing } from "@/components/PilotBriefing";
import {
  api,
  ApiError,
  DirectEvent,
  DirectEventContext,
  DirectOperatorEvent,
  DirectReport,
  DirectView,
  DirectWindow,
  PilotReport,
} from "@/lib/api";

const CAT_TEXT: Record<string, string> = {
  statistical_irregularity: "Statistical irregularity",
  state_transition: "State / pilot-input change",
  data_quality: "Data quality",
};
const KIND_TEXT: Record<string, string> = {
  spike: "brief spike",
  sustained_shift: "sustained shift",
  transient_excursion: "excursion that returned",
  variability_increase: "larger sample-to-sample variation",
  flatline: "repeated identical values",
  missing_values: "missing values",
  sampling_gap: "gap in the time column",
  status_change: "status change",
  command_change: "pilot-command change",
};
const REASON_TEXT: Record<string, string> = {
  truncated: "reply cut off at the output limit",
  not_json: "reply was not valid JSON",
  not_object: "reply was not a JSON object",
  empty: "reply was empty",
  schema: "reply had missing or wrongly typed fields",
  refusal: "the model declined",
  api_rate_limit: "rate limited",
  api_timeout: "timed out",
  api_auth: "credentials rejected",
  api_transient: "service unreachable",
  api_other: "request failed",
};

const NA = "Unavailable";
function f(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return NA;
  if (v === 0) return "0";
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : Number(v.toPrecision(3)).toString();
}
function win(w: DirectWindow): { text: string; sub?: string; na: boolean } {
  if (!w.available) return { text: NA, na: true };
  return { text: f(w.median), sub: `${f(w.min)} to ${f(w.max)} · n=${w.n}`, na: false };
}

export default function DirectAnalysisPage() {
  const id = String(useParams().id || "");
  const [view, setView] = useState<DirectView | null>(null);
  const [contexts, setContexts] = useState<Record<string, DirectEventContext | null>>({});
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rows, setRows] = useState<Awaited<ReturnType<typeof api.directRows>> | null>(null);

  const load = useCallback(async () => {
    setView(await api.directGet(id));
  }, [id]);

  useEffect(() => {
    load().catch((e) => setError(e instanceof Error ? e.message : "Failed to load"));
  }, [load]);

  const eventIds = useMemo(() => (view?.summary.events ?? []).map((e) => e.id), [view]);

  useEffect(() => {
    if (!eventIds.length || view?.stale_analysis) return;
    let live = true;
    Promise.all(
      eventIds.map(async (eventId) => {
        try {
          return [eventId, await api.directContext(id, eventId)] as const;
        } catch {
          return [eventId, null] as const;
        }
      })
    ).then((entries) => {
      if (live) setContexts(Object.fromEntries(entries));
    });
    return () => {
      live = false;
    };
  }, [id, eventIds, view?.stale_analysis]);

  async function generate() {
    setBusy(true);
    setError(null);
    try {
      setView(await api.directAi(id));
    } catch (e) {
      setError(e instanceof ApiError || e instanceof Error ? e.message : "Could not generate");
    } finally {
      setBusy(false);
    }
  }

  const rep: DirectReport | null = view?.report ?? null;
  const usable = !!rep && (rep.status === "ok" || rep.status === "partial");
  const s = view?.summary;
  const op = s?.operator;
  const pilot: PilotReport | undefined = op?.pilot_report;
  const pilotEvents = useMemo(() => Object.values(op?.pilot_events ?? {}), [op]);
  const aiFailed = rep && !usable && rep.status !== "not_requested";
  const lastAttempt = view?.meta.ai_attempt;

  const facts =
    s && pilot
      ? [
          { label: "Rows", value: String(s.data_quality.rows) },
          {
            label: "Recording",
            value: s.timing.duration_s ? `${f(s.timing.duration_s.value)} s` : "no usable time column",
            note:
              s.timing.duration_s && s.timing.duration_s.basis !== "verified"
                ? "conditional on the unverified time unit"
                : undefined,
          },
          { label: "Periods flagged", value: String(s.events.length) },
        ]
      : [];

  return (
    <>
      <AppNav />
      <main>
        <div className="page-head row between">
          <div>
            <p className="brief-page-kicker">AI Analysis · Flight briefing</p>
            <h1>What happened on this flight</h1>
            <p className="sub mono">{view?.meta.filename ?? id}</p>
          </div>
          <div className="row no-print">
            <button className="btn" onClick={() => window.print()} disabled={!view}>
              Print or save as PDF
            </button>
            <a className="btn" href={api.directExportUrl(id)}>
              Export JSON
            </a>
          </div>
        </div>

        {error && <div className="alert alert-error">{error}</div>}
        {!view && !error && <p className="muted">Loading…</p>}

        {view && s && view.stale_analysis && (
          <div className="alert alert-warn" role="alert">
            This analysis was created by an older version of the analyser and has no pilot briefing. Upload the file
            again to regenerate it. <Link href="/upload">Upload again</Link>
          </div>
        )}

        {view && s && op && pilot && (
          <>
            <PilotBriefing
              report={pilot}
              events={pilotEvents}
              facts={facts}
              charts={(eventId) => {
                const ctx = contexts[eventId];
                if (ctx === undefined) return <p className="muted small">Loading chart…</p>;
                if (ctx === null) return null;
                return <DirectEventChart ctx={ctx} maxChannels={3} />;
              }}
              summaryExtra={
                <>
                  <div className="ai-summary-block">
                    {usable && rep!.summary_source === "ai" && (
                      <div className="ai-note">
                        <p className="small" style={{ margin: 0 }}>
                          <span className="label-tag">Optional AI-written narrative</span>
                        </p>
                        <p style={{ margin: "0.3rem 0" }}>{rep!.executive_summary}</p>
                        <p className="small muted" style={{ margin: 0 }}>
                          Written by an AI model from the computed measurements and checked against them. It is not a
                          detector result and does not change the briefing above.
                        </p>
                      </div>
                    )}
                    {!aiFailed && rep?.message && usable && <p className="small muted">{rep.message}</p>}
                    {rep?.reused_from && (
                      <p className="small muted">
                        Reused the report of identical input ({rep.reused_from}); no new AI call was made.
                      </p>
                    )}

                    {!usable && (
                      <details className="ai-optional no-print">
                        <summary>
                          {aiFailed ? "AI-written narrative unavailable" : "Optional: add an AI-written narrative"}
                        </summary>
                        {aiFailed && (
                          <div className="alert alert-warn ai-error" role="status">
                            <span>
                              <strong>AI narrative not available.</strong>{" "}
                              {rep!.message || "The AI step did not complete."} The briefing above was computed from
                              the file and is unaffected.
                            </span>
                            {rep!.failure && (
                              <details className="small">
                                <summary>Why</summary>
                                <p>
                                  {REASON_TEXT[rep!.failure.reason] ?? rep!.failure.reason}
                                  {rep!.failure.attempts ? ` · ${rep!.failure.attempts} attempt(s)` : ""}
                                  {rep!.failure.fields && rep!.failure.fields.length
                                    ? ` · ${rep!.failure.fields.join(", ")}`
                                    : ""}
                                </p>
                              </details>
                            )}
                          </div>
                        )}
                        <p className="small muted">
                          This adds a plain-language narrative written by an AI model. The computed briefing is the
                          primary report and does not depend on it.
                        </p>
                        <label className="checkline">
                          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
                          <span>{view.consent_text}</span>
                        </label>
                        <button className="btn btn-primary" disabled={!consent || busy} onClick={generate}>
                          {busy
                            ? "Generating…"
                            : aiFailed || lastAttempt?.status === "invalid_output"
                              ? "Try AI narrative again"
                              : "Add an AI-written narrative"}
                        </button>
                      </details>
                    )}
                  </div>
                </>
              }
              technical={
                <TechnicalDetails
                  id={id}
                  s={s}
                  op={op}
                  contexts={contexts}
                  rep={rep}
                  usable={usable}
                  rows={rows}
                  setRows={setRows}
                />
              }
            />
            <p className="small muted">
              {view.separation_note} <Link href="/">Back to flights</Link>
            </p>
          </>
        )}

        {view && s && op && !pilot && (
          <div className="alert alert-info">
            This analysis has no pilot briefing payload. Upload the file again to regenerate it.
          </div>
        )}
      </main>
    </>
  );
}

function TechnicalDetails({
  id,
  s,
  op,
  contexts,
  rep,
  usable,
  rows,
  setRows,
}: {
  id: string;
  s: NonNullable<DirectView["summary"]>;
  op: NonNullable<NonNullable<DirectView["summary"]>["operator"]>;
  contexts: Record<string, DirectEventContext | null>;
  rep: DirectReport | null;
  usable: boolean;
  rows: Awaited<ReturnType<typeof api.directRows>> | null;
  setRows: (r: Awaited<ReturnType<typeof api.directRows>>) => void;
}) {
  const timeCol = s.timing.time_column ?? null;
  return (
    <>
      <h3>How events were found</h3>
      <p className="small">
        Each channel is compared with the median and robust spread (MAD) of its own preceding{" "}
        {s.engine?.baseline_window_samples ?? "—"} samples. A departure counts when it exceeds{" "}
        {s.engine?.z_threshold ?? "—"} robust sigmas and at least{" "}
        {Math.round((s.engine?.min_rel_change ?? 0) * 100)}% of the channel&apos;s range, floored by the channel&apos;s
        own noise and recording resolution. Departures of at most {s.engine?.spike_max_samples ?? "—"} samples are brief
        spikes and need three times the evidence. Longer ones are sustained shifts if the level is still displaced
        afterwards, otherwise excursions that returned. Variation is judged on sample-to-sample changes in blocks.
        Detections within {s.engine?.merge_tolerance_samples ?? "—"} samples are merged inside one category (data
        quality, state/pilot input, statistical irregularity); events in different categories are linked, not merged.
        Onsets within {s.engine?.simultaneity_samples ?? "—"} samples are called simultaneous. Baselines need{" "}
        {s.engine?.baseline_min_samples ?? "—"} earlier samples and post-event comparisons{" "}
        {s.engine?.post_min_samples ?? "—"} later samples; otherwise “{NA}” is shown. None of this uses the AI model or
        the anomaly detector.
      </p>

      <h3>Schema mapping and channel meanings</h3>
      <p className="small">
        Profile:{" "}
        {s.profile.id
          ? `${s.profile.id} v${s.profile.version} (${s.profile.match} match, documentation ${
              s.profile.documentation_status ?? "unknown"
            })`
          : "unrecognised layout"}
        . Time column <span className="mono">{timeCol ?? "none"}</span>: unit{" "}
        {s.timing.time_basis.status === "verified"
          ? "verified"
          : `not verified (hypothesis: ${s.timing.time_basis.unit_hypothesis ?? "none"})`}
        .
      </p>
      <div className="table-wrap">
        <table className="data stat-table">
          <thead>
            <tr>
              <th>Column</th>
              <th>Group</th>
              <th>Meaning</th>
              <th>Unit</th>
              <th>Min</th>
              <th>Median</th>
              <th>Max</th>
              <th>Std</th>
              <th>Missing</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(s.channels)
              .filter(([h]) => s.stats[h])
              .map(([h, c]) => {
                const st = s.stats[h];
                return (
                  <tr key={h}>
                    <td className="mono">{h}</td>
                    <td>{c.group}</td>
                    <td>{c.meaning_status}</td>
                    <td>
                      {c.unit_status}
                      {c.unit_hypothesis ? ` (${c.unit_hypothesis})` : ""}
                    </td>
                    <td className="num">{f(st.min as number)}</td>
                    <td className="num">{f(st.median as number)}</td>
                    <td className="num">{f(st.max as number)}</td>
                    <td className="num">{f(st.std as number)}</td>
                    <td className="num">{String(st.n_missing)}</td>
                  </tr>
                );
              })}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        All values are raw. Channels marked “unverified” or “candidate” have unconfirmed meanings and units; no
        conversion is applied.
      </p>

      <h3>Events — engineering evidence</h3>
      {s.events.length === 0 && <p className="muted">No intervals of notable change were found.</p>}
      {s.events.map((ev) => (
        <TechnicalEvent key={ev.id} ev={ev} op={op.events[ev.id]} ctx={contexts[ev.id] ?? null} timeCol={timeCol} />
      ))}

      <h3>Largest departures from each column’s own median</h3>
      <ul className="small">
        {s.peaks.map((p) => (
          <li key={p.id} className="mono">
            {p.id}: row {p.row}, time {f(p.raw_time)}, value {f(p.value)} (median {f(p.channel_median)})
          </li>
        ))}
      </ul>

      <h3>Whole-recording correlations</h3>
      <ul className="small">
        {[...s.correlations.pairs, ...s.correlations.rate_pairs].map((p) => (
          <li key={p.a + p.b} className="mono">
            {p.a} ~ {p.b}: r = {p.r}
          </li>
        ))}
      </ul>
      <p className="small muted">{s.correlations.note}</p>

      <h3>Data quality</h3>
      {op.data_quality_material.length > 0 || s.unassessed_periods.length > 0 ? (
        <ul className="small">
          {op.data_quality_material.map((d) => (
            <li key={d.issue}>
              <strong>{d.issue}.</strong> {d.detail}
            </li>
          ))}
          {s.unassessed_periods.map((u, i) => (
            <li key={i}>
              <strong>Period the data cannot speak for:</strong> rows {u.start_row}–{u.end_row},{" "}
              {u.kinds.map((k) => KIND_TEXT[k] ?? k).join(", ")}
              {u.channels.length ? ` in ${u.channels.join(", ")}` : ""}.
            </li>
          ))}
        </ul>
      ) : (
        <p className="small muted">No material data-quality limitations recorded.</p>
      )}

      <h3>Complete limitations</h3>
      <ul className="small">
        {s.limitations.map((l) => (
          <li key={l}>{l}</li>
        ))}
      </ul>

      {rep && (
        <>
          <h3>AI model output</h3>
          {usable ? (
            <>
              {rep.events.map((n) => (
                <div key={n.event_id} className="ai-note small">
                  <strong>
                    {n.event_id}: {n.title}
                  </strong>
                  <div>{n.summary}</div>
                  {n.same_time && <div>{n.same_time}</div>}
                  {n.conclusion && <div>{n.conclusion}</div>}
                </div>
              ))}
              {rep.unknowns.length > 0 && <p className="small">Unknowns (AI): {rep.unknowns.join(" ")}</p>}
              {rep.data_quality.length > 0 && (
                <ul className="small">
                  {rep.data_quality.map((d) => (
                    <li key={d.issue}>
                      <strong>{d.issue}.</strong> {d.detail}
                    </li>
                  ))}
                </ul>
              )}
            </>
          ) : (
            <p className="small muted">No usable AI output. {rep.message}</p>
          )}
          {rep.rejected.length > 0 && (
            <>
              <h3>AI statements removed by validation</h3>
              <ul className="small">
                {rep.rejected.map((r, i) => (
                  <li key={i}>
                    <span className="mono">{r.section}</span> — {r.reason}: “{r.text}”
                  </li>
                ))}
              </ul>
            </>
          )}
          {rep.normalized && rep.normalized.length > 0 && (
            <p className="small muted">
              Normalised: {rep.normalized.map((n) => `${n.where} (${n.action.replace(/_/g, " ")})`).join("; ")}.
            </p>
          )}
        </>
      )}

      <h3>Provenance</h3>
      <pre className="trace">
        {JSON.stringify(
          {
            analysis_id: id,
            analysis_version: s.analysis_version,
            input_fingerprint: s.input_fingerprint,
            report: rep?.provenance ?? null,
            failure: rep?.failure ?? null,
            profile: s.profile,
          },
          null,
          2
        )}
      </pre>

      <h3>Raw rows</h3>
      {!rows ? (
        <button className="btn no-print" onClick={() => api.directRows(id, 200).then(setRows)}>
          Show first 200 rows
        </button>
      ) : (
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                {rows.columns.map((c) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.rows.map((r, i) => (
                <tr key={i}>
                  {rows.columns.map((c) => (
                    <td className="num" key={c}>
                      {r[c] ?? ""}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

function TechnicalEvent({
  ev,
  op,
  ctx,
  timeCol,
}: {
  ev: DirectEvent;
  op: DirectOperatorEvent;
  ctx: DirectEventContext | null;
  timeCol: string | null;
}) {
  return (
    <article className="tech-event">
      <h4>
        {ev.id} · rows {ev.start_row}–{ev.end_row}
      </h4>
      <p className="small muted">
        <span className={`cat-chip cat-${ev.category}`}>{CAT_TEXT[ev.category]}</span>
        {ev.kinds.map((k) => (
          <span key={k} className="cat-chip">
            {KIND_TEXT[k] ?? k}
          </span>
        ))}
        {timeCol ? ` · ${timeCol} ${f(ev.start_raw)}–${f(ev.end_raw)} raw ticks` : " · no time column"}
      </p>
      {ctx ? <DirectEventChart ctx={ctx} technical /> : <p className="muted small">No chart data.</p>}

      <h4>Before, during and after (raw values)</h4>
      <div className="table-wrap">
        <table className="data stat-table">
          <thead>
            <tr>
              <th>Channel</th>
              <th>
                Before: median
                <small className="muted" style={{ display: "block" }}>
                  range · samples
                </small>
              </th>
              <th>During</th>
              <th>After</th>
              <th>Onset row</th>
            </tr>
          </thead>
          <tbody>
            {ev.measurements.map((m) => {
              const b = win(m.before),
                d = win(m.during),
                a = win(m.after);
              return (
                <tr key={m.channel}>
                  <td className="mono">{m.channel}</td>
                  <td className={b.na ? "na" : "num"}>
                    {b.text}
                    {b.sub && (
                      <small className="muted" style={{ display: "block" }}>
                        {b.sub}
                      </small>
                    )}
                  </td>
                  <td className="num">
                    {d.text}
                    {d.sub && (
                      <small className="muted" style={{ display: "block" }}>
                        {d.sub}
                      </small>
                    )}
                  </td>
                  <td className={a.na ? "na" : "num"}>
                    {a.text}
                    {a.sub && (
                      <small className="muted" style={{ display: "block" }}>
                        {a.sub}
                      </small>
                    )}
                  </td>
                  <td className="num">{m.onset_row}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="small muted">
        “{NA}” means the recording has too few samples before or after the event for a comparison (fewer than 20
        before, 10 after). Raw values; units of attitude, accelerometer, gyroscope, controller and time are not
        verified. Source: rows {ev.start_row}–{ev.end_row} of the uploaded file.
      </p>

      <h4>What changed</h4>
      <ul>
        {op.what_changed.map((t) => (
          <li key={t}>{t}</li>
        ))}
      </ul>

      <h4>What changed at about the same time</h4>
      {op.same_time.length === 0 && ev.concurrent_with.length === 0 ? (
        <p className="small muted">No other channel or event began changing at about the same time.</p>
      ) : (
        <ul>
          {op.same_time.map((t) => (
            <li key={t.text}>
              <span className="label-tag">{t.kind}</span> {t.text}
            </li>
          ))}
          {ev.concurrent_with.map((c) => (
            <li key={c.event_id}>
              <span className="label-tag">linked event</span> {c.event_id} ({CAT_TEXT[c.category]}) is {c.relation}.
            </li>
          ))}
        </ul>
      )}

      <h4>What can be concluded · what is unknown</h4>
      <dl className="qa">
        <dt>Supported</dt>
        <dd>
          <ul>
            {op.can_conclude.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </dd>
        <dt>Unknown</dt>
        <dd>
          <ul>
            {op.unknown.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </dd>
        <dt>Review next</dt>
        <dd>
          <ul>
            {op.review_next.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </dd>
      </dl>
    </article>
  );
}
