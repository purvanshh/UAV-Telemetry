"use client";

import { useMemo } from "react";
import { Download, FileCode2, Gauge, History, ListTree, ScanSearch, Search, Tags, Wrench } from "lucide-react";
import { FindingCard } from "@/components/FindingCard";
import { Alert, ButtonLink, Pill, Tabs } from "@/components/ui";
import { api, type AnalysisView, type ArjunaSummary } from "@/lib/api";

export type EngTab = "findings" | "codes" | "physics" | "timeline" | "conversion" | "labels" | "review" | "technical";

/** Everything an engineer needs, one tab at a time. */
export function Engineering({ view, tab, onTab, target }: {
  view: AnalysisView;
  tab: EngTab;
  onTab: (t: EngTab) => void;
  target: string | null;
}) {
  const s = view.summary as ArjunaSummary;
  const run = view.report;
  const usable = !!run?.report && (run.status === "ok" || run.status === "partial");
  const tabs = useMemo(() => {
    const t: { id: EngTab; label: string; count?: number; icon: typeof Search }[] = [
      { id: "findings", label: "Findings", count: s.findings.length, icon: Search },
      { id: "codes", label: "Fault codes", icon: ListTree },
      { id: "physics", label: "Physics", icon: Gauge },
      { id: "timeline", label: "Timeline", icon: History },
    ];
    if (s.conversion) t.push({ id: "conversion", label: "Conversion", icon: FileCode2 });
    if (s.engineer_labels) t.push({ id: "labels", label: "Engineer labels", icon: Tags });
    if (usable) t.push({ id: "review", label: "AI review", icon: ScanSearch });
    t.push({ id: "technical", label: "Technical", icon: Wrench });
    return t;
  }, [s, usable]);

  return (
    <div className="engineering-panel">
      <Tabs label="Engineering details" tabs={tabs} value={tab} onChange={onTab} />
      <div className="tab-panel" role="tabpanel" aria-labelledby={`tab-${tab}`} key={tab}>
        {tab === "findings" && <Findings view={view} target={target} />}
        {tab === "codes" && <Coverage s={s} />}
        {tab === "physics" && <Physics s={s} />}
        {tab === "timeline" && <Timeline s={s} />}
        {tab === "conversion" && s.conversion && <Conversion id={view.analysis_id} conv={s.conversion} />}
        {tab === "labels" && <Labels s={s} />}
        {tab === "review" && <Review view={view} />}
        {tab === "technical" && <Technical s={s} />}
      </div>
    </div>
  );
}

function Findings({ view, target }: { view: AnalysisView; target: string | null }) {
  const s = view.summary as ArjunaSummary;
  const reviews = new Map((view.report?.report?.finding_reviews ?? []).map((r) => [r.finding_id, r]));
  if (!s.findings.length) {
    return <p className="text-2">No detector flagged anything on the checks that could run.</p>;
  }
  return (
    <>
      <p className="small muted" style={{ marginBottom: "0.9rem" }}>
        Found by code: rule checks with the approved ARJUNA thresholds, and the physics model. The detailed review can
        comment on them but cannot remove them.
      </p>
      {s.findings.map((f, i) => (
        <FindingCard
          key={f.id}
          id={view.analysis_id}
          finding={f}
          review={reviews.get(f.id)}
          codeLabel={s.fault_codes?.[f.code]}
          defaultOpen={!target && i === 0}
          target={target === f.id}
        />
      ))}
    </>
  );
}

function Coverage({ s }: { s: ArjunaSummary }) {
  if (!s.coverage) return null;
  return (
    <div className="table-wrap">
      <table className="data">
        <thead>
          <tr>
            <th>Code</th>
            <th>What it covers</th>
            <th>Result</th>
            <th>Checks run, or why not</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(s.coverage).map(([code, c]) => {
            const found = s.findings.filter((f) => f.code === code).length;
            return (
              <tr key={code}>
                <td>
                  <span className="code">{code}</span>
                </td>
                <td className="small">{s.fault_codes?.[code]}</td>
                <td>
                  {found ? (
                    <Pill tone="caution">{found} found</Pill>
                  ) : c.status === "evaluated" ? (
                    <Pill tone="go">Clear</Pill>
                  ) : c.status === "partial" ? (
                    <Pill tone="neutral">Events only</Pill>
                  ) : (
                    <Pill tone="neutral">Not checkable</Pill>
                  )}
                </td>
                <td className="small text-2">
                  {c.checks.join(", ")}
                  {c.status !== "evaluated" && c.reasons.length > 0 && (
                    <span className="muted">
                      {c.checks.length ? " — " : ""}
                      {c.reasons.join("; ")}
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Physics({ s }: { s: ArjunaSummary }) {
  const hb = s.physics?.hover_balance;
  const tr = s.physics?.thrust_residual;
  const dy = s.physics?.dynamics;
  if (!hb) return <p className="text-2">No physics checks for this recording.</p>;
  return (
    <div>
      <h3 className="subhead">Hover motor balance</h3>
      {hb.status === "computed" ? (
        <div className="stack">
          <p>{hb.interpretation}</p>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>Motor</th>
                  <th className="num">Mean command in steady hover</th>
                  <th className="num">vs. average</th>
                  <th style={{ width: "34%" }} />
                </tr>
              </thead>
              <tbody>
                {(hb.motors ?? []).map((m, i) => {
                  const d = hb.deviation_pct?.[i] ?? 0;
                  return (
                    <tr key={m} className={m === hb.worst_motor ? "row-strong" : ""}>
                      <td>{m}</td>
                      <td className="num">{hb.mean_cmd?.[i]?.toFixed(3)}</td>
                      <td className="num">
                        {d > 0 ? "+" : ""}
                        {d}%
                      </td>
                      <td>
                        <DeviationBar pct={d} limit={12} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="small muted">
            From {hb.steady_hover_s} s of steady hover. The marks show the approved 12% limit either side.
          </p>
        </div>
      ) : (
        <p className="small muted">Not computed: {hb.reason}</p>
      )}
      {tr && (
        <>
          <h3 className="subhead">Vertical thrust residual</h3>
          {tr.status === "computed" ? (
            <div className="stack" style={{ ["--gap" as string]: "0.5rem" }}>
              <p className="small text-2">
                Model <span className="mono">{tr.model}</span>, {tr.voltage_compensation}. Residual spread {tr.residual_robust_spread_mps2} m/s².{" "}
                {tr.sustained_departures?.length ? `${tr.sustained_departures.length} sustained departure(s):` : "No sustained departures."}
              </p>
              {!!tr.sustained_departures?.length && (
                <ul className="bullets">
                  {tr.sustained_departures.slice(0, 6).map((d) => (
                    <li key={d.when}>
                      <span className="mono">{d.when}</span> {d.direction} (median {d.median_residual_mps2} m/s²)
                    </li>
                  ))}
                </ul>
              )}
              <p className="tiny muted">{tr.threshold_note}</p>
            </div>
          ) : (
            <p className="small muted">Not computed: {tr.reason}</p>
          )}
        </>
      )}
      {dy && (
        <>
          <h3 className="subhead">Per-motor effectiveness (η)</h3>
          {dy.status === "computed" ? (
            <p className="small text-2">
              {dy.motors?.map((m, i) => `${m}: η ${dy.eta?.[i]} (${dy.relative_to_others?.[i]} of the others)`).join(" · ")}
              {dy.flag_motor ? ` — ${dy.flag_motor} flagged.` : " — no motor flagged."}
            </p>
          ) : (
            <p className="small muted">Not computed: {dy.reason}</p>
          )}
        </>
      )}
    </div>
  );
}

function DeviationBar({ pct, limit }: { pct: number; limit: number }) {
  const span = Math.max(limit * 1.6, Math.abs(pct) * 1.1);
  const w = (Math.abs(pct) / span) * 50;
  const over = Math.abs(pct) > limit;
  return (
    <div style={{ position: "relative", height: 8, borderRadius: 99, background: "var(--surface-2)" }} aria-hidden="true">
      <span style={{ position: "absolute", top: -3, bottom: -3, left: `${50 - (limit / span) * 50}%`, width: 1.5, background: "var(--line-strong)" }} />
      <span style={{ position: "absolute", top: -3, bottom: -3, left: `${50 + (limit / span) * 50}%`, width: 1.5, background: "var(--line-strong)" }} />
      <span style={{ position: "absolute", top: -2, bottom: -2, left: "50%", width: 1, background: "var(--text-3)" }} />
      <span
        style={{
          position: "absolute", top: 0, bottom: 0, borderRadius: 99, transition: "width 600ms var(--ease)",
          left: pct >= 0 ? "50%" : `${50 - w}%`, width: `${w}%`, background: over ? "var(--caution)" : "var(--accent)",
        }}
      />
    </div>
  );
}

function Timeline({ s }: { s: ArjunaSummary }) {
  const tl = s.timeline;
  if (!tl) return null;
  return (
    <div>
      <h3 className="subhead">Flight modes</h3>
      <ul className="bullets">
        {(tl.modes ?? []).map((m, i) => (
          <li key={i}>
            <span className="mono">{m.when}</span> {m.value.replace(/_/g, " ")}
          </li>
        ))}
      </ul>
      <h3 className="subhead">Events · {tl.events?.length ?? 0}</h3>
      {(tl.events ?? []).length ? (
        <ul className="bullets">
          {(tl.events ?? []).map((e, i) => (
            <li key={i}>
              <span className="mono">{e.at}</span> {e.event.replace(/_/g, " ")}
              {e.severity !== "info" ? <span style={{ color: "var(--warn)" }}> ({e.severity})</span> : ""}
              {e.mode_after ? ` → ${e.mode_after}` : ""}
            </li>
          ))}
        </ul>
      ) : (
        <p className="small muted">No events recorded.</p>
      )}
      <h3 className="subhead">Messages, warning and above · {tl.important_messages?.length ?? 0}</h3>
      {(tl.important_messages ?? []).length ? (
        <ul className="bullets">
          {(tl.important_messages ?? []).map((m, i) => (
            <li key={i}>
              <span className="mono">{m.at}</span> <span className="muted">[{m.level}]</span> {m.text}
            </li>
          ))}
        </ul>
      ) : (
        <p className="small muted">None.</p>
      )}
      {!!tl.dropouts?.length && (
        <>
          <h3 className="subhead">Recording gaps · {tl.dropouts.length}</h3>
          <p className="small text-2">{tl.dropouts.slice(0, 20).map((d) => `${d.t.toFixed(1)} s (${d.duration_ms} ms)`).join(", ")}</p>
        </>
      )}
    </div>
  );
}

function Conversion({ id, conv }: { id: string; conv: NonNullable<ArjunaSummary["conversion"]> }) {
  return (
    <div>
      <p className="text-2">
        This CSV ({conv.format.label}) was converted into a standard ARJUNA log first ({conv.converter_version}). Everything
        on this page was computed from that log.
      </p>
      <dl className="kv" style={{ marginTop: "0.9rem" }}>
        <dt>Time column</dt>
        <dd>
          <span className="mono">{conv.time.column}</span> in {conv.time.unit} ({conv.time.how})
        </dd>
        <dt>Sample rate</dt>
        <dd>{conv.time.sample_rate_hz ?? "—"} Hz</dd>
        <dt>Rows used</dt>
        <dd>
          {conv.rows_written} of {conv.rows_in}
          {conv.duplicate_timestamps_dropped ? ` · ${conv.duplicate_timestamps_dropped} duplicate timestamps dropped` : ""}
          {conv.rows_without_valid_time ? ` · ${conv.rows_without_valid_time} without a valid time` : ""}
        </dd>
      </dl>
      <div className="row no-print" style={{ marginTop: "1rem" }}>
        <ButtonLink size="sm" icon={Download} href={api.downloadUrl(id, "ulg")} download>
          Converted flight.ulg
        </ButtonLink>
        <ButtonLink size="sm" variant="ghost" icon={Download} href={api.downloadUrl(id, "original")} download>
          Original CSV
        </ButtonLink>
      </div>
      <h3 className="subhead">Assumptions</h3>
      <ul className="bullets">
        {conv.assumptions.map((a) => (
          <li key={a}>{a}</li>
        ))}
      </ul>
      {conv.derived.length > 0 && (
        <>
          <h3 className="subhead">Derived by the converter</h3>
          <ul className="bullets">
            {conv.derived.map((d) => (
              <li key={d.record}>
                <span className="mono">{d.record}</span>: {d.how}. <span className="muted">{d.caveat}.</span>
              </li>
            ))}
          </ul>
        </>
      )}
      <h3 className="subhead">Column mapping</h3>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>CSV column</th>
              <th>ARJUNA field</th>
              <th>Conversion</th>
            </tr>
          </thead>
          <tbody>
            {conv.mapped.map((m, i) => (
              <tr key={i}>
                <td className="mono">{m.from}</td>
                <td className="mono small">{m.to.join(", ")}</td>
                <td className="small">{m.conversion || m.unit_in}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {conv.dropped.length > 0 && (
        <>
          <h3 className="subhead">Not converted</h3>
          <ul className="bullets">
            {conv.dropped.map((d) => (
              <li key={d.column}>
                <span className="mono">{d.column}</span>: {d.reason}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

function Labels({ s }: { s: ArjunaSummary }) {
  const labs = s.engineer_labels?.labels ?? [];
  const found = new Set(s.findings.map((f) => f.code));
  return (
    <div className="stack">
      <p className="small muted">
        From labels.json ({s.engineer_labels?.method ?? "engineer review"}). Shown for comparison only; never given to the AI.
      </p>
      {labs.length === 0 ? (
        <p>This folder is labelled healthy (no fault labels).</p>
      ) : (
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Code</th>
                <th>Severity</th>
                <th>When</th>
                <th>Note</th>
                <th>Found here?</th>
              </tr>
            </thead>
            <tbody>
              {labs.map((l, i) => (
                <tr key={i}>
                  <td>
                    <span className="code">{l.code}</span>
                  </td>
                  <td>{l.severity}</td>
                  <td className="mono small">{l.when}</td>
                  <td className="small">{l.note}</td>
                  <td>{found.has(l.code) ? <Pill tone="go">Yes</Pill> : <Pill tone="caution">No</Pill>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const ROLE: Record<string, { text: string; tone: string }> = {
  root_cause: { text: "Likely root cause", tone: "tone-warn" },
  consequence: { text: "Consequence", tone: "tone-caution" },
  independent: { text: "Independent", tone: "tone-neutral" },
};

function Review({ view }: { view: AnalysisView }) {
  const run = view.report!;
  const rep = run.report!;
  const titles = new Map((view.summary as ArjunaSummary).findings.map((f) => [f.id, f.title]));
  const p = run.provenance;
  return (
    <div>
      {run.status === "partial" && run.message && (
        <Alert tone="info">{run.message}</Alert>
      )}
      {rep.root_cause_chain.length > 0 && (
        <>
          <h3 className="subhead">What led to what</h3>
          <ol className="chain">
            {rep.root_cause_chain.map((c, i) => (
              <li key={c.finding_id} className={ROLE[c.role].tone}>
                <span className="chain-dot" aria-hidden="true">
                  {i + 1}
                </span>
                <div>
                  <div className="row" style={{ gap: "0.45rem" }}>
                    <span className={`pill ${ROLE[c.role].tone}`}>{ROLE[c.role].text}</span>
                    <span className="code">{c.finding_id}</span>
                    <span className="small" style={{ fontWeight: 600 }}>
                      {titles.get(c.finding_id)}
                    </span>
                  </div>
                  {c.link && <p className="small text-2" style={{ marginTop: "0.3rem" }}>{c.link}</p>}
                </div>
              </li>
            ))}
          </ol>
        </>
      )}
      {rep.additional_observations.length > 0 && (
        <>
          <h3 className="subhead">Found beyond the rules</h3>
          <ul className="plain stack" style={{ ["--gap" as string]: "0.7rem" }}>
            {rep.additional_observations.map((o, i) => (
              <li key={i}>
                <div className="row" style={{ gap: "0.45rem" }}>
                  <strong>{o.title}</strong>
                  {o.suggested_code && <span className="code">{o.suggested_code}</span>}
                </div>
                <p className="small text-2">{o.detail}</p>
                <p className="tiny muted mono">Evidence: {o.evidence_refs.join(", ")}</p>
              </li>
            ))}
          </ul>
        </>
      )}
      {rep.not_checked.length > 0 && (
        <>
          <h3 className="subhead">What could not be checked</h3>
          <ul className="bullets">
            {rep.not_checked.map((t, i) => (
              <li key={i}>{t}</li>
            ))}
          </ul>
        </>
      )}
      <h3 className="subhead">
        How it was worked out · {p.tool_calls ?? run.trace.length} tool calls · {p.model}
        {p.latency_ms ? ` · ${Math.round(p.latency_ms / 1000)} s` : ""}
      </h3>
      <p className="small muted" style={{ marginBottom: "0.6rem" }}>
        Every number in the review was computed by a tool and checked against its result before it was shown.
        {rep.rejected.length > 0 ? ` ${rep.rejected.length} statement(s) failed that check and were removed.` : ""}
      </p>
      <ol className="plain agent-trace">
        {run.trace.map((c) => (
          <li key={c.ref} className={c.ok ? "" : "err"}>
            {c.ref} {c.tool}({Object.entries(c.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})
            {c.error ? ` — ${c.error.slice(0, 120)}` : ""}
          </li>
        ))}
      </ol>
      {rep.rejected.length > 0 && (
        <>
          <h3 className="subhead">Removed statements</h3>
          <ul className="bullets">
            {rep.rejected.map((r, i) => (
              <li key={i}>
                <span className="mono">{r.where}</span> — {r.reason}: “{r.text}”
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

function Technical({ s }: { s: ArjunaSummary }) {
  return (
    <div>
      <h3 className="subhead">Integrity</h3>
      <p className="small text-2">{s.integrity.detail}</p>
      <h3 className="subhead">Compatibility</h3>
      <ul className="bullets">
        {[...s.compatibility.problems, ...s.compatibility.notes].map((n) => (
          <li key={n}>{n}</li>
        ))}
        <li>
          Frames ENU / FLU, profile {s.compatibility.log_profile}, schema {s.compatibility.schema_version}.
        </li>
      </ul>
      <h3 className="subhead">Topics</h3>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Topic</th>
              <th className="num">Rate</th>
              <th className="num">Samples</th>
              <th>Never recorded</th>
            </tr>
          </thead>
          <tbody>
            {(s.topics?.topics ?? []).map((t) => (
              <tr key={`${t.topic}-${t.instance}`}>
                <td className="mono">
                  {t.topic}
                  {t.instance ? `[${t.instance}]` : ""}
                </td>
                <td className="num">{t.rate_hz ?? "—"} Hz</td>
                <td className="num">{t.samples}</td>
                <td className="small muted">{t.unavailable_fields.join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!!s.topics?.missing_standard_topics.length && (
        <p className="small muted" style={{ marginTop: "0.6rem" }}>
          Not in this log: {s.topics.missing_standard_topics.join(", ")}.
        </p>
      )}
      <h3 className="subhead">Identity</h3>
      <pre className="trace">{JSON.stringify({ ...s.identity, sha256: s.file.sha256 }, null, 2)}</pre>
    </div>
  );
}
