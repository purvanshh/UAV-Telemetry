"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { AgentPanel } from "@/components/AgentPanel";
import { AppNav } from "@/components/AppNav";
import { FindingCard } from "@/components/FindingCard";
import { api, ApiError, AnalysisView, ArjunaSummary } from "@/lib/api";
import { duration } from "@/lib/format";

const INTEGRITY_TONE: Record<string, string> = {
  chain_verified: "s-ok",
  verified_up_to: "s-ok",
  modified: "s-error",
  no_integrity_data: "s-caution",
};

export default function AnalysisPage() {
  return (
    <Suspense fallback={null}>
      <Analysis />
    </Suspense>
  );
}

function Analysis() {
  const id = String(useParams().id || "");
  const params = useSearchParams();
  const router = useRouter();
  const [view, setView] = useState<AnalysisView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [agentError, setAgentError] = useState<string | null>(null);
  const started = useRef(false);

  useEffect(() => {
    api.get(id).then(setView).catch((e) => setError(e instanceof Error ? e.message : "Failed to load"));
  }, [id]);

  const runAgent = useCallback(async () => {
    setRunning(true);
    setAgentError(null);
    try {
      setView(await api.runAgent(id));
    } catch (e) {
      setAgentError(e instanceof ApiError || e instanceof Error ? e.message : "The investigation failed");
    } finally {
      setRunning(false);
    }
  }, [id]);

  // Consent given on the upload page: start the investigation once, then drop the flag from the URL.
  useEffect(() => {
    if (!view || started.current || params.get("agent") !== "1") return;
    started.current = true;
    router.replace(`/analysis/${encodeURIComponent(id)}`);
    if (!view.report && !view.summary.refused) void runAgent();
  }, [view, params, router, id, runAgent]);

  return (
    <>
      <AppNav />
      <main>
        <Link className="crumb no-print" href="/">
          ← All flights
        </Link>
        {error && <div className="alert alert-error">{error}</div>}
        {!view && !error && <p className="muted">Loading…</p>}
        {view && <UlogView view={view} running={running} agentError={agentError} onRun={runAgent} />}
      </main>
    </>
  );
}

function Header({ view, title, sub }: { view: AnalysisView; title: string; sub: string }) {
  return (
    <div className="page-head row between">
      <div>
        <p className="brief-page-kicker">Agentic Analysis</p>
        <h1>{title}</h1>
        <p className="sub">{sub}</p>
      </div>
      <div className="row no-print">
        <button className="btn" onClick={() => window.print()}>
          Print or save as PDF
        </button>
        <a className="btn btn-quiet" href={api.downloadUrl(view.analysis_id, "ulg")}>
          Download .ulg
        </a>
        <a className="btn btn-quiet" href={api.exportUrl(view.analysis_id)}>
          Export JSON
        </a>
      </div>
    </div>
  );
}

function UlogView({ view, running, agentError, onRun }: {
  view: AnalysisView;
  running: boolean;
  agentError: string | null;
  onRun: () => void;
}) {
  const s = view.summary as ArjunaSummary;
  const id = view.analysis_id;
  const reviews = useMemo(
    () => new Map((view.report?.report?.finding_reviews ?? []).map((r) => [r.finding_id, r])),
    [view.report]
  );
  const titles = useMemo(() => new Map(s.findings.map((f) => [f.id, f.title])), [s.findings]);
  const ident = s.identity;
  const tl = s.timeline;
  const conv = s.conversion;
  const title = conv ? conv.source_file : String(ident.flight_id ?? s.file.name);
  const sub = conv
    ? `Converted from CSV (${conv.format.label}) to ARJUNA log ${ident.flight_id}`
    : [ident.vehicle_id, ident.fw_version, ident.log_profile && `${ident.log_profile} profile`, s.file.name]
        .filter(Boolean)
        .join(" · ");

  if (s.refused) {
    return (
      <>
        <Header view={view} title={title} sub={sub} />
        <div className="alert alert-error" role="alert">
          <strong>This log was not analysed.</strong> The ARJUNA standard requires the app to stop rather than guess when
          a log&apos;s schema, frames or profile are unknown:
          <ul>
            {s.compatibility.problems.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </div>
      </>
    );
  }

  const v = s.verdict!;
  return (
    <>
      <Header view={view} title={title} sub={sub} />
      {s.integrity.result === "modified" && (
        <div className="alert alert-error" role="alert">
          <strong>Modified log.</strong> {s.integrity.detail} Every finding below comes from a modified log.
        </div>
      )}
      <div className="badges">
        <span className={`chip ${INTEGRITY_TONE[s.integrity.result] ?? ""}`} title={s.integrity.detail}>
          {s.integrity.label ?? s.integrity.result}
        </span>
        <span className="chip">schema {s.compatibility.schema_version}</span>
        {conv ? (
          <span className="chip s-caution">converted from CSV · {conv.format.id}</span>
        ) : (
          s.integrity.converted_log && <span className="chip s-caution">converted from another autopilot</span>
        )}
        {s.classification && s.classification.kind !== "flight" && (
          <span className="chip s-caution">{s.classification.kind.replace("_", " ")}: {s.classification.reason}</span>
        )}
      </div>

      <div className="facts">
        <div className="fact">
          <span className="muted small">Result</span>
          <strong className={v.worst === "critical" ? "t-crit" : v.worst === "warning" ? "t-warn" : "t-ok"}>
            {v.worst === "critical" ? "Critical findings" : v.worst === "warning" ? "Warnings" : "Nothing flagged"}
          </strong>
          <span className="small">
            {v.critical} critical · {v.warning} warning
          </span>
        </div>
        <div className="fact">
          <span className="muted small">Log length</span>
          <strong>{duration(tl?.duration_s)}</strong>
        </div>
        <div className="fact">
          <span className="muted small">In the air</span>
          <strong>{duration(tl?.airborne_s)}</strong>
          {tl?.air_intervals?.[0] && <span className="small mono">{tl.air_intervals.map((a) => a.when).join(", ")}</span>}
        </div>
        <div className="fact">
          <span className="muted small">Modes</span>
          <strong className="small">{Array.from(new Set((tl?.modes ?? []).filter((m) => m.value !== "unknown").map((m) => m.value.replace(/_/g, " ")))).join(", ") || "not logged"}</strong>
        </div>
      </div>

      <AgentPanel view={view} running={running} error={agentError} onRun={onRun} findingTitle={(fid) => titles.get(fid) ?? ""} />

      <section className="section">
        <h2>Findings ({s.findings.length})</h2>
        <p className="small muted">
          Found by code: rule checks with the approved ARJUNA thresholds and the physics model. The agent reviews them but
          cannot remove them.
        </p>
        {s.findings.length === 0 && (
          <p>
            No detector flagged anything on the checks that could run. That is not a statement that the flight or the
            aircraft was safe.
          </p>
        )}
        {s.findings.map((f, i) => (
          <FindingCard key={f.id} id={id} finding={f} review={reviews.get(f.id)} codeLabel={s.fault_codes?.[f.code]} open={i < 2} />
        ))}
      </section>

      {conv && <ConversionPanel id={id} conv={conv} />}
      <Coverage s={s} />
      <Physics s={s} />
      <Timeline s={s} />
      {s.engineer_labels && <Labels s={s} />}
      <Technical s={s} />
    </>
  );
}

function ConversionPanel({ id, conv }: { id: string; conv: NonNullable<ArjunaSummary["conversion"]> }) {
  return (
    <section className="section">
      <details open={conv.format.id === "generic"}>
        <summary>
          <h2 style={{ display: "inline" }}>Converted from CSV</h2>{" "}
          <span className="small muted">
            {conv.format.label} · {conv.mapped.length} columns mapped · {conv.derived.length} records derived ·{" "}
            {conv.dropped.length} dropped · {conv.assumptions.length} assumptions
          </span>
        </summary>
        <p className="small">
          The CSV was converted into a standard ARJUNA log first ({conv.converter_version}); everything on this page was
          computed from that log. Time: <span className="mono">{conv.time.column}</span> in {conv.time.unit} ({conv.time.how}),{" "}
          {conv.time.sample_rate_hz ?? "—"} Hz, {conv.rows_written} of {conv.rows_in} rows
          {conv.duplicate_timestamps_dropped ? `, ${conv.duplicate_timestamps_dropped} duplicate timestamps dropped` : ""}
          {conv.rows_without_valid_time ? `, ${conv.rows_without_valid_time} rows without a valid time dropped` : ""}.
        </p>
        <p className="row no-print">
          <a className="btn btn-quiet" href={api.downloadUrl(id, "ulg")}>Download the converted flight.ulg</a>
          <a className="btn btn-quiet" href={api.downloadUrl(id, "original")}>Download the original CSV</a>
        </p>
        <h3>Assumptions</h3>
        <ul className="small">
          {conv.assumptions.map((a) => (
            <li key={a}>{a}</li>
          ))}
        </ul>
        {conv.derived.length > 0 && (
          <>
            <h3>Derived by the converter</h3>
            <ul className="small">
              {conv.derived.map((d) => (
                <li key={d.record}>
                  <span className="mono">{d.record}</span>: {d.how}. <span className="muted">{d.caveat}.</span>
                </li>
              ))}
            </ul>
          </>
        )}
        <h3>Column mapping</h3>
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
            <h3>Not converted</h3>
            <ul className="small">
              {conv.dropped.map((d) => (
                <li key={d.column}>
                  <span className="mono">{d.column}</span>: {d.reason}
                </li>
              ))}
            </ul>
          </>
        )}
      </details>
    </section>
  );
}

function Coverage({ s }: { s: ArjunaSummary }) {
  if (!s.coverage) return null;
  return (
    <section className="section">
      <h2>Fault codes checked</h2>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Code</th>
              <th>Meaning</th>
              <th>Status</th>
              <th>Checks run / why not</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(s.coverage).map(([code, c]) => {
              const found = s.findings.filter((f) => f.code === code).length;
              return (
                <tr key={code}>
                  <td className="mono">{code}</td>
                  <td className="small">{s.fault_codes?.[code]}</td>
                  <td>
                    {found ? (
                      <span className="chip s-attention">{found} finding(s)</span>
                    ) : c.status === "evaluated" ? (
                      <span className="chip s-ok">checked, clear</span>
                    ) : c.status === "partial" ? (
                      <span className="chip s-caution">events only</span>
                    ) : (
                      <span className="chip">not checkable</span>
                    )}
                  </td>
                  <td className="small">
                    {c.checks.join(", ")}
                    {c.status !== "evaluated" && c.reasons.length > 0 && (
                      <span className="muted">{c.checks.length ? " — " : ""}{c.reasons.join("; ")}</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Physics({ s }: { s: ArjunaSummary }) {
  const hb = s.physics?.hover_balance;
  const tr = s.physics?.thrust_residual;
  const dy = s.physics?.dynamics;
  if (!hb) return null;
  return (
    <section className="section">
      <h2>Physics checks</h2>
      <h3>Hover motor balance</h3>
      {hb.status === "computed" ? (
        <>
          <p>{hb.interpretation}</p>
          <div className="table-wrap">
            <table className="data stat-table">
              <thead>
                <tr>
                  <th>Motor</th>
                  <th>Mean command in steady hover</th>
                  <th>vs. motor average</th>
                </tr>
              </thead>
              <tbody>
                {(hb.motors ?? []).map((m, i) => (
                  <tr key={m} className={m === hb.worst_motor ? "row-strong" : ""}>
                    <td>{m}</td>
                    <td className="num">{hb.mean_cmd?.[i]?.toFixed(3)}</td>
                    <td className="num">
                      {(hb.deviation_pct?.[i] ?? 0) > 0 ? "+" : ""}
                      {hb.deviation_pct?.[i]}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="small muted">
            From {hb.steady_hover_s} s of steady hover. In hover the net torque is zero, so a motor that needs a different
            command from the others shows its propulsion path or the load distribution. Approved limit: 12%.
          </p>
        </>
      ) : (
        <p className="small muted">Not computed: {hb.reason}</p>
      )}
      {tr && (
        <>
          <h3>Vertical thrust residual</h3>
          {tr.status === "computed" ? (
            <>
              <p className="small">
                Model <span className="mono">{tr.model}</span>, {tr.voltage_compensation}. Residual spread{" "}
                {tr.residual_robust_spread_mps2} m/s².{" "}
                {tr.sustained_departures?.length
                  ? `${tr.sustained_departures.length} sustained departure(s):`
                  : "No sustained departures."}
              </p>
              {!!tr.sustained_departures?.length && (
                <ul className="small">
                  {tr.sustained_departures.slice(0, 6).map((d) => (
                    <li key={d.when}>
                      <span className="mono">{d.when}</span> {d.direction} (median {d.median_residual_mps2} m/s²)
                    </li>
                  ))}
                </ul>
              )}
              <p className="small muted">{tr.threshold_note}</p>
            </>
          ) : (
            <p className="small muted">Not computed: {tr.reason}</p>
          )}
        </>
      )}
      {dy && (
        <>
          <h3>Per-motor effectiveness (η)</h3>
          {dy.status === "computed" ? (
            <p className="small">
              {dy.motors?.map((m, i) => `${m}: η ${dy.eta?.[i]} (${dy.relative_to_others?.[i]} of the others)`).join(" · ")}
              {dy.flag_motor ? ` — ${dy.flag_motor} flagged.` : " — no motor flagged."}
            </p>
          ) : (
            <p className="small muted">Not computed: {dy.reason}</p>
          )}
        </>
      )}
    </section>
  );
}

function Timeline({ s }: { s: ArjunaSummary }) {
  const tl = s.timeline;
  if (!tl) return null;
  return (
    <section className="section">
      <details>
        <summary>
          <h2 style={{ display: "inline" }}>Timeline</h2>{" "}
          <span className="small muted">
            {tl.modes?.length ?? 0} mode periods · {tl.events?.length ?? 0} events · {tl.important_messages?.length ?? 0} warnings
            or worse · {tl.dropouts?.length ?? 0} dropouts
          </span>
        </summary>
        <h3>Modes</h3>
        <ul className="small mono">
          {(tl.modes ?? []).map((m, i) => (
            <li key={i}>
              {m.when} {m.value}
            </li>
          ))}
        </ul>
        <h3>Events</h3>
        <ul className="small">
          {(tl.events ?? []).map((e, i) => (
            <li key={i}>
              <span className="mono">{e.at}</span> {e.event.replace(/_/g, " ")}
              {e.severity !== "info" ? ` (${e.severity})` : ""}
              {e.mode_after ? ` → ${e.mode_after}` : ""}
            </li>
          ))}
        </ul>
        <h3>Messages (warning and above)</h3>
        <ul className="small">
          {(tl.important_messages ?? []).map((m, i) => (
            <li key={i}>
              <span className="mono">{m.at}</span> [{m.level}] {m.text}
            </li>
          ))}
        </ul>
      </details>
    </section>
  );
}

function Labels({ s }: { s: ArjunaSummary }) {
  const labs = s.engineer_labels!.labels;
  const found = new Set(s.findings.map((f) => f.code));
  return (
    <section className="section">
      <h2>Engineer labels in this flight folder</h2>
      <p className="small muted">
        From labels.json ({s.engineer_labels!.method ?? "engineer review"}). Shown for comparison only; never given to the
        agent.
      </p>
      {labs.length === 0 ? (
        <p className="small">The folder is labelled healthy (no fault labels).</p>
      ) : (
        <ul className="small">
          {labs.map((l, i) => (
            <li key={i}>
              <span className="code-chip">{l.code}</span> {l.severity} · <span className="mono">{l.when}</span> {l.note}{" "}
              {found.has(l.code) ? <span className="chip s-ok">also found by the code</span> : <span className="chip s-attention">not found</span>}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Technical({ s }: { s: ArjunaSummary }) {
  return (
    <section className="section">
      <details>
        <summary>
          <h2 style={{ display: "inline" }}>Technical details</h2>
        </summary>
        <h3>Integrity</h3>
        <p className="small">{s.integrity.detail}</p>
        <h3>Compatibility</h3>
        <ul className="small">
          {[...s.compatibility.problems, ...s.compatibility.notes].map((n) => (
            <li key={n}>{n}</li>
          ))}
          <li>
            Frames ENU / FLU, profile {s.compatibility.log_profile}, schema {s.compatibility.schema_version}.
          </li>
        </ul>
        <h3>Topics</h3>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Topic</th>
                <th>Rate</th>
                <th>Samples</th>
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
                  <td className="small">{t.unavailable_fields.join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!!s.topics?.missing_standard_topics.length && (
          <p className="small">Not in this log: {s.topics.missing_standard_topics.join(", ")}.</p>
        )}
        <h3>Identity</h3>
        <pre className="trace">{JSON.stringify({ ...s.identity, sha256: s.file.sha256 }, null, 2)}</pre>
      </details>
    </section>
  );
}
