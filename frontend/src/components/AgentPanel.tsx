"use client";

import type { AgentRun, AnalysisView } from "@/lib/api";

const ROLE: Record<string, string> = { root_cause: "Likely root cause", consequence: "Consequence", independent: "Independent" };

/** The agent's investigation: headline, summary, pilot answer, cause chain, observations and how it worked. */
export function AgentPanel({
  view,
  running,
  error,
  onRun,
  findingTitle,
}: {
  view: AnalysisView;
  running: boolean;
  error: string | null;
  onRun: () => void;
  findingTitle: (id: string) => string;
}) {
  const run: AgentRun | null = view.report;
  const rep = run?.report ?? null;
  const usable = !!rep && (run!.status === "ok" || run!.status === "partial");
  const failed = view.last_attempt ?? (run && !usable ? run : null);

  if (running) {
    return (
      <section className="agent-panel" aria-live="polite">
        <div className="agent-running">
          <span className="spinner" />
          <div>
            <strong>The agent is investigating this flight.</strong>
            <p className="muted small">
              It verifies every finding on the raw signals, looks for problems the rules missed and works out what
              happened first. This usually takes 20–90 seconds.
            </p>
          </div>
        </div>
      </section>
    );
  }

  if (!usable) {
    return (
      <section className="agent-panel">
        <h2>Agentic investigation</h2>
        {failed && failed.status !== "ok" && (
          <div className="alert alert-warn" role="status">
            <strong>The last investigation did not finish.</strong> {failed.message}
            {failed.failure?.detail ? <span className="small muted"> ({failed.failure.detail})</span> : null}
          </div>
        )}
        {error && <div className="alert alert-error">{error}</div>}
        <p>
          The code findings below are complete. The agent adds an investigation on top: it checks each finding against the
          raw signals, looks for what the rules missed, orders events into a cause chain and answers the pilot&apos;s
          report, citing the computed evidence for every claim.
        </p>
        <p className="small muted">{view.consent_text}</p>
        <button className="btn btn-primary btn-lg" onClick={onRun}>
          {failed ? "Try the investigation again" : "Run the agentic investigation"}
        </button>
      </section>
    );
  }

  const p = run!.provenance;
  return (
    <section className="agent-panel">
      <p className="kicker">Agentic investigation</p>
      {rep!.headline && <h2 className="agent-headline">{rep!.headline}</h2>}
      {rep!.summary && <p className="agent-summary">{rep!.summary}</p>}
      {run!.status === "partial" && run!.message && <p className="small muted">{run!.message}</p>}

      {rep!.pilot_report_answer && (
        <div className="callout">
          <p className="small muted" style={{ margin: 0 }}>
            Answer to the pilot&apos;s report
            {p.pilot_report ? ` (${p.pilot_report.symptom_label ?? p.pilot_report.symptom})` : ""}
          </p>
          <p style={{ margin: "0.3rem 0 0" }}>{rep!.pilot_report_answer}</p>
        </div>
      )}

      {rep!.root_cause_chain.length > 0 && (
        <>
          <h3>What led to what</h3>
          <ol className="chain">
            {rep!.root_cause_chain.map((c) => (
              <li key={c.finding_id} className={`chain-${c.role}`}>
                <span className={`role role-${c.role}`}>{ROLE[c.role]}</span>{" "}
                <a href={`#${c.finding_id}`} className="mono">{c.finding_id}</a> {findingTitle(c.finding_id)}
                {c.link && <div className="small">{c.link}</div>}
              </li>
            ))}
          </ol>
        </>
      )}

      {rep!.recommendations.length > 0 && (
        <>
          <h3>What to check</h3>
          <ul className="recs">
            {rep!.recommendations.map((r, i) => (
              <li key={i}>
                <strong>{r.action}</strong>
                {r.reason && <span className="small"> — {r.reason}</span>}
                {r.finding_ids.length > 0 && <span className="small muted mono"> ({r.finding_ids.join(", ")})</span>}
              </li>
            ))}
          </ul>
        </>
      )}

      {rep!.additional_observations.length > 0 && (
        <>
          <h3>Found by the agent beyond the rules</h3>
          <ul className="obs">
            {rep!.additional_observations.map((o, i) => (
              <li key={i}>
                <strong>{o.title}</strong>
                {o.suggested_code && <span className="code-chip">{o.suggested_code}</span>}
                <div className="small">{o.detail}</div>
                <div className="small muted mono">Evidence: {o.evidence_refs.join(", ")}</div>
              </li>
            ))}
          </ul>
        </>
      )}

      {rep!.not_checked.length > 0 && (
        <>
          <h3>What could not be checked</h3>
          <ul className="small">
            {rep!.not_checked.map((t, i) => (
              <li key={i}>{t}</li>
            ))}
          </ul>
        </>
      )}

      <details className="agent-how">
        <summary>
          How the agent investigated · {p.tool_calls ?? run!.trace.length} tool calls · {p.model}
          {p.latency_ms ? ` · ${(p.latency_ms / 1000).toFixed(0)} s` : ""}
        </summary>
        <p className="small muted">
          Every number in this report was computed by a tool and checked against its result before it was shown.
          {rep!.rejected.length > 0 ? ` ${rep!.rejected.length} statement(s) that failed that check were removed (listed below).` : ""}
        </p>
        <ol className="trace small mono">
          {run!.trace.map((c) => (
            <li key={c.ref} className={c.ok ? "" : "trace-err"}>
              {c.ref} {c.tool}({Object.entries(c.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})
              {c.error ? ` — ${c.error.slice(0, 120)}` : ""}
            </li>
          ))}
        </ol>
        {rep!.rejected.length > 0 && (
          <>
            <h4>Removed statements</h4>
            <ul className="small">
              {rep!.rejected.map((r, i) => (
                <li key={i}>
                  <span className="mono">{r.where}</span> — {r.reason}: “{r.text}”
                </li>
              ))}
            </ul>
          </>
        )}
        <button className="btn btn-quiet no-print" onClick={onRun}>
          Run the investigation again
        </button>
      </details>
    </section>
  );
}
