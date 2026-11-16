"use client";

import type { AnalysisView, PilotView } from "@/lib/api";

const ICON: Record<string, string> = { critical: "⛔", warning: "⚠️", clear: "✅", not_a_flight: "ℹ️", unreadable: "❓" };

/** What a pilot needs: one verdict, the problems in plain words, and what to do before the next flight. */
export function PilotSummary({
  view,
  pilot,
  running,
  error,
  onRun,
}: {
  view: AnalysisView;
  pilot: PilotView;
  running: boolean;
  error: string | null;
  onRun: () => void;
}) {
  const run = view.report;
  const rep = run && (run.status === "ok" || run.status === "partial") ? run.report : null;
  // Serious items first; then the detailed review's advice, or the automatic advice when there is no review.
  const recs = rep?.recommendations.map((r) => r.action) ?? [];
  const rest = recs.length ? recs : pilot.checklist.filter((c) => !pilot.critical_checks.includes(c));
  const checklist = [...pilot.critical_checks, ...rest.filter((c) => !pilot.critical_checks.includes(c))].slice(0, 5);

  return (
    <section className="pilot">
      <div className={`verdict-banner v-${pilot.status}`} role="status">
        <span className="verdict-icon" aria-hidden="true">
          {ICON[pilot.status]}
        </span>
        <div>
          <strong>{pilot.status_text}</strong>
          <p>{pilot.status_detail}</p>
        </div>
      </div>
      {pilot.integrity_warning && <div className="alert alert-error">{pilot.integrity_warning}</div>}

      <div className="pilot-what">
        <h2>What happened</h2>
        {rep?.headline ? (
          <>
            <p className="pilot-headline">{rep.headline}</p>
            {rep.summary && <p>{rep.summary}</p>}
          </>
        ) : running ? (
          <p className="agent-running">
            <span className="spinner" /> Looking into the flight in detail… this takes about a minute.
          </p>
        ) : (
          <p className="muted">
            {pilot.problems.length
              ? "The problems found by the automatic checks are listed below."
              : "The automatic checks found nothing wrong."}{" "}
            {view.report?.status && view.report.status !== "ok" && view.report.status !== "partial"
              ? "The detailed AI review did not finish. "
              : ""}
            <button className="linklike no-print" onClick={onRun} title={view.consent_text}>
              Get a detailed explanation
            </button>
            <span className="small muted no-print"> (an AI reviews the computed results; the raw log is not sent)</span>
          </p>
        )}
        {error && <p className="small t-crit">{error}</p>}
        {rep?.pilot_report_answer && (
          <div className="callout">
            <p className="small muted" style={{ margin: 0 }}>
              About what you saw
            </p>
            <p style={{ margin: "0.3rem 0 0" }}>{rep.pilot_report_answer}</p>
          </div>
        )}
      </div>

      {pilot.problems.length > 0 && (
        <>
          <h2>Problems found ({pilot.problems.length})</h2>
          <ul className="problems">
            {pilot.problems.map((p) => (
              <li key={p.code} className={`problem p-${p.severity}${p.likely_false_alarm ? " p-doubt" : ""}`}>
                <div className="problem-head">
                  <span className={`sev sev-${p.severity}`}>{p.severity === "critical" ? "Serious" : "Check"}</span>
                  <strong>{p.name}</strong>
                </div>
                <p className="problem-line">{p.headline}</p>
                {p.also.length > 0 && (
                  <ul className="small problem-also">
                    {p.also.map((a) => (
                      <li key={a}>{a}</li>
                    ))}
                  </ul>
                )}
                <p className="small muted">{p.when ? `When: ${p.when}. ` : ""}{p.meaning}</p>
                {p.likely_false_alarm && (
                  <p className="small">The detailed review thinks this may be a false alarm; see the engineering details.</p>
                )}
                <a
                  className="small no-print"
                  href={`#${p.finding_id}`}
                  onClick={() => {
                    const d = document.getElementById("engineering") as HTMLDetailsElement | null;
                    if (d) d.open = true;
                  }}
                >
                  Show the data
                </a>
              </li>
            ))}
          </ul>
        </>
      )}

      {checklist.length > 0 && (
        <>
          <h2>Before the next flight</h2>
          <ol className="checklist-plain">
            {checklist.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ol>
        </>
      )}

      {pilot.not_checked.length > 0 && (
        <p className="small muted">
          Not recorded on this flight, so not checked: {pilot.not_checked.join(", ")}.
        </p>
      )}
    </section>
  );
}
