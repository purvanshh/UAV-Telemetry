"use client";

import { useEffect, useState } from "react";
import { SignalChart } from "@/components/SignalChart";
import { api, ChartPack, Finding, FindingReview } from "@/lib/api";

const ASSESS: Record<string, string> = {
  confirmed: "Confirmed by the agent",
  likely_false_alarm: "Likely false alarm",
  needs_review: "Needs review",
};

function v(x: number | null | undefined, unit: string | null | undefined) {
  if (x == null || !Number.isFinite(x)) return "—";
  const a = Math.abs(x);
  const s = a >= 100 ? x.toFixed(0) : Number(x.toPrecision(3)).toString();
  return unit ? `${s} ${unit}` : s;
}

export function FindingCard({ id, finding, review, codeLabel, open: initial }: {
  id: string;
  finding: Finding;
  review?: FindingReview;
  codeLabel?: string;
  open?: boolean;
}) {
  const [open, setOpen] = useState(!!initial);
  const [ctx, setCtx] = useState<ChartPack | null | undefined>(undefined);

  useEffect(() => {
    if (!open || ctx !== undefined) return;
    api.findingContext(id, finding.id).then(setCtx).catch(() => setCtx(null));
  }, [open, ctx, id, finding.id]);

  return (
    <article id={finding.id} className={`finding sev-${finding.severity}`}>
      <button type="button" className="finding-head" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span className={`sev sev-${finding.severity}`}>{finding.severity}</span>
        <span className="code-chip" title={codeLabel}>{finding.code}</span>
        <span className="finding-title">{finding.title}</span>
        <span className="finding-when mono">{finding.window === "whole_flight" ? "whole flight" : finding.when}</span>
        {review && review.reviewed_by_agent && (
          <span className={`assess assess-${review.assessment}`}>{ASSESS[review.assessment]}</span>
        )}
      </button>
      {open && (
        <div className="finding-body">
          {review?.reviewed_by_agent && (review.explanation || review.suggested_cause || review.check_next) && (
            <div className="review">
              {review.explanation && <p>{review.explanation}</p>}
              {review.suggested_cause && (
                <p>
                  <span className="label-tag">Suggested cause</span> {review.suggested_cause}
                </p>
              )}
              {review.check_next && (
                <p>
                  <span className="label-tag">Check next</span> {review.check_next}
                </p>
              )}
              {review.evidence_refs.length > 0 && (
                <p className="small muted mono">Evidence: {review.evidence_refs.join(", ")}</p>
              )}
            </div>
          )}
          {review && !review.reviewed_by_agent && (
            <p className="small muted">The agent did not review this finding; it stands as the code found it.</p>
          )}
          <p className="small">
            <span className="muted">Found by</span> <span className="mono">{finding.detector}</span>
            <span className="muted"> · confidence</span> {finding.confidence}
            {codeLabel ? <span className="muted"> · {codeLabel}</span> : null}
          </p>
          {finding.basis && <p className="small muted">{finding.basis}</p>}
          <div className="table-wrap">
            <table className="data stat-table">
              <thead>
                <tr>
                  <th>Signal</th>
                  <th>Statistic</th>
                  <th>Value</th>
                  <th>At</th>
                </tr>
              </thead>
              <tbody>
                {finding.evidence.map((e, i) => (
                  <tr key={i}>
                    <td className="mono">
                      {e.topic}
                      {e.instance ? `[${e.instance}]` : ""}.{e.field}
                    </td>
                    <td>{e.stat}</td>
                    <td className="num">{v(e.value, e.unit)}</td>
                    <td className="mono">{e.at ?? ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {ctx === undefined && <p className="muted small">Loading signals…</p>}
          {ctx && ctx.charts.length > 0 && (
            <div className="sig-grid">
              {ctx.charts.map((c) => (
                <SignalChart key={c.title + c.unit} chart={c} window={ctx.window} highlight={ctx.highlight} />
              ))}
            </div>
          )}
        </div>
      )}
    </article>
  );
}
