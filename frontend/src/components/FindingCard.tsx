"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronRight, ScanSearch } from "lucide-react";
import { SignalChart } from "@/components/SignalChart";
import { Pill, Skeleton } from "@/components/ui";
import { api, ChartPack, Finding, FindingReview } from "@/lib/api";
import { SEVERITY_TONE } from "@/lib/status";

const ASSESS: Record<string, { text: string; tone: "go" | "caution" | "neutral" }> = {
  confirmed: { text: "Confirmed by review", tone: "go" },
  likely_false_alarm: { text: "Likely false alarm", tone: "caution" },
  needs_review: { text: "Needs review", tone: "neutral" },
};

function value(x: number | null | undefined, unit: string | null | undefined) {
  if (x == null || !Number.isFinite(x)) return "—";
  const a = Math.abs(x);
  const s = a >= 100 ? x.toFixed(0) : Number(x.toPrecision(3)).toString();
  return unit ? `${s} ${unit}` : s;
}

/** One detector finding with its evidence and signals. Charts load the first time the card is opened. */
export function FindingCard({ id, finding, review, codeLabel, defaultOpen, target }: {
  id: string;
  finding: Finding;
  review?: FindingReview;
  codeLabel?: string;
  defaultOpen?: boolean;
  target?: boolean;
}) {
  const [open, setOpen] = useState(!!defaultOpen || !!target);
  const [ctx, setCtx] = useState<ChartPack | null | undefined>(undefined);
  const el = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!target) return;
    setOpen(true);
    const t = window.setTimeout(() => el.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 120);
    return () => window.clearTimeout(t);
  }, [target]);

  useEffect(() => {
    if (!open || ctx !== undefined) return;
    api.findingContext(id, finding.id).then(setCtx).catch(() => setCtx(null));
  }, [open, ctx, id, finding.id]);

  const tone = SEVERITY_TONE[finding.severity] ?? "neutral";
  return (
    <article ref={el} id={finding.id} className={`finding${target ? " is-target" : ""}`} style={{ scrollMarginTop: 80 }}>
      <button type="button" className="finding-head" aria-expanded={open} onClick={() => setOpen(!open)}>
        <ChevronRight className="chev" size={16} aria-hidden="true" />
        <Pill tone={tone}>{finding.severity === "critical" ? "Serious" : "Check"}</Pill>
        <span className="finding-title">{finding.title}</span>
        <span className="finding-meta">
          {review?.reviewed_by_agent && <Pill tone={ASSESS[review.assessment].tone}>{ASSESS[review.assessment].text}</Pill>}
          <span className="code" title={codeLabel}>
            {finding.code}
          </span>
          <span className="tiny muted mono">{finding.window === "whole_flight" ? "whole flight" : finding.when}</span>
        </span>
      </button>
      <div className={`disclosure-body${open ? " is-open" : ""}`} inert={!open}>
        <div>
          <div className="finding-body">
            {review?.reviewed_by_agent && (review.explanation || review.suggested_cause || review.check_next) && (
              <div className="review">
                <div className="eyebrow row" style={{ gap: "0.35rem", color: "var(--accent)" }}>
                  <ScanSearch size={13} aria-hidden="true" /> Detailed review
                </div>
                {review.explanation && <p>{review.explanation}</p>}
                {review.suggested_cause && (
                  <p>
                    <strong>Suggested cause:</strong> {review.suggested_cause}
                  </p>
                )}
                {review.check_next && (
                  <p>
                    <strong>Check next:</strong> {review.check_next}
                  </p>
                )}
                {review.evidence_refs.length > 0 && <p className="tiny muted mono">Evidence: {review.evidence_refs.join(", ")}</p>}
              </div>
            )}
            <p className="small text-2">
              {finding.basis} <span className="muted">Found by {finding.detector}, confidence {finding.confidence}.</span>
            </p>
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>Signal</th>
                    <th>Measure</th>
                    <th className="num">Value</th>
                    <th className="num">At</th>
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
                      <td className="num">{value(e.value, e.unit)}</td>
                      <td className="num">{e.at ?? ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {open && ctx === undefined && (
              <div className="sig-grid" aria-busy="true" aria-label="Loading signals">
                <Skeleton h={190} r={10} />
                <Skeleton h={190} r={10} />
              </div>
            )}
            {ctx && ctx.charts.length > 0 && (
              <div className="sig-grid">
                {ctx.charts.map((c, i) => (
                  <SignalChart key={`${c.title}-${c.unit}-${i}`} chart={c} window={ctx.window} highlight={ctx.highlight} />
                ))}
              </div>
            )}
            {ctx === null && <p className="small muted">The signals for this finding could not be loaded.</p>}
          </div>
        </div>
      </div>
    </article>
  );
}
