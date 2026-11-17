"use client";

import { useEffect, useMemo, useState } from "react";
import { BarChart3, Check, CircleMinus, Clock, ListChecks, ScanSearch } from "lucide-react";
import { DroneProcessing } from "@/components/drone/DroneProcessing";
import { Button, Card, Pill } from "@/components/ui";
import type { AgentProgress, AnalysisView, PilotProblem, PilotView } from "@/lib/api";
import { stepMessage } from "@/lib/agentSteps";
import { useLinger } from "@/lib/useLinger";
import { SEVERITY_TONE, SEVERITY_WORD, TONE_ICON, toneClass } from "@/lib/status";

/* ------------------------------------------------------------------------------------------------ what happened */

export function StoryCard({ view, pilot, running, progress, error, onRun }: {
  view: AnalysisView;
  pilot: PilotView;
  running: boolean;
  progress: AgentProgress | null;
  error: string | null;
  onRun: () => void;
}) {
  const run = view.report;
  const rep = run && (run.status === "ok" || run.status === "partial") ? run.report : null;
  const failed = !running && run && !rep ? run : view.last_attempt ?? null;
  // after a successful review the drone climbs away before the explanation appears
  const leaving = useLinger(running, 650) && !running && !!rep?.headline && !view.last_attempt;

  return (
    <Card title="What happened" icon={ScanSearch}>
      {running || leaving ? (
        <DroneProcessing
          leaving={leaving}
          title={leaving ? "Review complete." : stepMessage(progress)}
          detail={
            progress?.running
              ? `${progress.calls ?? 0} ${progress.calls === 1 ? "check" : "checks"} on the telemetry so far · ${Math.round(progress.elapsed_s ?? 0)} s. The automatic checks below are already complete.`
              : "The automatic checks below are already complete."
          }
        />
      ) : rep?.headline ? (
        <div className="fade">
          <p className="story-headline">{rep.headline}</p>
          {rep.summary && <p className="story-body">{rep.summary}</p>}
          {rep.pilot_report_answer && (
            <div className="quote">
              <div className="eyebrow">About what you noticed</div>
              <p>{rep.pilot_report_answer}</p>
            </div>
          )}
          <div className="ai-note">
            <ScanSearch size={13} aria-hidden="true" />
            Written by an AI from the computed results ({run!.provenance.model}). Every number was checked against the data.
          </div>
        </div>
      ) : (
        <div className="stack">
          <p className="text-2">
            {pilot.problems.length
              ? "The automatic checks found the problems listed below."
              : pilot.status === "clear"
                ? "The automatic checks found nothing wrong on this flight."
                : pilot.status_detail}
          </p>
          {failed && failed.status !== "ok" && failed.status !== "partial" && (
            <p className="small muted">The last detailed review did not finish: {failed.message}</p>
          )}
          {error && <p className="small" style={{ color: "var(--warn)" }}>{error}</p>}
          <div className="row no-print">
            <Button variant="secondary" icon={ScanSearch} onClick={onRun} title={view.consent_text}>
              {failed ? "Try the detailed review again" : "Explain this flight in detail"}
            </Button>
            <span className="tiny muted">About a minute. The raw log is not sent.</span>
          </div>
        </div>
      )}
    </Card>
  );
}

/* ------------------------------------------------------------------------------------------------ problems */

export function ProblemList({ problems, onShowData }: { problems: PilotProblem[]; onShowData: (findingId: string) => void }) {
  if (!problems.length) return null;
  return (
    <section aria-labelledby="problems-title">
      <h2 className="section-title" id="problems-title" style={{ marginTop: 0 }}>
        Problems found <span className="count">{problems.length}</span>
      </h2>
      <ul className="plain stack" style={{ ["--gap" as string]: "0.65rem" }}>
        {problems.map((p, i) => {
          const tone = SEVERITY_TONE[p.severity] ?? "neutral";
          const Icon = TONE_ICON[tone];
          return (
            <li key={p.code} className={`problem ${toneClass(tone)} rise${p.likely_false_alarm ? " is-doubt" : ""}`} style={{ ["--i" as string]: i }}>
              <span className="problem-icon" aria-hidden="true">
                <Icon />
              </span>
              <div className="problem-top">
                <Pill tone={tone}>{SEVERITY_WORD[p.severity]}</Pill>
                <span className="problem-name">{p.name}</span>
              </div>
              <p className="problem-line">{p.headline}</p>
              <div>
                {p.also.length > 0 && (
                  <ul className="problem-also">
                    {p.also.map((a) => (
                      <li key={a}>{a}</li>
                    ))}
                  </ul>
                )}
                <div className="problem-meta">
                  {p.when && (
                    <span className="when">
                      <Clock size={13} aria-hidden="true" /> {p.when}
                    </span>
                  )}
                  <button type="button" className="link-btn no-print" onClick={() => onShowData(p.finding_id)}>
                    <BarChart3 size={14} aria-hidden="true" /> Show the data
                  </button>
                </div>
                <p className="problem-why">{p.meaning}</p>
                {p.likely_false_alarm && (
                  <p className="problem-why">The detailed review thinks this may be a false alarm.</p>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

/* ------------------------------------------------------------------------------------------------ checklist */

/** Things to do before the next flight. Ticks are remembered on this device only. */
export function Checklist({ storageKey, items, critical, title = "Before the next flight" }: {
  storageKey: string;
  items: string[];
  critical: string[];
  title?: string;
}) {
  const [done, setDone] = useState<Record<string, boolean>>({});
  useEffect(() => {
    try {
      setDone(JSON.parse(localStorage.getItem(storageKey) || "{}"));
    } catch {
      setDone({});
    }
  }, [storageKey]);
  const toggle = (item: string) =>
    setDone((d) => {
      const next = { ...d, [item]: !d[item] };
      try {
        localStorage.setItem(storageKey, JSON.stringify(next));
      } catch {
        /* storage unavailable: ticks last for this visit */
      }
      return next;
    });
  const n = useMemo(() => items.filter((i) => done[i]).length, [items, done]);
  if (!items.length) return null;
  const all = n === items.length;
  return (
    <Card title={title} icon={ListChecks} actions={<Pill tone={all ? "go" : "neutral"} icon={all ? true : undefined}>{n} of {items.length} done</Pill>}>
      <div className="checklist">
        {items.map((item) => (
          <label key={item} className={`check-item${critical.includes(item) ? " is-critical" : ""}`}>
            <input type="checkbox" checked={!!done[item]} onChange={() => toggle(item)} />
            <span className="check-box" aria-hidden="true">
              <Check size={14} strokeWidth={3} />
            </span>
            <span className="check-text">{item}</span>
          </label>
        ))}
      </div>
      <div className="check-progress" style={{ marginTop: "0.75rem" }}>
        <div className="progress">
          <span style={{ width: `${(n / items.length) * 100}%`, background: all ? "var(--go)" : undefined }} />
        </div>
        <span className="no-print">Ticks are saved on this device</span>
      </div>
    </Card>
  );
}

export function NotChecked({ items }: { items: string[] }) {
  if (!items.length) return null;
  return (
    <p className="small muted row" style={{ gap: "0.4rem" }}>
      <CircleMinus size={15} aria-hidden="true" />
      Not recorded on this flight, so not checked: {items.join(", ")}.
    </p>
  );
}

/** Serious items first; then the detailed review's advice, or the automatic advice when there is no review. */
export function checklistItems(view: AnalysisView, pilot: PilotView): string[] {
  const run = view.report;
  const recs = run && (run.status === "ok" || run.status === "partial") ? run.report?.recommendations.map((r) => r.action) ?? [] : [];
  const rest = recs.length ? recs : pilot.checklist.filter((c) => !pilot.critical_checks.includes(c));
  return [...pilot.critical_checks, ...rest.filter((c) => !pilot.critical_checks.includes(c))].slice(0, 6);
}
