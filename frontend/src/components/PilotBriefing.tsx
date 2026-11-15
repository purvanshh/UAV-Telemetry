import type { ReactNode } from "react";
import type { PilotEvent, PilotReport, PilotReviewLabel } from "@/lib/api";

const BADGE_CLASS: Record<PilotReviewLabel, string> = {
  "Check this first": "review-first",
  "Worth checking": "review",
  "For reference": "review-minor",
};

/** Analyses stored before the wording change still carry the old labels. */
const LEGACY_LABELS: Record<string, PilotReviewLabel> = {
  "Review recommended": "Check this first",
  "No notable change detected": "For reference",
};

export function ReviewBadge({ label }: { label: PilotReviewLabel }) {
  const shown = LEGACY_LABELS[label] ?? label;
  return <span className={`review-badge ${BADGE_CLASS[shown] ?? "review-minor"}`}>{shown}</span>;
}

function StatusChip({ status }: { status?: string | null }) {
  const ok = status === "consistent";
  const warn = status === "not_consistent" || status === "partially_consistent";
  return <span className={`check-status ${ok ? "ok" : warn ? "warn" : ""}`}>{String(status ?? "unknown").replace(/_/g, " ")}</span>;
}

function mmss(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

const uniq = (items?: string[]) => Array.from(new Set(items ?? []));

function EventAnswers({ ev }: { ev: PilotEvent }) {
  return (
    <dl className="event-qa">
      <dt>What happened?</dt>
      <dd>{ev.what_happened}</dd>
      <dt>When?</dt>
      <dd>
        <span className="event-when">{ev.when.display}</span>
        {ev.when.approximate && <span className="when-note">{ev.when.approximate}</span>}
      </dd>
      {ev.physical_values && ev.physical_values.length > 0 && (
        <>
          <dt>Measured before / during</dt>
          <dd>
            <ul className="value-list">
              {uniq(ev.physical_values).map((t, i) => (
                <li key={`value-${i}-${t}`}>{t}</li>
              ))}
            </ul>
          </dd>
        </>
      )}
      {ev.command_evidence && ev.command_evidence.length > 0 && (
        <>
          <dt>Pilot input check</dt>
          <dd>
            <ul>
              {uniq(ev.command_evidence).map((t, i) => (
                <li key={`cmd-${i}-${t}`}>{t}</li>
              ))}
            </ul>
          </dd>
        </>
      )}
      {ev.consistency_evidence && ev.consistency_evidence.length > 0 && (
        <>
          <dt>Related measurements</dt>
          <dd>
            <ul>
              {uniq(ev.consistency_evidence).map((t, i) => (
                <li key={`consistency-${i}-${t}`}>{t}</li>
              ))}
            </ul>
          </dd>
        </>
      )}
      <dt>Why it matters</dt>
      <dd>{ev.why_review}</dd>
      <dt>What to do</dt>
      <dd>
        <ul>
          {ev.what_to_check.map((t, i) => (
            <li key={`check-${i}-${t}`}>{t}</li>
          ))}
        </ul>
      </dd>
      {ev.when.rows && (
        <dd className="event-rows">
          <span className="muted small">File reference: {ev.when.rows}</span>
        </dd>
      )}
    </dl>
  );
}

function PilotEventCard({
  ev,
  label,
  chart,
}: {
  ev: PilotEvent;
  label: PilotReviewLabel;
  chart?: ReactNode;
}) {
  return (
    <article className={`pilot-event ${BADGE_CLASS[label] ?? ""}`} id={`event-${ev.event_id}`}>
      <div className="pilot-event-head">
        <ReviewBadge label={label} />
        <h3>{ev.title}</h3>
        <span className="event-when-inline">{ev.when.short ?? ev.when.display}</span>
      </div>
      {(ev.interpretation_label || ev.confidence) && (
        <p className="interp-line">
          {ev.interpretation_label}
          {ev.confidence ? ` · confidence ${ev.confidence}` : ""}
        </p>
      )}
      <EventAnswers ev={ev} />
      {chart && <div className="pilot-event-chart">{chart}</div>}
    </article>
  );
}

/** A single bar across the recording, so the pilot can see at a glance where each period sits. */
function FlightTimeline({ report, total }: { report: PilotReport; total: number }) {
  const marks = [
    ...report.attention.flatMap((a) => {
      const start = a.when.start_rel_s;
      const end = a.when.end_rel_s;
      if (start == null || end == null) return [];
      return [{ key: `a-${a.event_id}`, title: a.title, start, end, kind: "main" }];
    }),
    ...report.other_intervals.flatMap((o, i) => {
      const start = o.start_rel_s;
      const end = o.end_rel_s;
      if (start == null || end == null) return [];
      return [{ key: `o-${i}`, title: o.title, start, end, kind: "ref" }];
    }),
  ];
  if (!marks.length || total <= 0) return null;
  const pct = (v: number) => Math.min(100, Math.max(0, (v / total) * 100));
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => mmss(total * f));
  return (
    <figure className="flight-timeline">
      <div className="ft-bar" role="img" aria-label="Where the flagged periods sit in the flight">
        {marks.map((m) => (
          <span
            key={m.key}
            className={`ft-mark ${m.kind}`}
            style={{
              left: `${pct(m.start)}%`,
              width: `max(0.7%, ${pct(Math.max(m.end - m.start, total * 0.007))}%)`,
            }}
            title={`${m.title} — ${mmss(m.start)}`}
          />
        ))}
      </div>
      <div className="ft-axis" aria-hidden="true">
        {ticks.map((t, i) => (
          <span key={`${t}-${i}`}>{t}</span>
        ))}
      </div>
      <figcaption>
        <span className="ft-key main" /> worth checking <span className="ft-key ref" /> for reference. The wider the band,
        the longer the period.
      </figcaption>
    </figure>
  );
}

export type BriefFact = { label: string; value: string; note?: string };

/**
 * Shared pilot briefing layout. Reading order: what this means, where it happened, what to check,
 * what changed, what to do, period detail, overall behaviour, what was checked, limitations.
 * Engineer-only evidence stays in the collapsed appendix.
 */
export function PilotBriefing({
  report,
  events,
  charts,
  facts,
  summaryExtra,
  narrativeSlot,
  technical,
  technicalLabel = "Technical details (engineering review)",
}: {
  report: PilotReport;
  events: PilotEvent[];
  charts?: (eventId: string) => ReactNode;
  facts?: BriefFact[];
  summaryExtra?: ReactNode;
  narrativeSlot?: ReactNode;
  technical?: ReactNode;
  technicalLabel?: string;
}) {
  const byId = new Map(events.map((e) => [e.event_id, e]));
  const attention = report.attention;
  const lastMark = Math.max(
    0,
    ...attention.map((a) => a.when.end_rel_s ?? 0),
    ...report.other_intervals.map((o) => o.end_rel_s ?? 0)
  );
  const total = report.recording_s ?? report.coverage?.flight_s ?? (lastMark > 0 ? lastMark * 1.05 : null);
  const overview = report.overview;

  return (
    <div className="briefing">
      <section className="brief-section brief-summary" aria-labelledby="brief-summary">
        <h2 className="brief-kicker" id="brief-summary">
          At a glance
        </h2>
        {report.bottom_line && <p className="bottom-line">{report.bottom_line}</p>}
        <p className="brief-lede">{report.summary}</p>
        {facts && facts.length > 0 && (
          <dl className="brief-facts">
            {facts.map((f) => (
              <div key={f.label} className="fact">
                <dt>{f.label}</dt>
                <dd>
                  {f.value}
                  {f.note && <small>{f.note}</small>}
                </dd>
              </div>
            ))}
          </dl>
        )}
        {report.time_note && <p className="time-note">{report.time_note}</p>}
        {overview?.sequence && overview.sequence.length > 0 && (
          <>
            <h3 className="mini-head">What the recording shows, in order</h3>
            <ol className="flight-sequence">
              {uniq(overview.sequence).map((t, i) => (
                <li key={`seq-${i}-${t}`}>{t}</li>
              ))}
            </ol>
          </>
        )}
        {summaryExtra}
        {narrativeSlot}
      </section>

      {total != null && total > 0 && (
        <section className="brief-section" aria-labelledby="brief-where">
          <h2 id="brief-where">Where in the flight</h2>
          <FlightTimeline report={report} total={total} />
        </section>
      )}

      <section className="brief-section" aria-labelledby="brief-attention">
        <h2 id="brief-attention">What to check</h2>
        <p className="section-note">
          Ordered by how closely to look, not by how serious the flight was. A label is not a safety judgement.
        </p>
        {attention.length === 0 ? (
          <p className="muted">
            No period needed explanation in the part of the recording that could be checked.
          </p>
        ) : (
          <ul className="attention-summary">
            {attention.map((a, i) => (
              <li key={a.event_id ?? i}>
                <ReviewBadge label={a.review_label} />
                <span className="attention-body">
                  {a.event_id ? (
                    <a className="attention-link" href={`#event-${a.event_id}`}>
                      {a.title}
                    </a>
                  ) : (
                    <span className="attention-link">{a.title}</span>
                  )}
                  <span className="attention-when">
                    {a.when.short ?? a.when.display}
                    {a.interpretation_label ? ` · ${a.interpretation_label}` : ""}
                  </span>
                </span>
                {a.event_id && (
                  <a className="attention-jump no-print" href={`#event-${a.event_id}`}>
                    Details ↓
                  </a>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="brief-section" aria-labelledby="brief-observations">
        <h2 id="brief-observations">What changed</h2>
        <ul className="brief-observations">
          {uniq(report.key_observations).map((t, i) => (
            <li key={`obs-${i}-${t}`}>{t}</li>
          ))}
        </ul>
      </section>

      <section className="brief-section" aria-labelledby="brief-review">
        <h2 id="brief-review">What to do next</h2>
        <ol className="brief-steps">
          {report.next_steps.map((s, i) => (
            <li key={`${s.event_id}-${i}`}>{s.text}</li>
          ))}
        </ol>
        <p className="section-note">
          Start with the first item. The recording shows what changed, but not the pilot&apos;s intent or the cause.
        </p>
      </section>

      {attention.length > 0 && (
        <section className="brief-section brief-details" aria-labelledby="brief-details">
          <h2 id="brief-details">Period by period</h2>
          <p className="section-note">Each card answers what happened, when, why it matters and what to do.</p>
          <div className="attention-list">
            {attention.map((a) => {
              const ev = a.event_id ? byId.get(a.event_id) : undefined;
              if (!ev) return null;
              return (
                <PilotEventCard
                  key={a.event_id}
                  ev={ev}
                  label={a.review_label}
                  chart={charts && a.event_id ? charts(a.event_id) : undefined}
                />
              );
            })}
          </div>
        </section>
      )}

      {report.consistent_events && report.consistent_events.length > 0 && (
        <section className="brief-section" aria-labelledby="brief-consistent">
          <h2 id="brief-consistent">Consistent with the recorded input</h2>
          <p className="section-note">
            These changes follow the pilot input and are listed so they are not mistaken for deviations.
          </p>
          <ul className="consistent-list">
            {report.consistent_events.map((c, i) => (
              <li key={`${c.event_id}-${i}`}>
                <span className="check-status ok">consistent</span>
                <span>{c.title}</span>
                <span className="minor-when">{c.when}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {report.category_summary && report.category_summary.length > 0 && (
        <section className="brief-section" aria-labelledby="brief-overall">
          <h2 id="brief-overall">How the flight behaves overall</h2>
          <ul className="category-summary">
            {report.category_summary.map((c) => (
              <li key={c.category}>
                <strong>{c.title}</strong>
                <span>{c.text}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {report.system_checks && report.system_checks.length > 0 && (
        <section className="brief-section" aria-labelledby="brief-checks">
          <h2 id="brief-checks">What the analysis compared</h2>
          <ul className="system-checks">
            {report.system_checks.map((c) => (
              <li key={c.id}>
                <StatusChip status={c.status} />
                <span className="check-label">{c.label}</span>
                <span className="check-summary">{c.summary}</span>
              </li>
            ))}
          </ul>
          {report.status_changes && report.status_changes.count > 0 && (
            <p className="section-note">
              {report.status_changes.count} status-flag transition{report.status_changes.count === 1 ? "" : "s"} on{" "}
              {report.status_changes.signals.join(", ")}. {report.status_changes.note}
            </p>
          )}
        </section>
      )}

      {report.other_intervals.length > 0 && (
        <section className="brief-section" aria-labelledby="brief-other">
          <h2 id="brief-other">Also noted</h2>
          <p className="section-note">Lower priority. Listed so nothing is hidden, not because these need action.</p>
          <div className="other-list">
            {report.other_intervals.map((o, i) => {
              const ev = o.event_id ? byId.get(o.event_id) : undefined;
              return (
                <details className="minor-event" key={`${o.event_id}-${i}`}>
                  <summary>
                    <ReviewBadge label={o.review_label} />
                    <span className="minor-title">{o.title}</span>
                    <span className="minor-when">{o.when}</span>
                  </summary>
                  {ev ? <EventAnswers ev={ev} /> : <p className="muted small">{o.when}</p>}
                </details>
              );
            })}
          </div>
        </section>
      )}

      <section className="brief-section brief-limits" aria-labelledby="brief-limits">
        <h2 id="brief-limits">What this report cannot tell you</h2>
        <p className="limitation-note">{report.limitation}</p>
        <ul className="cannot-list">
          {uniq(report.what_system_cannot_determine).map((t, i) => (
            <li key={`cannot-${i}-${t}`}>{t}</li>
          ))}
        </ul>
        <details className="about-analysis">
          <summary>How this analysis was produced</summary>
          <p>
            <span className="label-tag">{report.mode_label}</span> {report.mode_note}
          </p>
        </details>
      </section>

      {technical && (
        <details className="appendix technical-details">
          <summary>{technicalLabel}</summary>
          <div>{technical}</div>
        </details>
      )}
    </div>
  );
}
