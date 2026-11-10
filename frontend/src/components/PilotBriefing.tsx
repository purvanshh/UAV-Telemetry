import type { ReactNode } from "react";
import type { PilotEvent, PilotReport, PilotReviewLabel } from "@/lib/api";

const BADGE_CLASS: Record<PilotReviewLabel, string> = {
  "Review recommended": "review-first",
  "Worth checking": "review",
  "No notable change detected": "review-minor",
};

export function ReviewBadge({ label }: { label: PilotReviewLabel }) {
  return <span className={`review-badge ${BADGE_CLASS[label] ?? "review-minor"}`}>{label}</span>;
}

function EventAnswers({ ev }: { ev: PilotEvent }) {
  return (
    <dl className="event-qa">
      <dt>What happened?</dt>
      <dd>{ev.what_happened}</dd>
      <dt>When?</dt>
      <dd>
        {ev.when.display}
        {ev.when.approximate && <span className="when-note">{ev.when.approximate}</span>}
      </dd>
      <dt>Why should I look at it?</dt>
      <dd>{ev.why_review}</dd>
      <dt>What should I check?</dt>
      <dd>
        <ul>
          {ev.what_to_check.map((t) => (
            <li key={t}>{t}</li>
          ))}
        </ul>
      </dd>
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
      </div>
      <EventAnswers ev={ev} />
      {chart && <div className="pilot-event-chart">{chart}</div>}
    </article>
  );
}

export type BriefFact = { label: string; value: string; note?: string };

/**
 * Shared pilot briefing layout for both analysis modes. The summary, attention
 * list, observations and review steps are always in the same place; only the
 * wording supplied by the backend differs between AI Analysis and Deep Analysis.
 */
export function PilotBriefing({
  report,
  events,
  charts,
  facts,
  summaryExtra,
  narrativeSlot,
  technical,
  technicalLabel = "Technical Details (for engineering review)",
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

  return (
    <div className="briefing">
      <section className="brief-section brief-summary" aria-labelledby="brief-summary">
        <h2 className="brief-kicker" id="brief-summary">
          Flight summary
        </h2>
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
        <p className="mode-note">
          <span className="label-tag">{report.mode_label}</span> {report.mode_note}
        </p>
        {summaryExtra}
        {narrativeSlot}
      </section>

      <section className="brief-section" aria-labelledby="brief-attention">
        <h2 id="brief-attention">What needs attention?</h2>
        <p className="section-note">
          Review labels order the list. They describe how closely to look, not how serious or unsafe the flight was.
        </p>
        {attention.length === 0 ? (
          <p className="muted">No period stood out in the part of the recording that could be assessed.</p>
        ) : (
          <ul className="attention-summary">
            {attention.map((a) => (
              <li key={a.event_id}>
                <ReviewBadge label={a.review_label} />
                <span className="attention-body">
                  {a.event_id ? (
                    <a className="attention-link" href={`#event-${a.event_id}`}>
                      {a.title}
                    </a>
                  ) : (
                    <span className="attention-link">{a.title}</span>
                  )}
                  <span className="attention-when">{a.when.display}</span>
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="brief-section" aria-labelledby="brief-observations">
        <h2 id="brief-observations">Key observations</h2>
        <ul className="brief-observations">
          {report.key_observations.map((t) => (
            <li key={t}>{t}</li>
          ))}
        </ul>
      </section>

      <section className="brief-section" aria-labelledby="brief-review">
        <h2 id="brief-review">What should I review?</h2>
        <ol className="brief-steps">
          {report.next_steps.map((s, i) => (
            <li key={`${s.event_id}-${i}`}>{s.text}</li>
          ))}
        </ol>
        <p className="section-note">
          Start with the first item. The recording shows what changed, but it cannot show the pilot&apos;s intent or the
          cause.
        </p>
      </section>

      {attention.length > 0 && (
        <section className="brief-section brief-details" aria-labelledby="brief-details">
          <h2 id="brief-details">Flagged periods in detail</h2>
          <p className="section-note">
            Each card answers what happened, when, why it is worth a look, and what to check.
          </p>
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

      {report.other_intervals.length > 0 && (
        <section className="brief-section" aria-labelledby="brief-other">
          <h2 id="brief-other">Other intervals</h2>
          <p className="section-note">
            Listed for completeness. These did not meet the same review threshold as the periods above.
          </p>
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

      <p className="limitation-note">{report.limitation}</p>

      {technical && (
        <details className="appendix technical-details">
          <summary>{technicalLabel}</summary>
          <div>{technical}</div>
        </details>
      )}
    </div>
  );
}
