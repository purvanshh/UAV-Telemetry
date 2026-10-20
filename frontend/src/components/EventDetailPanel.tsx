"use client";

import { useState } from "react";
import { api, EventDetail, Explanation } from "@/lib/api";
import { num } from "@/lib/format";

function ChannelTable({ detail }: { detail: EventDetail }) {
  if (!detail.channels.length) return <p className="muted">No per-signal values are available.</p>;
  return (
    <div className="table-wrap">
      <table className="data">
        <thead>
          <tr>
            <th>Signal</th>
            <th>During the event</th>
            <th>Preceding {detail.channels[0]?.before?.span_s ?? 30} s (typical range)</th>
          </tr>
        </thead>
        <tbody>
          {detail.channels.map((c) => (
            <tr key={c.signal}>
              <td>{c.label}</td>
              <td className="num">
                {c.during ? `${num(c.during.min)} to ${num(c.during.max)} ${c.unit ?? ""}` : "—"}
              </td>
              <td className="num">
                {c.before ? `${num(c.before.p05)} to ${num(c.before.p95)} ${c.unit ?? ""}` : "not enough earlier data"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** What was observed / when / evidence / what is uncertain / suggested action. */
export function EventDetailPanel({
  flightId,
  detail,
  showAi = true,
}: {
  flightId: string;
  detail: EventDetail;
  showAi?: boolean;
}) {
  return (
    <div className="detail">
      <h3>What was observed</h3>
      <p>{detail.what_observed}</p>

      <h3>When it occurred</h3>
      <p>{detail.when}</p>

      <h3>Evidence</h3>
      <p className="small">{detail.detector_note}</p>
      <ChannelTable detail={detail} />
      {detail.signal_checks.length > 0 && (
        <>
          <p className="small" style={{ marginTop: "0.6rem" }}>
            <span className="label-tag">Rule-based signal check</span> {detail.signal_checks_note}
          </p>
          <ul>
            {detail.signal_checks.map((c, i) => (
              <li key={i}>{c.text}</li>
            ))}
          </ul>
        </>
      )}

      <h3>What remains uncertain</h3>
      <ul>
        {detail.uncertain.map((u) => (
          <li key={u}>{u}</li>
        ))}
      </ul>

      <h3>Suggested operator action</h3>
      <p>{detail.suggested_action}</p>

      {showAi && <ExplanationBox flightId={flightId} eventId={detail.event_id} />}
    </div>
  );
}

function ExplanationBox({ flightId, eventId }: { flightId: string; eventId: string }) {
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [exp, setExp] = useState<Explanation | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function generate() {
    setBusy(true);
    setErr(null);
    try {
      setExp(await api.createExplanation(flightId));
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Could not generate");
    } finally {
      setBusy(false);
    }
  }
  const mine = exp?.events.find((e) => e.event_id === eventId);

  return (
    <details className="appendix no-print" style={{ marginTop: "1.25rem" }}>
      <summary>Optional: AI-written explanation</summary>
      <div>
        <p className="small">
          Rewords the verified results above in plain language. It cannot change any finding.
        </p>
        <label className="checkline">
          <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
          <span>
            I agree to send a compact summary of this flight&apos;s detected events (times, signal values near
            each event, coverage and limitations — not the raw telemetry) to OpenAI (GPT-4o mini).
          </span>
        </label>
        <button className="btn" onClick={generate} disabled={!consent || busy}>
          {busy ? "Generating…" : "Generate explanation"}
        </button>
        {err && <div className="alert alert-error">{err}</div>}
        {exp && exp.status !== "ok" && <div className="alert alert-info">{exp.message}</div>}
        {exp?.status === "ok" && !mine && (
          <div className="alert alert-info">
            No explanation was produced for this event (text making unsupported claims is discarded).
          </div>
        )}
        {mine && (
          <div>
            <p>
              <span className="label-tag">{exp?.label ?? "AI-generated"}</span>
            </p>
            {(
              [
                ["Observed facts", mine.observed_facts],
                ["Interpretation (uncertain)", mine.interpretation],
                ["Uncertainty", mine.uncertainty],
                ["Suggested review", mine.suggested_review],
              ] as [string, string[]][]
            ).map(([h, items]) =>
              items.length ? (
                <div key={h}>
                  <h3>{h}</h3>
                  <ul>
                    {items.map((t) => (
                      <li key={t}>{t}</li>
                    ))}
                  </ul>
                </div>
              ) : null
            )}
          </div>
        )}
      </div>
    </details>
  );
}
