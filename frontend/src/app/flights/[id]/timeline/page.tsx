"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { AppNav } from "@/components/AppNav";
import { EventChart } from "@/components/EventChart";
import { EventDetailPanel } from "@/components/EventDetailPanel";
import { FlightTabs } from "@/components/FlightTabs";
import { api, EventContext, EventDetail, Timeline } from "@/lib/api";
import { clock, duration } from "@/lib/format";

function TimelineView() {
  const flightId = String(useParams().id || "");
  const router = useRouter();
  const search = useSearchParams();
  const selected = search.get("event");
  const [tl, setTl] = useState<Timeline | null>(null);
  const [detail, setDetail] = useState<EventDetail | null>(null);
  const [ctx, setCtx] = useState<EventContext | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.timeline(flightId).then(setTl).catch((e) => setError(e instanceof Error ? e.message : "Failed"));
  }, [flightId]);

  const select = useCallback(
    (id: string | null) => {
      const base = `/flights/${encodeURIComponent(flightId)}/timeline`;
      router.replace(id ? `${base}?event=${encodeURIComponent(id)}` : base, { scroll: false });
    },
    [flightId, router]
  );

  // Default to the first event once the timeline is known.
  useEffect(() => {
    if (tl && !selected) {
      const first = tl.items.find((i) => i.type === "event");
      if (first) select(first.id);
    }
  }, [tl, selected, select]);

  useEffect(() => {
    setDetail(null);
    setCtx(null);
    if (!selected) return;
    let cancelled = false;
    Promise.all([api.eventDetail(flightId, selected), api.eventContext(flightId, selected)])
      .then(([d, c]) => {
        if (!cancelled) {
          setDetail(d);
          setCtx(c);
        }
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : "Failed"));
    return () => {
      cancelled = true;
    };
  }, [flightId, selected]);

  const total = tl?.total_s || 0;
  const before = useMemo(() => {
    const m: Record<string, number | null | undefined> = {};
    detail?.channels.forEach((c) => (m[c.signal] = c.before?.median));
    return m;
  }, [detail]);

  return (
    <main>
      <div className="page-head">
        <h1>Timeline</h1>
        <p className="sub mono">{flightId}</p>
      </div>
      <FlightTabs flightId={flightId} />
      {error && <div className="alert alert-error">{error}</div>}
      {!tl && !error && <p className="muted">Loading…</p>}

      {tl && tl.items.length === 0 && (
        <p>
          Nothing was flagged and no gaps were recorded. This does not confirm that the aircraft behaved well or
          was safe.
        </p>
      )}

      {tl && tl.items.length > 0 && (
        <>
          {total > 0 && (
            <div
              className="bar-track"
              role="img"
              aria-label={`Recording of ${duration(total)} with ${tl.items.filter((i) => i.type === "event").length} flagged events and ${tl.items.filter((i) => i.type === "unassessed").length} unassessed periods`}
            >
              {tl.items.map((i) => (
                <span
                  key={i.id}
                  className={`bar-mark ${i.type === "unassessed" ? "gap" : ""} ${i.id === selected ? "selected" : ""}`}
                  style={{ left: `${(i.start_rel_s / total) * 100}%`, width: `${(Math.max(i.duration_s, 0.5) / total) * 100}%` }}
                />
              ))}
            </div>
          )}
          <p className="muted small">
            Times are {tl.time_reference}. Hatched marks are periods that could not be assessed.
          </p>
          <div className="split">
            <ol className="event-list" aria-label="Events and unassessed periods">
              {tl.items.map((i) => (
                <li key={i.id}>
                  {i.type === "event" ? (
                    <button
                      type="button"
                      className="event-row"
                      aria-pressed={i.id === selected}
                      onClick={() => select(i.id)}
                      style={{ gridTemplateColumns: "1fr", gap: "0.1rem" }}
                    >
                      <span className="when">
                        {clock(i.start_rel_s)} – {clock(i.end_rel_s)} · {duration(i.duration_s)}
                      </span>
                      <span className="what">
                        {i.title}
                        <small>
                          Segment {(i.session_id ?? 0) + 1}
                          {i.channels.length ? ` · ${i.channels.slice(0, 2).join(" · ")}` : ""}
                        </small>
                      </span>
                    </button>
                  ) : (
                    <div className="event-row gap" style={{ gridTemplateColumns: "1fr", gap: "0.1rem" }}>
                      <span className="when">
                        {clock(i.start_rel_s)} – {clock(i.end_rel_s)} · {duration(i.duration_s)}
                      </span>
                      <span className="what">
                        <span className="gap-mark" />
                        {i.title} — not assessed
                      </span>
                    </div>
                  )}
                </li>
              ))}
            </ol>

            <div>
              {!selected && <p className="muted">Select an event to see what was observed.</p>}
              {selected && !detail && <p className="muted">Loading event…</p>}
              {detail && (
                <>
                  <h2>
                    Event at {detail.start_label}
                    <span className="muted" style={{ fontWeight: 400 }}> · {duration(detail.duration_s)}</span>
                  </h2>
                  {ctx && (
                    <section className="section" aria-label="Chart around the event">
                      <EventChart ctx={ctx} before={before} />
                    </section>
                  )}
                  <EventDetailPanel flightId={flightId} detail={detail} />
                </>
              )}
            </div>
          </div>
        </>
      )}
    </main>
  );
}

export default function TimelinePage() {
  return (
    <>
      <AppNav />
      <Suspense fallback={<main><p className="muted">Loading…</p></main>}>
        <TimelineView />
      </Suspense>
    </>
  );
}
