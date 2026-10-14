"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { AppNav } from "@/components/AppNav";
import { PlotChart } from "@/components/PlotChart";
import {
  api,
  ApiError,
  AnalyzeResponse,
  AnomalyEvent,
  FlightDetail,
  setLastFlightId,
} from "@/lib/api";

export default function FlightAnalysisPage() {
  const params = useParams();
  const flightId = String(params.id || "");
  const [flight, setFlight] = useState<FlightDetail | null>(null);
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [events, setEvents] = useState<AnomalyEvent[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<string>("");
  const [scoreFig, setScoreFig] = useState<Record<string, unknown> | null>(null);
  const [eventFigs, setEventFigs] = useState<{
    telemetry?: Record<string, unknown>;
    perFeature?: Record<string, unknown>;
    baro?: Record<string, unknown>;
  }>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadFlight = useCallback(async () => {
    setLastFlightId(flightId);
    const f = await api.getFlight(flightId);
    setFlight(f);
    if (f.analysis && typeof f.analysis === "object") {
      setAnalysis(f.analysis as AnalyzeResponse);
    }
    if (f.has_report) {
      const ev = await api.events(flightId);
      setEvents(ev);
      if (ev.length && !selectedEvent) setSelectedEvent(ev[0].event_id);
      const fig = await api.chart(flightId, "score_timeline");
      setScoreFig(fig);
    }
  }, [flightId, selectedEvent]);

  useEffect(() => {
    loadFlight().catch((err) =>
      setError(err instanceof Error ? err.message : "Failed to load flight")
    );
  }, [flightId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!selectedEvent) {
      setEventFigs({});
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const [telemetry, perFeature, baro] = await Promise.all([
          api.chart(flightId, "event_telemetry", selectedEvent),
          api.chart(flightId, "per_feature", selectedEvent),
          api.chart(flightId, "baro", selectedEvent),
        ]);
        if (!cancelled) setEventFigs({ telemetry, perFeature, baro });
      } catch {
        if (!cancelled) setEventFigs({});
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [flightId, selectedEvent]);

  const metrics = useMemo(() => {
    const a = analysis;
    if (!a) return null;
    return [
      { label: "Windows", value: String(a.n_windows ?? 0) },
      { label: "Flagged", value: String(a.n_flagged ?? 0) },
      { label: "Events", value: String(a.n_events ?? 0) },
      { label: "Score max", value: a.score_max != null ? a.score_max.toFixed(5) : "—" },
      { label: "Threshold", value: a.threshold?.toFixed(5) ?? "—" },
      { label: "Status", value: a.status },
    ];
  }, [analysis]);

  async function runAnalyze() {
    setBusy(true);
    setError(null);
    try {
      const res = await api.analyze(flightId);
      setAnalysis(res);
      await loadFlight();
      if (res.ok) {
        const fig = await api.chart(flightId, "score_timeline");
        setScoreFig(fig);
        const ev = await api.events(flightId);
        setEvents(ev);
        if (ev.length) setSelectedEvent(ev[0].event_id);
      }
    } catch (err) {
      setError(err instanceof ApiError || err instanceof Error ? err.message : "Analyze failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <AppNav flightId={flightId} />
      <main>
        <section className="hero">
          <span className="kicker">Reconstruction</span>
          <h1>Flight analysis</h1>
          <p className="mono">{flightId}</p>
        </section>

        {error && <div className="alert alert-error">{error}</div>}

        <section className="panel">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <div>
              <h2 style={{ marginBottom: 4 }}>Detector run</h2>
              <p className="muted" style={{ margin: 0 }}>
                Status: <span className="chip">{flight?.status || "…"}</span>{" "}
                · LSTM scores 12 IMU/ATT features only (BARO excluded)
              </p>
            </div>
            <button className="btn btn-primary" onClick={runAnalyze} disabled={busy}>
              {busy ? "Scoring…" : "Run anomaly analysis"}
            </button>
          </div>

          {metrics && (
            <div className="grid-metrics">
              {metrics.map((m) => (
                <div className="metric" key={m.label}>
                  <div className="label">{m.label}</div>
                  <div className="value">{m.value}</div>
                </div>
              ))}
            </div>
          )}

          {analysis?.warnings?.length ? (
            <div className="alert alert-warn">
              {analysis.warnings.map((w) => (
                <div key={w}>{w}</div>
              ))}
            </div>
          ) : null}
          {analysis?.errors?.length ? (
            <div className="alert alert-error">
              {analysis.errors.map((e) => (
                <div key={e}>{e}</div>
              ))}
            </div>
          ) : null}
        </section>

        {scoreFig && (
          <section className="panel">
            <h2>Anomaly score timeline</h2>
            <PlotChart figure={scoreFig as never} />
          </section>
        )}

        {events.length > 0 && (
          <section className="panel">
            <h2>Events</h2>
            <div className="file-field" style={{ marginBottom: "0.85rem", maxWidth: 360 }}>
              <label htmlFor="event">Selected event</label>
              <select
                id="event"
                value={selectedEvent}
                onChange={(e) => setSelectedEvent(e.target.value)}
              >
                {events.map((ev) => (
                  <option key={ev.event_id} value={ev.event_id}>
                    {ev.event_id} · peak {ev.peak_score?.toFixed?.(4) ?? ev.peak_score}
                  </option>
                ))}
              </select>
            </div>
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>Event</th>
                    <th>Start</th>
                    <th>End</th>
                    <th>Duration</th>
                    <th>Peak</th>
                    <th>Windows</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((ev) => (
                    <tr key={ev.event_id}>
                      <td className="mono">{ev.event_id}</td>
                      <td>{ev.start_time_s.toFixed(2)}s</td>
                      <td>{ev.end_time_s.toFixed(2)}s</td>
                      <td>{ev.duration_s.toFixed(2)}s</td>
                      <td>{Number(ev.peak_score).toFixed(4)}</td>
                      <td>{ev.flagged_window_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {eventFigs.perFeature && (
              <>
                <h3 style={{ marginTop: "1.25rem" }}>Per-feature recon error</h3>
                <PlotChart figure={eventFigs.perFeature as never} />
              </>
            )}
            {eventFigs.telemetry && (
              <>
                <h3>Telemetry around event</h3>
                <PlotChart figure={eventFigs.telemetry as never} />
              </>
            )}
            {eventFigs.baro && (
              <>
                <h3>BARO context (not scored by LSTM)</h3>
                <PlotChart figure={eventFigs.baro as never} />
              </>
            )}
          </section>
        )}

        <section className="panel">
          <div className="row">
            <Link className="btn" href={`/flights/${flightId}/ai`}>
              AI Diagnostics
            </Link>
            <Link className="btn" href={`/flights/${flightId}/reports`}>
              Reports & exports
            </Link>
          </div>
        </section>
      </main>
    </>
  );
}
