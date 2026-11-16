"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { AppNav } from "@/components/AppNav";
import { api, ApiError, AnalysisMeta } from "@/lib/api";

function formatWhen(iso?: string | null) {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** Plain list wording. Never says "normal" or "safe". */
function result(m: AnalysisMeta): { text: string; tone: string } {
  if (m.refused) return { text: "Could not read this log", tone: "s-error" };
  if (m.classification && m.classification !== "flight") return { text: "Not a full flight", tone: "s-caution" };
  if (m.worst === "critical") return { text: "Do not fly until checked", tone: "s-error" };
  if (m.worst === "warning") return { text: "Check before next flight", tone: "s-attention" };
  return { text: "No problems found", tone: "s-ok" };
}

export default function FlightsPage() {
  const [items, setItems] = useState<AnalysisMeta[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setItems(await api.list());
      setError(null);
    } catch {
      setItems([]);
      setError("The analysis service is not reachable. Check System status.");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function remove(id: string) {
    setDeleting(id);
    try {
      await api.remove(id);
      setConfirming(null);
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError || err instanceof Error ? err.message : "Delete failed");
    } finally {
      setDeleting(null);
    }
  }

  return (
    <>
      <AppNav />
      <main>
        <div className="page-head row between">
          <div>
            <h1>Flights</h1>
            <p className="sub">Upload a flight log to see what happened and what to check before the next flight.</p>
          </div>
          <Link className="btn btn-primary btn-lg" href="/upload">
            Analyse a new flight
          </Link>
        </div>
        {error && <div className="alert alert-error" role="alert">{error}</div>}
        {items === null && <p className="muted">Loading…</p>}
        {items && items.length === 0 && !error && (
          <div className="empty">
            <h2>No flights analysed yet</h2>
            <p>
              Upload an ARJUNA flight log or a zipped flight folder. The code checks every fault code, and the agent
              investigates what happened and why.
            </p>
            <Link className="btn btn-primary btn-lg" href="/upload">
              Analyse your first flight
            </Link>
          </div>
        )}
        {items && items.length > 0 && (
          <ul className="flight-list">
            {items.map((m) => {
              const r = result(m);
              return (
                <li key={m.analysis_id} className="flight-item">
                  <Link className="flight-open" href={`/analysis/${encodeURIComponent(m.analysis_id)}`}>
                    <span className="flight-name">
                      {m.flight_id ?? m.filename}
                      <span className="flight-meta">
                        {[m.converted_from === "csv" ? `CSV → ARJUNA (${m.source_profile})` : "ARJUNA log", m.vehicle_id !== "unknown" ? m.vehicle_id : null, m.flight_id ? m.filename : null]
                          .filter(Boolean)
                          .join(" · ")}
                      </span>
                    </span>
                    <span className="flight-result">
                      <span className={`chip ${r.tone}`}>{r.text}</span>
                      <br />
                      <span className="mode-tag">
                        {m.agent?.status === "ok" || m.agent?.status === "partial" ? "detailed review done" : "automatic checks"}
                      </span>
                    </span>
                    <span className="flight-date">{formatWhen(m.created_at)}</span>
                  </Link>
                  <div className="row">
                    {confirming === m.analysis_id ? (
                      <span className="inline-confirm">
                        Delete this analysis?
                        <button type="button" className="btn btn-danger" disabled={deleting === m.analysis_id} onClick={() => remove(m.analysis_id)}>
                          {deleting === m.analysis_id ? "Deleting…" : "Delete"}
                        </button>
                        <button type="button" className="btn btn-quiet" onClick={() => setConfirming(null)}>
                          Cancel
                        </button>
                      </span>
                    ) : (
                      <button type="button" className="btn btn-quiet" onClick={() => setConfirming(m.analysis_id)}>
                        Delete
                      </button>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </main>
    </>
  );
}
