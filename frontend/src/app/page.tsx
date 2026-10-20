"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { AppNav } from "@/components/AppNav";
import { api, ApiError, ArtifactsStatus, FlightListItem, clearLastFlightId } from "@/lib/api";
import { verdictLabel, verdictTone } from "@/lib/format";

function formatWhen(iso?: string | null) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

const SOURCE: Record<string, string> = { individual: "Separate files", zip: "ZIP", combined: "Combined CSV" };
const CHIP: Record<string, string> = { attention: "s-attention", caution: "s-caution", error: "s-error", neutral: "" };

export default function FlightsPage() {
  const [artifacts, setArtifacts] = useState<ArtifactsStatus | null>(null);
  const [flights, setFlights] = useState<FlightListItem[] | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setFlights(await api.listFlights());
    } catch {
      setFlights([]);
      setError("The analysis service is not reachable.");
    }
  }, []);

  useEffect(() => {
    api.artifacts().then(setArtifacts).catch(() => undefined);
    refresh();
  }, [refresh]);

  async function remove(id: string) {
    if (!window.confirm(`Delete saved flight "${id}"? This cannot be undone.`)) return;
    setDeletingId(id);
    setError(null);
    try {
      await api.deleteFlight(id);
      clearLastFlightId(id);
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError || err instanceof Error ? err.message : "Delete failed");
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <>
      <AppNav />
      <main>
        <div className="page-head row between">
          <div>
            <h1>Flights</h1>
            <p className="sub">Open a flight to see what was detected, when, and what to review.</p>
          </div>
          <Link className="btn btn-primary" href="/upload">
            New analysis
          </Link>
        </div>

        {artifacts && !artifacts.ready && (
          <div className="alert alert-error" role="alert">
            The detector is not available ({(artifacts.missing || []).join(", ") || artifacts.error}). Flights
            cannot be analysed until this is fixed.
          </div>
        )}
        {error && <div className="alert alert-error">{error}</div>}

        {flights === null && <p className="muted">Loading…</p>}
        {flights && flights.length === 0 && !error && (
          <p className="muted">No flights yet. Start with a new analysis.</p>
        )}
        {flights && flights.length > 0 && (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>Flight</th>
                  <th>Result</th>
                  <th>Source</th>
                  <th>Added</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {flights.map((f) => (
                  <tr key={f.flight_id}>
                    <td>
                      <Link href={`/flights/${encodeURIComponent(f.flight_id)}`} className="mono">
                        {f.flight_id}
                      </Link>
                    </td>
                    <td>
                      <span className={`chip ${CHIP[verdictTone(f.verdict_state)]}`}>
                        {verdictLabel(f.verdict_state)}
                      </span>
                    </td>
                    <td>{f.source ? SOURCE[f.source] ?? f.source : "—"}</td>
                    <td>{formatWhen(f.created_at)}</td>
                    <td>
                      <button
                        type="button"
                        className="btn btn-danger"
                        disabled={deletingId === f.flight_id}
                        onClick={() => remove(f.flight_id)}
                      >
                        {deletingId === f.flight_id ? "Deleting…" : "Delete"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="muted small" style={{ marginTop: "1.5rem" }}>
          “No alerts” means the detector flagged nothing in the part of the flight it could assess. It does not
          mean the aircraft was safe.
        </p>
      </main>
    </>
  );
}
