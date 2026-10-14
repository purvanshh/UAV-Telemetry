"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { AppNav } from "@/components/AppNav";
import {
  api,
  ApiError,
  ArtifactsStatus,
  FlightListItem,
  clearLastFlightId,
  setLastFlightId,
} from "@/lib/api";

function formatWhen(iso?: string | null) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function OverviewPage() {
  const router = useRouter();
  const [artifacts, setArtifacts] = useState<ArtifactsStatus | null>(null);
  const [flights, setFlights] = useState<FlightListItem[]>([]);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshFlights = useCallback(async () => {
    try {
      setFlights(await api.listFlights());
    } catch {
      setFlights([]);
    }
  }, []);

  useEffect(() => {
    api.artifacts().then(setArtifacts).catch(() =>
      setArtifacts({
        ready: false,
        missing: ["(API unreachable)"],
        model_dir: "",
        features: [],
        warnings: [],
      })
    );
    refreshFlights();
  }, [refreshFlights]);

  const analyzed = flights.filter((f) => f.status === "analyzed").length;

  async function deleteFlight(id: string) {
    if (!window.confirm(`Delete saved flight "${id}"? This cannot be undone.`)) return;
    setDeletingId(id);
    setError(null);
    try {
      await api.deleteFlight(id);
      clearLastFlightId(id);
      await refreshFlights();
    } catch (err) {
      setError(err instanceof ApiError || err instanceof Error ? err.message : "Delete failed");
    } finally {
      setDeletingId(null);
    }
  }

  function openFlight(id: string) {
    setLastFlightId(id);
    router.push(`/flights/${id}`);
  }

  return (
    <>
      <AppNav />
      <main>
        <section className="hero">
          <span className="kicker">Workspace</span>
          <h1>Overview</h1>
          <p>
            Saved flights stay on disk. Open one to score the LSTM autoencoder, read events,
            and continue into diagnostics or exports.
          </p>
        </section>

        <section className="grid-metrics">
          <div className="metric">
            <div className="label">Saved flights</div>
            <div className="value">{flights.length}</div>
          </div>
          <div className="metric">
            <div className="label">Analyzed</div>
            <div className="value">{analyzed}</div>
          </div>
          <div className="metric">
            <div className="label">Awaiting analysis</div>
            <div className="value">{Math.max(0, flights.length - analyzed)}</div>
          </div>
          <div className="metric">
            <div className="label">Detector</div>
            <div className="value" style={{ fontSize: "1rem" }}>
              {artifacts?.ready ? artifacts.variant || "ready" : "offline"}
            </div>
          </div>
        </section>

        {artifacts && !artifacts.ready && (
          <div className="alert alert-error">
            Missing artifacts: {(artifacts.missing || []).join(", ") || artifacts.error}
          </div>
        )}
        {error && <div className="alert alert-error">{error}</div>}

        <section className="panel">
          <div className="row" style={{ justifyContent: "space-between", marginBottom: "0.75rem" }}>
            <div>
              <h2 style={{ marginBottom: 4 }}>Recent flights</h2>
              <p className="muted" style={{ margin: 0 }}>
                Stored under <span className="mono">data/flights/</span>.
              </p>
            </div>
            <div className="row">
              <button type="button" className="btn" onClick={() => refreshFlights()}>
                Refresh
              </button>
              <Link className="btn btn-primary" href="/upload">
                Upload telemetry
              </Link>
            </div>
          </div>

          {!flights.length ? (
            <p className="muted">No saved flights yet. Start from New Analysis.</p>
          ) : (
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>Flight ID</th>
                    <th>Status</th>
                    <th>Source</th>
                    <th>Created</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {flights.map((f) => (
                    <tr key={f.flight_id}>
                      <td className="mono">{f.flight_id}</td>
                      <td>
                        <span className="chip">{f.status}</span>
                      </td>
                      <td>{f.source || "—"}</td>
                      <td>{formatWhen(f.created_at)}</td>
                      <td>
                        <div className="row">
                          <button
                            type="button"
                            className="btn btn-primary"
                            onClick={() => openFlight(f.flight_id)}
                          >
                            Open
                          </button>
                          <Link className="btn" href={`/flights/${f.flight_id}/reports`}>
                            Reports
                          </Link>
                          <button
                            type="button"
                            className="btn btn-danger"
                            disabled={deletingId === f.flight_id}
                            onClick={() => deleteFlight(f.flight_id)}
                          >
                            {deletingId === f.flight_id ? "Deleting…" : "Delete"}
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
    </>
  );
}
