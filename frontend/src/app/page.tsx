"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { AppNav } from "@/components/AppNav";
import {
  api,
  ApiError,
  ArtifactsStatus,
  FlightListItem,
  UploadResponse,
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

export default function UploadPage() {
  const router = useRouter();
  const [artifacts, setArtifacts] = useState<ArtifactsStatus | null>(null);
  const [flights, setFlights] = useState<FlightListItem[]>([]);
  const [mode, setMode] = useState<"csv" | "zip">("csv");
  const [flightId, setFlightId] = useState("");
  const [busy, setBusy] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadResponse | null>(null);

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

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);
    const form = e.currentTarget;
    try {
      let res: UploadResponse;
      if (mode === "zip") {
        const zip = (form.elements.namedItem("zip") as HTMLInputElement).files?.[0];
        if (!zip) throw new Error("Choose a ZIP archive.");
        res = await api.uploadZip(zip, flightId || undefined);
      } else {
        const imu = (form.elements.namedItem("imu") as HTMLInputElement).files?.[0];
        const att = (form.elements.namedItem("att") as HTMLInputElement).files?.[0];
        const baro = (form.elements.namedItem("baro") as HTMLInputElement).files?.[0];
        if (!imu || !att || !baro) throw new Error("Upload IMU, ATT, and BARO CSV files.");
        res = await api.uploadCsvs({ imu, att, baro, flightId: flightId || undefined });
      }
      setResult(res);
      if (res.ok && res.flight_id) {
        setLastFlightId(res.flight_id);
        await refreshFlights();
        router.push(`/flights/${res.flight_id}`);
      }
    } catch (err) {
      setError(err instanceof ApiError || err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

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
          <h1>UAV Sentinel</h1>
          <p>
            Upload IMU / ATT / BARO telemetry, score the conservative LSTM autoencoder,
            inspect events, and ask GPT-4o mini for evidence-grounded diagnostics.
          </p>
        </section>

        <section className="panel">
          <h2>Model artifacts</h2>
          {!artifacts && <p className="muted">Checking…</p>}
          {artifacts?.ready && (
            <div className="alert alert-ok">
              Ready · variant <span className="mono">{artifacts.variant}</span> · threshold{" "}
              <span className="mono">{artifacts.threshold?.toFixed(5)}</span> ·{" "}
              {artifacts.n_features} features
            </div>
          )}
          {artifacts && !artifacts.ready && (
            <div className="alert alert-error">
              Missing artifacts: {(artifacts.missing || []).join(", ") || artifacts.error}
            </div>
          )}
        </section>

        <section className="panel">
          <div className="row" style={{ justifyContent: "space-between", marginBottom: "0.75rem" }}>
            <div>
              <h2 style={{ marginBottom: 4 }}>Saved flights</h2>
              <p className="muted" style={{ margin: 0 }}>
                Flights stay on disk under <span className="mono">data/flights/</span>. Open one
                anytime — no need to re-upload.
              </p>
            </div>
            <button type="button" className="btn" onClick={() => refreshFlights()}>
              Refresh
            </button>
          </div>

          {!flights.length ? (
            <p className="muted">No saved flights yet. Upload below to create one.</p>
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

        <section className="panel">
          <h2>Upload new flight</h2>
          <p className="muted">
            Use this only when bringing in a new telemetry set. Existing flights are listed above.
          </p>
          <div className="tabs">
            <button
              type="button"
              className={mode === "csv" ? "btn btn-primary" : "btn"}
              onClick={() => setMode("csv")}
            >
              Three CSVs
            </button>
            <button
              type="button"
              className={mode === "zip" ? "btn btn-primary" : "btn"}
              onClick={() => setMode("zip")}
            >
              ZIP archive
            </button>
          </div>

          <form onSubmit={onSubmit}>
            <div className="file-field" style={{ marginBottom: "0.85rem" }}>
              <label htmlFor="flight_id">Optional flight ID</label>
              <input
                id="flight_id"
                type="text"
                value={flightId}
                onChange={(e) => setFlightId(e.target.value)}
                placeholder="auto-generated if empty"
              />
            </div>

            {mode === "csv" ? (
              <div className="grid-2">
                <div className="file-field">
                  <label htmlFor="imu">IMU.csv</label>
                  <input id="imu" name="imu" type="file" accept=".csv,text/csv" required />
                </div>
                <div className="file-field">
                  <label htmlFor="att">ATT.csv</label>
                  <input id="att" name="att" type="file" accept=".csv,text/csv" required />
                </div>
                <div className="file-field">
                  <label htmlFor="baro">BARO.csv</label>
                  <input id="baro" name="baro" type="file" accept=".csv,text/csv" required />
                </div>
              </div>
            ) : (
              <div className="file-field">
                <label htmlFor="zip">Flight ZIP (contains IMU / ATT / BARO CSVs)</label>
                <input id="zip" name="zip" type="file" accept=".zip,application/zip" required />
              </div>
            )}

            {error && <div className="alert alert-error">{error}</div>}
            {result && !result.ok && (
              <div className="alert alert-error">
                {(result.errors || []).map((e) => (
                  <div key={e}>{e}</div>
                ))}
              </div>
            )}
            {result?.warnings?.length ? (
              <div className="alert alert-warn">
                {result.warnings.map((w) => (
                  <div key={w}>{w}</div>
                ))}
              </div>
            ) : null}

            <div className="row" style={{ marginTop: "1rem" }}>
              <button
                className="btn btn-primary"
                type="submit"
                disabled={busy || artifacts?.ready === false}
              >
                {busy ? "Validating…" : "Validate & continue"}
              </button>
            </div>
          </form>
        </section>
      </main>
    </>
  );
}
