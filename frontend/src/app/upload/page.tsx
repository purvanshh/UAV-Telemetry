"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { AppNav } from "@/components/AppNav";
import { api, ApiError, ArtifactsStatus, UploadResponse, setLastFlightId } from "@/lib/api";

export default function NewAnalysisPage() {
  const router = useRouter();
  const [artifacts, setArtifacts] = useState<ArtifactsStatus | null>(null);
  const [mode, setMode] = useState<"csv" | "zip">("csv");
  const [flightId, setFlightId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadResponse | null>(null);

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
  }, []);

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
        router.push(`/flights/${res.flight_id}`);
      }
    } catch (err) {
      setError(err instanceof ApiError || err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <AppNav />
      <main>
        <section className="hero">
          <span className="kicker">Ingest</span>
          <h1>New analysis</h1>
          <p>
            Upload synchronized IMU, ATT, and BARO files. Validation stays on the server;
            scoring starts from Flight Analysis.
          </p>
        </section>

        <section className="panel">
          <h2>Detector</h2>
          {!artifacts && <p className="muted">Checking artifacts…</p>}
          {artifacts?.ready && (
            <div className="alert alert-ok">
              Ready · variant <span className="mono">{artifacts.variant}</span> · threshold{" "}
              <span className="mono">{artifacts.threshold?.toFixed(5)}</span> ·{" "}
              {artifacts.n_features} features · window {artifacts.win_samples} / stride{" "}
              {artifacts.stride_samples}
            </div>
          )}
          {artifacts && !artifacts.ready && (
            <div className="alert alert-error">
              Missing artifacts: {(artifacts.missing || []).join(", ") || artifacts.error}
            </div>
          )}
        </section>

        <section className="panel">
          <h2>Telemetry files</h2>
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
            <div className="drop">
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
                    <label htmlFor="imu">IMU.csv — gyro and accelerometer</label>
                    <input id="imu" name="imu" type="file" accept=".csv,text/csv" required />
                  </div>
                  <div className="file-field">
                    <label htmlFor="att">ATT.csv — roll, pitch, yaw</label>
                    <input id="att" name="att" type="file" accept=".csv,text/csv" required />
                  </div>
                  <div className="file-field">
                    <label htmlFor="baro">BARO.csv — pressure and temperature</label>
                    <input id="baro" name="baro" type="file" accept=".csv,text/csv" required />
                  </div>
                </div>
              ) : (
                <div className="file-field">
                  <label htmlFor="zip">ZIP containing IMU, ATT, and BARO CSVs</label>
                  <input id="zip" name="zip" type="file" accept=".zip,application/zip" required />
                </div>
              )}
            </div>

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

            <button className="btn btn-primary" type="submit" disabled={busy || artifacts?.ready === false}>
              {busy ? "Validating…" : "Validate and open flight"}
            </button>
          </form>
        </section>
      </main>
    </>
  );
}
