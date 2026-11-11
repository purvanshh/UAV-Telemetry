"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { AppNav } from "@/components/AppNav";
import { DeepReport } from "@/components/DeepReport";
import { FlightTabs } from "@/components/FlightTabs";
import { UploadValidationSummary } from "@/components/UploadValidationSummary";
import { api, ApiError, FlightDetail, Overview, UploadSummary, setLastFlightId } from "@/lib/api";

export default function FlightPage() {
  const flightId = String(useParams().id || "");
  const [flight, setFlight] = useState<FlightDetail | null>(null);
  const [ov, setOv] = useState<Overview | null>(null);
  const [uploadSummary, setUploadSummary] = useState<UploadSummary | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reportKey, setReportKey] = useState(0);

  const load = useCallback(async () => {
    setLastFlightId(flightId);
    const [f, o] = await Promise.all([api.getFlight(flightId), api.overview(flightId)]);
    setFlight(f);
    setOv(o);
    if (o.verdict.state === "not_analyzed") {
      api.uploadSummary(flightId).then(setUploadSummary).catch(() => undefined);
    }
  }, [flightId]);

  useEffect(() => {
    load().catch((e) => setError(e instanceof Error ? e.message : "Could not load this flight."));
  }, [load]);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      await api.analyze(flightId);
      await load();
      setReportKey((k) => k + 1);
    } catch (e) {
      setError(e instanceof ApiError || e instanceof Error ? e.message : "Analysis did not finish.");
    } finally {
      setBusy(false);
    }
  }

  const analysed = !!ov && ov.verdict.state !== "not_analyzed";
  const files = ov ? Array.from(new Set(Object.values(ov.filenames))) : [];

  return (
    <>
      <AppNav />
      <main>
        <Link className="crumb no-print" href="/">
          ← All flights
        </Link>
        <div className="page-head row between">
          <div>
            <p className="brief-page-kicker">Deep Analysis · Flight report</p>
            <h1>What happened on this flight</h1>
            <p className="sub mono">
              {flightId}
              {files.length ? ` · ${files.join(", ")}` : ""}
            </p>
            {analysed && ov?.analyzed_at && (
              <p className="muted small">Analysed {new Date(ov.analyzed_at).toLocaleString()}</p>
            )}
          </div>
          <div className="row no-print">
            {analysed && (
              <button className="btn" onClick={() => window.print()}>
                Print or save as PDF
              </button>
            )}
            {analysed && (
              <button className="btn btn-quiet" onClick={run} disabled={busy}>
                {busy ? "Analysing…" : "Run again"}
              </button>
            )}
          </div>
        </div>
        <FlightTabs flightId={flightId} />

        {error && (
          <div className="alert alert-error" role="alert">
            {error}
          </div>
        )}
        {!ov && !error && <p className="muted">Loading…</p>}

        {ov && !analysed && (
          <section className="section">
            <div className="callout">
              <p>This flight has been uploaded but not analysed yet.</p>
            </div>
            {flight && flight.warnings.length > 0 && (
              <div className="alert alert-warn">
                {flight.warnings.slice(0, 6).map((w) => (
                  <div key={w}>{w}</div>
                ))}
              </div>
            )}
            {uploadSummary && <UploadValidationSummary summary={uploadSummary} />}
            <div className="row" style={{ marginTop: "1rem" }}>
              <button
                className="btn btn-primary btn-lg"
                onClick={run}
                disabled={busy || (uploadSummary ? !uploadSummary.ready_for_analysis : false)}
              >
                {busy ? "Analysing…" : "Run Deep Analysis"}
              </button>
            </div>
            {busy && (
              <p className="muted" role="status" style={{ marginTop: "0.75rem" }}>
                <span className="spinner" aria-hidden="true" /> Checking the flight for periods that look different.
                The report appears here when it is ready.
              </p>
            )}
          </section>
        )}

        {analysed && <DeepReport key={reportKey} flightId={flightId} />}
      </main>
    </>
  );
}
