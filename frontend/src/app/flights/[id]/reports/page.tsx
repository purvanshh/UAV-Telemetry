"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { AppNav } from "@/components/AppNav";
import { api, exportUrl, FlightDetail, setLastFlightId } from "@/lib/api";

export default function ReportsPage() {
  const params = useParams();
  const flightId = String(params.id || "");
  const [flight, setFlight] = useState<FlightDetail | null>(null);
  const [trace, setTrace] = useState("");
  const [reportPreview, setReportPreview] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        setLastFlightId(flightId);
        const f = await api.getFlight(flightId);
        setFlight(f);
        if (f.has_report) {
          const t = await api.evidenceTrace(flightId);
          setTrace(t.formatted);
          const report = await api.report(flightId);
          setReportPreview(JSON.stringify(report, null, 2));
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load reports");
      }
    })();
  }, [flightId]);

  return (
    <>
      <AppNav flightId={flightId} />
      <main>
        <section className="hero">
          <span className="kicker">Archive</span>
          <h1>Reports</h1>
          <p className="mono">{flightId}</p>
        </section>

        {error && <div className="alert alert-error">{error}</div>}

        <section className="panel">
          <h2>Downloads</h2>
          {!flight?.has_report && (
            <div className="alert alert-info">
              Run anomaly analysis first to generate exports.{" "}
              <Link href={`/flights/${flightId}`}>Open analysis</Link>
            </div>
          )}
          <div className="row">
            <a className="btn btn-primary" href={exportUrl(flightId, "json")}>
              Flight report JSON
            </a>
            <a className="btn" href={exportUrl(flightId, "windows.csv")}>
              Windows CSV
            </a>
            <a className="btn" href={exportUrl(flightId, "events.csv")}>
              Events CSV
            </a>
          </div>
          {flight?.has_ai_diagnostic && (
            <p className="muted" style={{ marginTop: "0.85rem" }}>
              The JSON report includes <span className="mono">ai_analysis</span> when a
              diagnostic has been generated.
            </p>
          )}
        </section>

        <section className="panel">
          <h2>Evidence pipeline trace</h2>
          {trace ? <pre className="trace">{trace}</pre> : <p className="muted">No trace yet.</p>}
        </section>

        {reportPreview && (
          <section className="panel">
            <h2>Report preview</h2>
            <pre className="trace" style={{ maxHeight: 480 }}>
              {reportPreview}
            </pre>
          </section>
        )}
      </main>
    </>
  );
}
