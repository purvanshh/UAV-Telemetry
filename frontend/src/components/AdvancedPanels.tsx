"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { num } from "@/lib/format";

export function TelemetryTable({ flightId }: { flightId: string }) {
  const [data, setData] = useState<Awaited<ReturnType<typeof api.telemetry>> | null>(null);
  const [open, setOpen] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!open || data) return;
    api.telemetry(flightId, 200).then(setData).catch((e) => setErr(e instanceof Error ? e.message : "Failed"));
  }, [open, data, flightId]);

  return (
    <details className="appendix" onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>Raw aligned telemetry (10 Hz, first 200 rows)</summary>
      <div>
        {err && <div className="alert alert-error">{err}</div>}
        {!data && !err && <p className="muted">Loading…</p>}
        {data && (
          <>
            <p className="muted small">
              {data.returned} of {data.total_rows} rows. Times are log-clock seconds. Empty cells are samples
              with no data.
            </p>
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    {data.columns.map((c) => (
                      <th key={c}>{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r, i) => (
                    <tr key={i}>
                      {data.columns.map((c) => (
                        <td className="num" key={c}>
                          {r[c] == null ? "" : num(r[c], 5)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </details>
  );
}

export function Provenance({ flightId }: { flightId: string }) {
  const [rep, setRep] = useState<Record<string, unknown> | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open || rep) return;
    api.report(flightId).then(setRep).catch(() => setRep({ error: "No report available" }));
  }, [open, rep, flightId]);

  const prov = rep ? { provenance: rep.provenance ?? null, detector: rep.detector ?? null, data_quality: rep.data_quality ?? null } : null;
  return (
    <details className="appendix" onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary>Preprocessing, artifact versions and provenance</summary>
      <div>
        {!prov && <p className="muted">Loading…</p>}
        {prov && <pre className="trace">{JSON.stringify(prov, null, 2)}</pre>}
      </div>
    </details>
  );
}
