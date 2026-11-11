"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { AppNav } from "@/components/AppNav";
import { api, ApiError, ArtifactsStatus, FlightListItem, VerdictState, clearLastFlightId } from "@/lib/api";

type DirectItem = Awaited<ReturnType<typeof api.directList>>[number];

type Row = {
  key: string;
  kind: "ai" | "deep";
  id: string;
  href: string;
  name: string;
  detail: string;
  result: string;
  tone: "" | "s-attention" | "s-caution" | "s-error" | "s-ok";
  when: string | null;
};

const SOURCE: Record<string, string> = {
  individual: "Three separate files",
  zip: "ZIP archive",
  combined: "Combined CSV",
};

function formatWhen(iso?: string | null) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** Plain list wording. Never says "normal" or "safe". */
function deepResult(state?: VerdictState | null): Pick<Row, "result" | "tone"> {
  switch (state) {
    case "anomalies_detected":
      return { result: "Periods to review", tone: "s-attention" };
    case "no_alerts_adequate_coverage":
      return { result: "Nothing flagged", tone: "s-ok" };
    case "no_alerts_low_coverage":
      return { result: "Nothing flagged, but only part of the flight could be checked", tone: "s-caution" };
    case "insufficient_data":
      return { result: "Not enough data to analyse", tone: "s-caution" };
    case "analysis_failed":
      return { result: "Analysis did not finish", tone: "s-error" };
    default:
      return { result: "Not analysed yet", tone: "" };
  }
}

function toRows(flights: FlightListItem[], direct: DirectItem[]): Row[] {
  const deep: Row[] = flights.map((f) => {
    const files = Array.from(new Set(Object.values(f.filenames ?? {})));
    return {
      key: `deep-${f.flight_id}`,
      kind: "deep",
      id: f.flight_id,
      href: `/flights/${encodeURIComponent(f.flight_id)}`,
      name: f.flight_id,
      detail: [f.source ? SOURCE[f.source] ?? f.source : "", files.slice(0, 2).join(", ")].filter(Boolean).join(" · "),
      when: f.created_at ?? null,
      ...deepResult(f.verdict_state),
    };
  });
  const ai: Row[] = direct.map((d) => ({
    key: `ai-${d.analysis_id}`,
    kind: "ai",
    id: d.analysis_id,
    href: `/direct/${encodeURIComponent(d.analysis_id)}`,
    name: d.filename,
    detail: `${d.rows.toLocaleString()} rows`,
    result: d.events > 0 ? `${d.events} ${d.events === 1 ? "period" : "periods"} listed` : "Nothing flagged",
    tone: d.events > 0 ? "s-attention" : "s-ok",
    when: d.created_at,
  }));
  return [...deep, ...ai].sort((a, b) => (Date.parse(b.when ?? "") || 0) - (Date.parse(a.when ?? "") || 0));
}

export default function FlightsPage() {
  const [artifacts, setArtifacts] = useState<ArtifactsStatus | null>(null);
  const [flights, setFlights] = useState<FlightListItem[] | null>(null);
  const [direct, setDirect] = useState<DirectItem[] | null>(null);
  const [filter, setFilter] = useState<"all" | "ai" | "deep">("all");
  const [confirming, setConfirming] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const [f, d] = await Promise.allSettled([api.listFlights(), api.directList()]);
    setFlights(f.status === "fulfilled" ? f.value : []);
    setDirect(d.status === "fulfilled" ? d.value : []);
    setError(
      f.status === "rejected" && d.status === "rejected"
        ? "The analysis service is not reachable. Check System status."
        : null
    );
  }, []);

  useEffect(() => {
    api.artifacts().then(setArtifacts).catch(() => undefined);
    void refresh();
  }, [refresh]);

  const rows = useMemo(() => (flights && direct ? toRows(flights, direct) : []), [flights, direct]);
  const visible = rows.filter((r) => filter === "all" || r.kind === filter);
  const loading = flights === null || direct === null;

  async function remove(r: Row) {
    setDeleting(r.key);
    setError(null);
    try {
      if (r.kind === "deep") {
        await api.deleteFlight(r.id);
        clearLastFlightId(r.id);
      } else {
        await api.directDelete(r.id);
      }
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
            <p className="sub">Reopen a report, or analyse a new flight.</p>
          </div>
          <Link className="btn btn-primary btn-lg" href="/upload">
            Analyse a new flight
          </Link>
        </div>

        {artifacts && !artifacts.ready && (
          <div className="alert alert-warn" role="alert">
            Deep Analysis is not available right now. AI Analysis still works.{" "}
            <Link href="/system">See System status</Link>
          </div>
        )}
        {error && (
          <div className="alert alert-error" role="alert">
            {error}
          </div>
        )}

        {loading && !error && <p className="muted">Loading…</p>}

        {!loading && rows.length === 0 && !error && (
          <div className="empty">
            <h2>No flights analysed yet</h2>
            <p>
              Upload a telemetry file and you will get a short report: what happened, which moments to review and
              what to check next.
            </p>
            <Link className="btn btn-primary btn-lg" href="/upload">
              Analyse your first flight
            </Link>
          </div>
        )}

        {!loading && rows.length > 0 && (
          <>
            <div className="filter-row" role="group" aria-label="Show analyses">
              {(
                [
                  ["all", "All"],
                  ["ai", "AI Analysis"],
                  ["deep", "Deep Analysis"],
                ] as const
              ).map(([k, label]) => (
                <button key={k} type="button" aria-pressed={filter === k} onClick={() => setFilter(k)}>
                  {label}
                </button>
              ))}
            </div>

            {visible.length === 0 ? (
              <p className="muted">No {filter === "ai" ? "AI Analysis" : "Deep Analysis"} results yet.</p>
            ) : (
              <ul className="flight-list">
                {visible.map((r) => (
                  <li key={r.key} className="flight-item">
                    <Link className="flight-open" href={r.href}>
                      <span className="flight-name">
                        {r.name}
                        <span className="flight-meta">{r.detail}</span>
                      </span>
                      <span className="flight-result">
                        <span className="mode-tag">{r.kind === "ai" ? "AI Analysis" : "Deep Analysis"}</span>
                        <br />
                        <span className={`chip ${r.tone}`}>{r.result}</span>
                      </span>
                      <span className="flight-date">{formatWhen(r.when)}</span>
                    </Link>
                    <div className="row">
                      {confirming === r.key ? (
                        <span className="inline-confirm">
                          Delete this analysis?
                          <button
                            type="button"
                            className="btn btn-danger"
                            disabled={deleting === r.key}
                            onClick={() => remove(r)}
                          >
                            {deleting === r.key ? "Deleting…" : "Delete"}
                          </button>
                          <button type="button" className="btn btn-quiet" onClick={() => setConfirming(null)}>
                            Cancel
                          </button>
                        </span>
                      ) : (
                        <button
                          type="button"
                          className="btn btn-quiet"
                          aria-label={`Delete ${r.name}`}
                          onClick={() => setConfirming(r.key)}
                        >
                          Delete
                        </button>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            )}
            <p className="muted small" style={{ marginTop: "1rem" }}>
              &ldquo;Nothing flagged&rdquo; means nothing stood out in the part of the recording that could be checked.
              It is not a statement that the flight or the aircraft was safe.
            </p>
          </>
        )}
      </main>
    </>
  );
}
