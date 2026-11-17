"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ClipboardCheck, Drone, ScanSearch, Search, Trash2, Upload } from "lucide-react";
import { useToast } from "@/components/Toast";
import { Alert, Button, ButtonLink, Empty, PageHeader, Segmented, Skeleton, StatusIcon } from "@/components/ui";
import { api, ApiError, AnalysisMeta } from "@/lib/api";
import { fullDate, relativeTime } from "@/lib/format";
import { metaKind, metaTitle, metaVerdict } from "@/lib/verdict";

type Filter = "all" | "flight" | "preflight" | "attention";

const needsAttention = (m: AnalysisMeta) => ["warn", "caution"].includes(metaVerdict(m).tone);

export default function FlightsPage() {
  const toast = useToast();
  const [items, setItems] = useState<AnalysisMeta[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const [confirming, setConfirming] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setItems(await api.list());
      setError(null);
    } catch {
      setItems([]);
      setError("The analysis service is not reachable right now.");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function remove(m: AnalysisMeta) {
    setDeleting(m.analysis_id);
    try {
      await api.remove(m.analysis_id);
      setItems((xs) => (xs ?? []).filter((x) => x.analysis_id !== m.analysis_id));
      toast({ tone: "go", title: "Deleted", body: metaTitle(m) });
    } catch (e) {
      toast({ tone: "warn", title: "Could not delete", body: e instanceof ApiError || e instanceof Error ? e.message : undefined });
    } finally {
      setDeleting(null);
      setConfirming(null);
    }
  }

  const counts = useMemo(() => {
    const xs = items ?? [];
    return {
      all: xs.length,
      attention: xs.filter(needsAttention).length,
      flight: xs.filter((m) => m.purpose !== "preflight").length,
      preflight: xs.filter((m) => m.purpose === "preflight").length,
    };
  }, [items]);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (items ?? []).filter((m) => {
      if (filter === "flight" && m.purpose === "preflight") return false;
      if (filter === "preflight" && m.purpose !== "preflight") return false;
      if (filter === "attention" && !needsAttention(m)) return false;
      return !q || [m.filename, m.flight_id, m.vehicle_id].some((v) => v && v.toLowerCase().includes(q));
    });
  }, [items, filter, query]);

  return (
    <main className="page">
      <PageHeader
        title="Flights"
        lede="Every drone flight and pre-flight check you have uploaded, newest first."
      />

      {error && (
        <div style={{ marginBottom: "1rem" }}>
          <Alert tone="warn" title="Can't reach ARJUNA.">
            {error}{" "}
            <button className="link-btn" onClick={() => void refresh()}>
              Try again
            </button>
          </Alert>
        </div>
      )}

      {items === null && (
        <div className="card flight-list" aria-busy="true" aria-label="Loading flights">
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="flight-row">
              <Skeleton h={36} w={36} r={10} />
              <div className="stack" style={{ ["--gap" as string]: "0.45rem" }}>
                <Skeleton h={14} w="55%" />
                <Skeleton h={11} w="35%" />
              </div>
              <Skeleton h={22} w={150} r={99} />
              <Skeleton h={11} w={70} />
              <span />
            </div>
          ))}
        </div>
      )}

      {items && items.length === 0 && !error && (
        <Empty
          icon={Drone}
          title="No drone flights yet"
          action={
            <ButtonLink href="/upload" variant="primary" size="lg" icon={Upload}>
              Upload your first flight
            </ButtonLink>
          }
        >
          Upload a UAV flight log to see what happened, what to check, and whether the drone is ready for its next mission.
        </Empty>
      )}

      {items && items.length > 0 && (
        <>
          <div className="toolbar">
            <Segmented<Filter>
              label="Show"
              value={filter}
              onChange={setFilter}
              options={[
                { value: "all", label: `All · ${counts.all}` },
                { value: "attention", label: `Needs attention · ${counts.attention}` },
                { value: "flight", label: `Flights · ${counts.flight}` },
                { value: "preflight", label: `Pre-flight · ${counts.preflight}` },
              ]}
            />
            <label className="search">
              <Search size={16} aria-hidden="true" />
              <span className="sr-only">Search flights</span>
              <input
                className="input"
                type="search"
                placeholder="Search by file or flight ID"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </label>
          </div>

          {visible.length === 0 ? (
            <Empty icon={Search} title="Nothing matches">
              Try another search, or show all flights.
            </Empty>
          ) : (
            <div className="card flight-list">
              {visible.map((m, i) => {
                const v = metaVerdict(m);
                const reviewed = m.agent?.status === "ok" || m.agent?.status === "partial";
                return (
                  <div key={m.analysis_id} className="flight-row rise" style={{ ["--i" as string]: Math.min(i, 10) }}>
                    <Link
                      className="row-link"
                      href={`/analysis/${encodeURIComponent(m.analysis_id)}`}
                      aria-label={`Open ${metaTitle(m)}: ${v.text}`}
                    />
                    <StatusIcon tone={v.tone} icon={m.purpose === "preflight" ? ClipboardCheck : undefined} />
                    <div style={{ minWidth: 0 }}>
                      <div className="flight-title">{metaTitle(m)}</div>
                      <div className="flight-sub">
                        <span>{metaKind(m)}</span>
                        {m.vehicle_id && m.vehicle_id !== "unknown" && <span>· {m.vehicle_id}</span>}
                        {reviewed && (
                          <span className="tag" title="An AI review explained this flight">
                            <ScanSearch size={12} aria-hidden="true" /> Reviewed
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flight-status">
                      <span className={`pill tone-${v.tone} pill-dot`}>{v.text}</span>
                    </div>
                    <div className="flight-date" title={fullDate(m.created_at)}>
                      {relativeTime(m.created_at)}
                    </div>
                    <div className="row-actions">
                      <Button
                        variant="ghost"
                        size="sm"
                        icon={Trash2}
                        aria-label={`Delete ${metaTitle(m)}`}
                        onClick={() => setConfirming(m.analysis_id)}
                      />
                    </div>
                    {confirming === m.analysis_id && (
                      <div className="confirm" role="alertdialog" aria-label="Confirm delete">
                        <span>Delete this analysis?</span>
                        <Button size="sm" variant="ghost" onClick={() => setConfirming(null)} autoFocus>
                          Cancel
                        </Button>
                        <Button size="sm" variant="danger-solid" loading={deleting === m.analysis_id} onClick={() => void remove(m)}>
                          Delete
                        </Button>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
          <p className="tiny muted" style={{ marginTop: "0.9rem" }}>
            “No problems found” means nothing went wrong in the parts of the flight that could be checked. It is not a
            guarantee that the drone is safe.
          </p>
        </>
      )}
    </main>
  );
}
