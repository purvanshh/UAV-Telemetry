"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Clock,
  Drone,
  FileText,
  GitCompareArrows,
  Navigation,
  ArrowUpFromDot,
  ScanSearch,
  ShieldCheck,
  ShieldX,
} from "lucide-react";
import { DroneProcessing } from "@/components/drone/DroneProcessing";
import { Button, ButtonLink, Card, Pill } from "@/components/ui";
import type { AgentProgress, AnalysisMeta, AnalysisView, ArjunaSummary } from "@/lib/api";
import { api } from "@/lib/api";
import { duration, fullDate } from "@/lib/format";
import { metaTitle } from "@/lib/verdict";

export function FlightFacts({ s }: { s: ArjunaSummary }) {
  const tl = s.timeline;
  const modes = Array.from(new Set((tl?.modes ?? []).filter((m) => m.value !== "unknown").map((m) => m.value.replace(/_/g, " "))));
  const integrityOk = ["chain_verified", "verified_up_to"].includes(s.integrity.result);
  const rows: { icon: typeof Clock; label: string; value: React.ReactNode }[] = [
    { icon: Clock, label: "Recording", value: duration(tl?.duration_s) },
    { icon: ArrowUpFromDot, label: "In the air", value: duration(tl?.airborne_s) },
    { icon: Navigation, label: "Modes", value: modes.length ? modes.join(", ") : "not recorded" },
    {
      icon: Drone,
      label: "Drone",
      value: s.identity.vehicle_id && s.identity.vehicle_id !== "unknown" ? String(s.identity.vehicle_id) : "not recorded",
    },
    {
      icon: FileText,
      label: "Source",
      value: s.conversion ? "CSV, converted" : s.integrity.converted_log ? "Converted log" : "ARJUNA log",
    },
  ];
  return (
    <Card title="This flight">
      <div className="facts">
        {rows.map((r) => (
          <div key={r.label} className="fact">
            <r.icon aria-hidden="true" />
            <span className="fact-label">{r.label}</span>
            <span className="fact-value" style={{ textTransform: r.label === "Modes" ? "capitalize" : undefined }}>
              {r.value}
            </span>
          </div>
        ))}
        <div className="fact">
          {integrityOk ? <ShieldCheck aria-hidden="true" /> : <ShieldX aria-hidden="true" />}
          <span className="fact-label">Log integrity</span>
          <span className="fact-value" title={s.integrity.detail}>
            <Pill tone={integrityOk ? "go" : s.integrity.result === "modified" ? "warn" : "neutral"}>
              {integrityOk ? "Verified" : s.integrity.result === "modified" ? "Modified" : "Can't verify"}
            </Pill>
          </span>
        </div>
      </div>
    </Card>
  );
}

export function ReviewCard({ view, running, progress, onRun, onOpen }: {
  view: AnalysisView;
  running: boolean;
  progress: AgentProgress | null;
  onRun: () => void;
  onOpen: () => void;
}) {
  const run = view.report;
  const ok = run && (run.status === "ok" || run.status === "partial");
  return (
    <Card title="Detailed review" icon={ScanSearch}>
      {running ? (
        <DroneProcessing
          compact
          title="In progress"
          detail={progress?.running ? `${progress.calls ?? 0} checks so far · ${Math.round(progress.elapsed_s ?? 0)} s` : "About a minute"}
        />
      ) : ok ? (
        <div className="stack" style={{ ["--gap" as string]: "0.6rem" }}>
          <Pill tone="go" icon>
            Done
          </Pill>
          <p className="small muted">
            {run!.provenance.tool_calls ?? run!.trace.length} checks on the data in {Math.round((run!.provenance.latency_ms ?? 0) / 1000)} s by{" "}
            {run!.provenance.model}.
          </p>
          <div className="row">
            <button type="button" className="link-btn small" onClick={onOpen}>
              How it was worked out
            </button>
          </div>
        </div>
      ) : (
        <div className="stack" style={{ ["--gap" as string]: "0.7rem" }}>
          <p className="small muted">An AI explains what happened and why, using only the computed results.</p>
          <Button size="sm" icon={ScanSearch} onClick={onRun} title={view.consent_text}>
            {run ? "Try again" : "Run the detailed review"}
          </Button>
        </div>
      )}
    </Card>
  );
}

/** After a fix: compare with an earlier flight. Earlier flights of the same drone come first. */
export function CompareCard({ view }: { view: AnalysisView }) {
  const [items, setItems] = useState<AnalysisMeta[] | null>(null);
  const [pick, setPick] = useState("");
  useEffect(() => {
    api.list().then(setItems).catch(() => setItems([]));
  }, []);
  const me = view.meta;
  const earlier = useMemo(() => {
    const same = (m: AnalysisMeta) => (me.vehicle_id && me.vehicle_id !== "unknown" && m.vehicle_id === me.vehicle_id ? 0 : 1);
    return (items ?? [])
      .filter((m) => m.analysis_id !== me.analysis_id && !m.refused && m.created_at < me.created_at)
      .sort((a, b) => same(a) - same(b) || b.created_at.localeCompare(a.created_at));
  }, [items, me]);
  if (!earlier.length) return null;
  return (
    <Card title="Did a fix work?" icon={GitCompareArrows} className="no-print">
      <div className="stack" style={{ ["--gap" as string]: "0.65rem" }}>
        <p className="small muted">Compare with an earlier flight to see what got better or worse.</p>
        <select className="select" value={pick} onChange={(e) => setPick(e.target.value)} aria-label="Earlier flight">
          <option value="">Choose the earlier flight</option>
          {earlier.map((m) => (
            <option key={m.analysis_id} value={m.analysis_id}>
              {metaTitle(m)} · {fullDate(m.created_at)}
            </option>
          ))}
        </select>
        <ButtonLink
          size="sm"
          variant="primary"
          icon={GitCompareArrows}
          disabled={!pick}
          href={`/compare?before=${encodeURIComponent(pick)}&after=${encodeURIComponent(me.analysis_id)}`}
        >
          Compare
        </ButtonLink>
      </div>
    </Card>
  );
}
