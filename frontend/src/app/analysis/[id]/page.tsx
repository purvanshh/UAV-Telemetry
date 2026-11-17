"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { ArrowLeft, ClipboardCheck, Download, FileJson, Printer, Wrench } from "lucide-react";
import { Checklist, checklistItems, NotChecked, ProblemList, StoryCard } from "@/components/report/Briefing";
import { Engineering, type EngTab } from "@/components/report/Engineering";
import { PreflightChecks } from "@/components/report/Preflight";
import { CompareCard, FlightFacts, ReviewCard } from "@/components/report/Sidebar";
import { useToast } from "@/components/Toast";
import { Alert, Button, ButtonLink, Hero, PageHeader, Skeleton } from "@/components/ui";
import { api, ApiError, type AgentProgress, type AnalysisView, type ArjunaSummary } from "@/lib/api";
import { duration, fullDate } from "@/lib/format";
import { PILOT_TONE, PREFLIGHT_TONE } from "@/lib/status";

export default function AnalysisPage() {
  return (
    <Suspense fallback={null}>
      <Analysis />
    </Suspense>
  );
}

function Analysis() {
  const id = String(useParams().id || "");
  const params = useSearchParams();
  const router = useRouter();
  const toast = useToast();
  const [view, setView] = useState<AnalysisView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [agentError, setAgentError] = useState<string | null>(null);
  const [engOpen, setEngOpen] = useState(false);
  const [tab, setTab] = useState<EngTab>("findings");
  const [target, setTarget] = useState<string | null>(null);
  const [progress, setProgress] = useState<AgentProgress | null>(null);
  const started = useRef(false);
  const engRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.get(id).then(setView).catch((e) => setError(e instanceof Error ? e.message : "This report could not be loaded."));
  }, [id]);

  const runAgent = useCallback(async () => {
    setRunning(true);
    setAgentError(null);
    try {
      const v = await api.runAgent(id);
      setView(v);
      const ok = v.report && (v.report.status === "ok" || v.report.status === "partial") && !v.last_attempt;
      toast(ok ? { tone: "go", title: "Detailed review ready" } : { tone: "caution", title: "The detailed review did not finish", body: v.last_attempt?.message });
    } catch (e) {
      const msg = e instanceof ApiError || e instanceof Error ? e.message : "The detailed review failed.";
      setAgentError(msg);
      toast({ tone: "warn", title: "The detailed review failed", body: msg });
    } finally {
      setRunning(false);
    }
  }, [id, toast]);

  // While the review runs, ask the server what it is doing right now.
  useEffect(() => {
    if (!running) {
      setProgress(null);
      return;
    }
    let live = true;
    const poll = () =>
      api
        .agentProgress(id)
        .then((p) => live && p.running && setProgress(p))
        .catch(() => undefined);
    void poll();
    const t = window.setInterval(poll, 1200);
    return () => {
      live = false;
      window.clearInterval(t);
    };
  }, [running, id]);

  // Consent was given on the upload page: start the review once, then drop the flag from the URL.
  useEffect(() => {
    if (!view || started.current || params.get("agent") !== "1") return;
    started.current = true;
    router.replace(`/analysis/${encodeURIComponent(id)}`);
    if (!view.report && !view.summary.refused) void runAgent();
  }, [view, params, router, id, runAgent]);

  const openEngineering = useCallback((t: EngTab, findingId: string | null = null) => {
    setEngOpen(true);
    setTab(t);
    setTarget(null);
    window.setTimeout(() => {
      setTarget(findingId);
      if (!findingId) engRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 60);
  }, []);

  if (error) {
    return (
      <main className="page">
        <Crumb />
        <Alert tone="warn" title="This report could not be opened.">
          {error}
        </Alert>
      </main>
    );
  }
  if (!view) return <Loading />;

  const s = view.summary as ArjunaSummary;
  const pilot = s.pilot;
  const pf = s.preflight;
  const isPre = view.meta.purpose === "preflight";
  const tl = s.timeline;
  const title = s.conversion ? s.conversion.source_file : String(s.identity.flight_id ?? s.file.name);
  const lede = [
    tl?.duration_s != null ? `${duration(tl.duration_s)} recording` : null,
    tl?.airborne_s != null ? `${duration(tl.airborne_s)} in the air` : null,
    `analysed ${fullDate(view.meta.created_at)}`,
  ]
    .filter(Boolean)
    .join(" · ");
  const items = pilot ? checklistItems(view, pilot) : [];

  return (
    <main className="page">
      <Crumb />
      <PageHeader
        eyebrow={isPre ? "Pre-flight check" : "Flight report"}
        title={title}
        lede={lede}
        actions={
          <>
            <Button variant="ghost" size="sm" icon={Printer} onClick={() => window.print()}>
              Print
            </Button>
            <ButtonLink variant="ghost" size="sm" icon={Download} href={api.downloadUrl(view.analysis_id, "ulg")} download>
              Download log
            </ButtonLink>
            <ButtonLink variant="ghost" size="sm" icon={FileJson} href={api.exportUrl(view.analysis_id)} download>
              Export
            </ButtonLink>
          </>
        }
      />

      {isPre && pf ? (
        <Hero tone={PREFLIGHT_TONE[pf.status] ?? "neutral"} icon={pf.status === "ready" ? ClipboardCheck : undefined} title={pf.status_text} detail={pf.status_detail} />
      ) : (
        pilot && <Hero tone={PILOT_TONE[pilot.status] ?? "neutral"} title={pilot.status_text} detail={pilot.status_detail} />
      )}
      {pilot?.integrity_warning && (
        <div style={{ marginTop: "0.9rem" }}>
          <Alert tone="warn" title="Modified recording.">
            {pilot.integrity_warning}
          </Alert>
        </div>
      )}

      {s.refused ? (
        <div style={{ marginTop: "1rem" }}>
          <Alert tone="neutral" title="For the engineering team:">
            the ARJUNA standard requires the app to stop rather than guess when a log&apos;s schema, frames or profile are
            unknown. {s.compatibility.problems.join(" ")}
          </Alert>
        </div>
      ) : (
        <>
          <div className="report">
            <div className="report-main">
              {isPre && pf ? (
                <>
                  <PreflightChecks pf={pf} />
                  <Checklist
                    storageKey={`arjuna:checks:${view.analysis_id}`}
                    title="Before you fly"
                    items={pf.actions}
                    critical={pf.checks.filter((c) => c.status === "fail").map((c) => c.action)}
                  />
                  {pf.caveat && <p className="tiny muted">{pf.caveat}</p>}
                  {pilot && pilot.problems.length > 0 && <ProblemList problems={pilot.problems} onShowData={(f) => openEngineering("findings", f)} />}
                </>
              ) : (
                pilot && (
                  <>
                    <StoryCard view={view} pilot={pilot} running={running} progress={progress} error={agentError} onRun={() => void runAgent()} />
                    <ProblemList problems={pilot.problems} onShowData={(f) => openEngineering("findings", f)} />
                    <Checklist storageKey={`arjuna:checks:${view.analysis_id}`} items={items} critical={pilot.critical_checks} />
                    <NotChecked items={pilot.not_checked} />
                  </>
                )
              )}
            </div>
            <aside className="report-side">
              <FlightFacts s={s} />
              {!isPre && <ReviewCard view={view} running={running} progress={progress} onRun={() => void runAgent()} onOpen={() => openEngineering("review")} />}
              <CompareCard view={view} />
            </aside>
          </div>

          <section className="engineering" ref={engRef} style={{ scrollMarginTop: 76 }}>
            <button type="button" className="engineering-toggle" aria-expanded={engOpen} aria-controls="engineering" onClick={() => setEngOpen(!engOpen)}>
              <Wrench size={18} aria-hidden="true" className="muted" />
              <span style={{ flex: 1 }}>
                <span className="title">Engineering details</span>
                <br />
                <span className="desc">Signals and charts, fault codes, physics checks, timeline and how the review worked</span>
              </span>
              <svg className="chev" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                <path d="m9 18 6-6-6-6" />
              </svg>
            </button>
            <div id="engineering" className={`disclosure-body${engOpen ? " is-open" : ""}`} inert={!engOpen}>
              <div>{engOpen && <Engineering view={view} tab={tab} onTab={setTab} target={target} />}</div>
            </div>
          </section>
        </>
      )}
    </main>
  );
}

function Crumb() {
  return (
    <Link className="crumb no-print" href="/">
      <ArrowLeft size={15} aria-hidden="true" /> All flights
    </Link>
  );
}

function Loading() {
  return (
    <main className="page" aria-busy="true" aria-label="Loading the report">
      <Crumb />
      <div className="stack" style={{ ["--gap" as string]: "0.6rem", marginBottom: "1.5rem" }}>
        <Skeleton h={11} w={110} />
        <Skeleton h={28} w="40%" />
        <Skeleton h={13} w="30%" />
      </div>
      <Skeleton h={96} r={14} />
      <div className="report">
        <div className="report-main">
          <Skeleton h={150} r={14} />
          <Skeleton h={110} r={12} />
          <Skeleton h={110} r={12} />
        </div>
        <div className="report-side">
          <Skeleton h={260} r={14} />
        </div>
      </div>
    </main>
  );
}
