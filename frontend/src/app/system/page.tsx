"use client";

import { useCallback, useEffect, useState } from "react";
import { AppNav } from "@/components/AppNav";
import { AiStatus, api, ArtifactsStatus } from "@/lib/api";

type Level = "ok" | "warn" | "bad";

type Check = { title: string; level: Level; text: string };

export default function SystemPage() {
  const [service, setService] = useState<boolean | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactsStatus | null>(null);
  const [ai, setAi] = useState<AiStatus | null>(null);
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);
  const [busy, setBusy] = useState(false);

  const check = useCallback(async () => {
    setBusy(true);
    const [h, a, s] = await Promise.allSettled([api.health(), api.artifacts(), api.aiStatus()]);
    setService(h.status === "fulfilled");
    setArtifacts(a.status === "fulfilled" ? a.value : null);
    setAi(s.status === "fulfilled" ? s.value : null);
    setCheckedAt(new Date());
    setBusy(false);
  }, []);

  useEffect(() => {
    void check();
  }, [check]);

  const checks: Check[] = [];
  if (service !== null) {
    checks.push({
      title: "Analysis service",
      level: service ? "ok" : "bad",
      text: service
        ? "Running. Uploads and reports are available."
        : "Not reachable. Uploads and saved reports cannot be loaded until it is running again.",
    });
    const deepReady = !!artifacts?.ready;
    checks.push({
      title: "Deep Analysis",
      level: !service ? "bad" : deepReady ? "ok" : "bad",
      text: !service
        ? "Cannot be checked while the analysis service is down."
        : deepReady
          ? "Ready."
          : "Not available. Deep Analysis cannot run until the detector files are in place. AI Analysis is not affected.",
    });
    checks.push({
      title: "AI Analysis",
      level: service ? "ok" : "bad",
      text: !service
        ? "Cannot be checked while the analysis service is down."
        : ai?.configured
          ? "Ready, including the optional AI-written narrative."
          : "Ready. The optional AI-written narrative is not set up, so reports show the computed summary only.",
    });
  }

  return (
    <>
      <AppNav />
      <main>
        <div className="page-head row between">
          <div>
            <h1>System status</h1>
            <p className="sub">Whether uploads and analysis are working right now.</p>
          </div>
          <button className="btn" onClick={() => void check()} disabled={busy}>
            {busy ? "Checking…" : "Check again"}
          </button>
        </div>

        {service === null && <p className="muted">Checking…</p>}

        {checks.length > 0 && (
          <ul className="status-list" aria-live="polite">
            {checks.map((c) => (
              <li key={c.title}>
                <span className={`dot ${c.level}`} aria-hidden="true" />
                <div>
                  <h3>
                    {c.title}{" "}
                    <span className="sr-only">{c.level === "ok" ? "(working)" : "(needs attention)"}</span>
                  </h3>
                  <p>{c.text}</p>
                </div>
              </li>
            ))}
          </ul>
        )}
        {checkedAt && <p className="muted small">Last checked {checkedAt.toLocaleTimeString()}.</p>}

        <details className="appendix technical-details">
          <summary>Technical diagnostics</summary>
          <div>
            {artifacts ? (
              <ul className="small">
                <li>Detector files: {artifacts.ready ? "all present" : `missing — ${(artifacts.missing || []).join(", ") || artifacts.error || "unknown"}`}</li>
                {artifacts.model_dir && <li>Location: <span className="mono">{artifacts.model_dir}</span></li>}
                {artifacts.n_features != null && <li>Signals used: {artifacts.n_features}</li>}
                {artifacts.win_samples != null && (
                  <li>
                    Analysis window: {artifacts.win_samples} samples, step {artifacts.stride_samples ?? "?"}
                  </li>
                )}
                {(artifacts.warnings ?? []).map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            ) : (
              <p className="muted small">Detector status could not be read.</p>
            )}
            {ai && <p className="small">AI narrative configured: {ai.configured ? "yes" : "no"}.</p>}
          </div>
        </details>
      </main>
    </>
  );
}
