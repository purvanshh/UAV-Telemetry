"use client";

import { FormEvent, Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { AppNav } from "@/components/AppNav";
import { AssessmentSummary, MappingDetails, UnitQuestions } from "@/components/CombinedResult";
import { Dropzone } from "@/components/Dropzone";
import { UploadValidationSummary } from "@/components/UploadValidationSummary";
import {
  api,
  ApiError,
  ArtifactsStatus,
  Assignments,
  AutoResult,
  UploadResponse,
  UploadSummary,
  setLastFlightId,
} from "@/lib/api";

type AnalysisMode = "ai" | "deep" | null;

type Stage = { label: string; state: "done" | "active" | "waiting" };

/** Honest progress: only stages that correspond to a real request, no percentages. */
function ProgressPanel({ title, stages, note }: { title: string; stages: Stage[]; note?: string }) {
  return (
    <div className="progress-panel" role="status" aria-live="polite">
      <h2>{title}</h2>
      <ul className="stages">
        {stages.map((s) => (
          <li key={s.label} className={s.state}>
            <span className="mark" aria-hidden="true">
              {s.state === "done" ? "✓" : s.state === "active" ? <span className="spinner" /> : "·"}
            </span>
            {s.label}
          </li>
        ))}
      </ul>
      {note && <p className="muted small" style={{ margin: 0 }}>{note}</p>}
    </div>
  );
}

/** Turn backend wording into something a pilot can act on. */
function UploadProblems({ messages }: { messages: string[] }) {
  return (
    <div className="alert alert-error" role="alert">
      {messages.map((m) => {
        const dup = /Duplicate upload.*existing flight '([^']+)'/.exec(m);
        if (dup) {
          return (
            <div key={m}>
              These files match a flight you have already uploaded.{" "}
              <Link href={`/flights/${encodeURIComponent(dup[1])}`}>Open that flight ({dup[1]})</Link> instead.
            </div>
          );
        }
        return <div key={m}>{m}</div>;
      })}
    </div>
  );
}

export default function NewAnalysisPage() {
  return (
    <Suspense fallback={null}>
      <NewAnalysis />
    </Suspense>
  );
}

function NewAnalysis() {
  const router = useRouter();
  const params = useSearchParams();
  const q = params.get("mode");
  const mode: AnalysisMode = q === "ai" || q === "deep" ? q : null;
  const [artifacts, setArtifacts] = useState<ArtifactsStatus | null>(null);

  // AI Analysis state
  const [aiFile, setAiFile] = useState<File | null>(null);
  const [aiConsent, setAiConsent] = useState(false);
  const [aiBusy, setAiBusy] = useState(false);
  const [aiError, setAiError] = useState<string | null>(null);

  // Deep Analysis state
  const [deepKind, setDeepKind] = useState<"csv" | "zip" | "combined">("csv");
  const [imuFile, setImuFile] = useState<File | null>(null);
  const [attFile, setAttFile] = useState<File | null>(null);
  const [baroFile, setBaroFile] = useState<File | null>(null);
  const [zipFile, setZipFile] = useState<File | null>(null);
  const [flightId, setFlightId] = useState("");
  const [deepBusy, setDeepBusy] = useState(false);
  const [deepError, setDeepError] = useState<string | null>(null);
  const [deepResult, setDeepResult] = useState<UploadResponse | null>(null);
  // Upload summary shown after a successful store (before running analysis)
  const [uploadSummary, setUploadSummary] = useState<UploadSummary | null>(null);
  const [analysisBusy, setAnalysisBusy] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // Combined CSV (model path) state
  const [combinedFile, setCombinedFile] = useState<File | null>(null);
  const [auto, setAuto] = useState<AutoResult | null>(null);
  const [assign, setAssign] = useState<Assignments>({});
  const [llmConsent, setLlmConsent] = useState(false);

  useEffect(() => {
    api
      .artifacts()
      .then(setArtifacts)
      .catch(() =>
        setArtifacts({
          ready: false,
          missing: ["(service unreachable)"],
          model_dir: "",
          features: [],
          warnings: [],
        })
      );
  }, []);

  // ── AI Analysis handlers ─────────────────────────────────────────────────────

  async function runAiAnalysis() {
    if (!aiFile) return;
    setAiBusy(true);
    setAiError(null);
    try {
      const r = await api.directAnalyze(aiFile, aiConsent);
      router.push(`/direct/${encodeURIComponent(r.analysis_id)}`);
    } catch (err) {
      setAiError(
        err instanceof ApiError || err instanceof Error ? err.message : "Analysis failed"
      );
    } finally {
      setAiBusy(false);
    }
  }

  // ── Deep Analysis handlers ───────────────────────────────────────────────────

  /** After a successful upload, fetch the validation summary and show it. */
  async function showValidationSummary(res: UploadResponse) {
    setDeepResult(res);
    if (!res.ok || !res.flight_id) return;
    setLastFlightId(res.flight_id);
    try {
      const s = await api.uploadSummary(res.flight_id);
      setUploadSummary(s);
    } catch {
      // Non-fatal — just navigate directly if summary fetch fails
      router.push(`/flights/${encodeURIComponent(res.flight_id)}`);
    }
  }

  async function submitDeepSeparate(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setDeepBusy(true);
    setDeepError(null);
    setDeepResult(null);
    setUploadSummary(null);
    setAnalysisError(null);
    try {
      if (deepKind === "zip") {
        if (!zipFile) throw new Error("Choose a ZIP archive.");
        await showValidationSummary(await api.uploadZip(zipFile, flightId || undefined));
      } else {
        if (!imuFile || !attFile || !baroFile)
          throw new Error("Add all three files: IMU, ATT and BARO.");
        await showValidationSummary(
          await api.uploadCsvs({ imu: imuFile, att: attFile, baro: baroFile, flightId: flightId || undefined })
        );
      }
    } catch (err) {
      setDeepError(
        err instanceof ApiError || err instanceof Error ? err.message : "Upload failed"
      );
    } finally {
      setDeepBusy(false);
    }
  }

  /** Run analysis after reviewing the validation summary. */
  async function runAnalysis() {
    if (!deepResult?.flight_id) return;
    setAnalysisBusy(true);
    setAnalysisError(null);
    try {
      await api.analyze(deepResult.flight_id);
      router.push(`/flights/${encodeURIComponent(deepResult.flight_id)}`);
    } catch (err) {
      setAnalysisError(
        err instanceof ApiError || err instanceof Error ? err.message : "Analysis failed"
      );
      setAnalysisBusy(false);
    }
  }

  async function runCombinedAuto(f: File, a: Assignments, consent: boolean) {
    setDeepBusy(true);
    setDeepError(null);
    setDeepResult(null);
    setUploadSummary(null);
    setAnalysisError(null);
    try {
      const r = await api.combinedAuto(f, a, consent, flightId || undefined);
      setAuto(r);
      if (r.stored && r.flight_id) {
        setLastFlightId(r.flight_id);
        if (r.stage === "analyzed")
          router.push(`/flights/${encodeURIComponent(r.flight_id)}`);
      }
    } catch (err) {
      setDeepError(
        err instanceof ApiError || err instanceof Error
          ? err.message
          : "Could not read the file"
      );
    } finally {
      setDeepBusy(false);
    }
  }

  function onPickCombined(f: File | null) {
    setCombinedFile(f);
    setAuto(null);
    setAssign({});
    if (f) void runCombinedAuto(f, {}, llmConsent);
  }

  function setCol(name: string, patch: Partial<{ canonical: string; unit: string }>) {
    setAssign((cur) => {
      const next = { ...cur };
      const merged = {
        canonical: cur[name]?.canonical ?? "",
        unit: cur[name]?.unit ?? "",
        ...patch,
      };
      if (!merged.canonical && !merged.unit) delete next[name];
      else next[name] = merged;
      return next;
    });
  }

  function resetDeep() {
    router.push("/upload");
    setDeepError(null);
    setDeepResult(null);
    setUploadSummary(null);
    setAnalysisError(null);
    setCombinedFile(null);
    setAuto(null);
    setAssign({});
  }

  // ── Render ───────────────────────────────────────────────────────────────────

  const deepUnavailable = artifacts?.ready === false;
  const step = mode === null ? 1 : uploadSummary ? 3 : 2;
  const steps = ["Choose type", "Add files", "Get report"];

  return (
    <>
      <AppNav />
      <main>
        <div className="page-head">
          <h1>Analyse a new flight</h1>
          <p className="sub">
            {mode === null
              ? "Pick the kind of analysis that matches the files you have."
              : mode === "ai"
                ? "AI Analysis"
                : "Deep Analysis"}
          </p>
        </div>

        <ol className="stepper" aria-label="Progress">
          {steps.map((label, i) => (
            <li key={label} aria-current={step === i + 1 ? "step" : undefined} className={step > i + 1 ? "done" : ""}>
              <span className="n">{step > i + 1 ? "✓" : i + 1}</span>
              {label}
            </li>
          ))}
        </ol>

        {/* ── Step 1: choose a mode ─────────────────────────────────────── */}
        {mode === null && (
          <div className="mode-grid" role="group" aria-label="Choose analysis type">
            <button type="button" className="mode-option" onClick={() => router.push("/upload?mode=ai")}>
              <div>
                <h2>AI Analysis</h2>
                <p className="tagline">Quick flight analysis using a single telemetry CSV file.</p>
              </div>
              <dl>
                <div>
                  <dt>What you need</dt>
                  <dd>One telemetry CSV file with all signals in it.</dd>
                </div>
                <div>
                  <dt>What you get</dt>
                  <dd>A short flight briefing: what happened, which moments to review, and what to check next.</dd>
                </div>
              </dl>
              <span className="go">Choose AI Analysis →</span>
            </button>

            <button
              type="button"
              className="mode-option"
              disabled={deepUnavailable}
              onClick={() => router.push("/upload?mode=deep")}
            >
              <div>
                <h2>Deep Analysis</h2>
                <p className="tagline">Detailed anomaly detection using the required separate telemetry files.</p>
              </div>
              <dl>
                <div>
                  <dt>What you need</dt>
                  <dd>Three CSV files — IMU, ATT and BARO — or one ZIP that contains them.</dd>
                </div>
                <div>
                  <dt>What you get</dt>
                  <dd>
                    A briefing plus a timeline of flagged periods, and how much of the flight could be checked.
                  </dd>
                </div>
              </dl>
              {deepUnavailable ? (
                <span className="unavail">
                  Not available right now. <Link href="/system">See System status</Link>
                </span>
              ) : (
                <span className="go">Choose Deep Analysis →</span>
              )}
            </button>
          </div>
        )}

        {/* ── AI Analysis ───────────────────────────────────────────────── */}
        {mode === "ai" && (
          <div style={{ maxWidth: "40rem" }}>
            <Link className="crumb" href="/upload">
              ← Change analysis type
            </Link>

            {aiBusy ? (
              <ProgressPanel
                title={`Analysing ${aiFile?.name ?? "your file"}`}
                stages={[
                  { label: "File received", state: "done" },
                  { label: "Reading the signals and finding periods to review", state: "active" },
                  { label: "Opening your report", state: "waiting" },
                ]}
                note={
                  aiConsent
                    ? "Writing the AI summary can take a little longer. The report opens on its own when it is ready."
                    : "The report opens on its own when it is ready."
                }
              />
            ) : (
              <>
                <p>Add one telemetry CSV. Nothing else is needed — known file layouts are recognised automatically.</p>
                <Dropzone
                  label="Telemetry CSV"
                  accept=".csv,text/csv"
                  file={aiFile}
                  onFile={(f) => {
                    setAiFile(f);
                    setAiError(null);
                  }}
                />

                {aiFile && (
                  <label className="checkline">
                    <input type="checkbox" checked={aiConsent} onChange={(e) => setAiConsent(e.target.checked)} />
                    <span>
                      <strong>Also write the summary with AI (optional).</strong> Only a short statistical summary of
                      the file is sent, never the full file. Without this you still get the full briefing.
                    </span>
                  </label>
                )}

                {aiError && (
                  <div className="alert alert-error" role="alert">
                    {aiError}
                  </div>
                )}

                <button className="btn btn-primary btn-lg" disabled={!aiFile} onClick={runAiAnalysis}>
                  {aiFile ? "Analyse this flight" : "Add a file to continue"}
                </button>
              </>
            )}
          </div>
        )}

        {/* ── Deep Analysis ─────────────────────────────────────────────── */}
        {mode === "deep" && (
          <div>
            {!analysisBusy && (
              <Link className="crumb" href="/upload" onClick={() => void 0}>
                ← Change analysis type
              </Link>
            )}

            {analysisBusy && (
              <ProgressPanel
                title="Running Deep Analysis"
                stages={[
                  { label: "Files uploaded", state: "done" },
                  { label: "Files checked", state: "done" },
                  { label: "Checking the flight for periods that look different", state: "active" },
                  { label: "Opening your report", state: "waiting" },
                ]}
                note="This can take up to a minute. The report opens on its own when it is ready."
              />
            )}

            {!uploadSummary && !analysisBusy && (
              <div style={{ maxWidth: "52rem" }}>
                {deepBusy && deepKind !== "combined" ? (
                  <ProgressPanel
                    title="Checking your files"
                    stages={[
                      { label: "Uploading files", state: "done" },
                      { label: "Checking that every file has the signals Deep Analysis needs", state: "active" },
                      { label: "Showing you the result", state: "waiting" },
                    ]}
                  />
                ) : (
                  <>
                    <p>Choose how your telemetry is stored, then add the files.</p>
                    <div className="seg" role="group" aria-label="How is your telemetry stored?">
                      {(
                        [
                          ["csv", "Three CSV files"],
                          ["zip", "ZIP archive"],
                          ["combined", "One combined CSV"],
                        ] as const
                      ).map(([k, label]) => (
                        <button
                          key={k}
                          type="button"
                          aria-pressed={deepKind === k}
                          onClick={() => {
                            setDeepKind(k);
                            setDeepResult(null);
                            setDeepError(null);
                            setCombinedFile(null);
                            setAuto(null);
                            setAssign({});
                          }}
                        >
                          {label}
                        </button>
                      ))}
                    </div>

                    {deepError && (
                      <div className="alert alert-error" role="alert">
                        {deepError}
                      </div>
                    )}

                    {deepKind === "csv" && (
                      <form onSubmit={submitDeepSeparate}>
                        <div className="required-files">
                          <Dropzone
                            label="IMU file"
                            hint="Gyroscope and accelerometer readings."
                            accept=".csv,text/csv"
                            file={imuFile}
                            onFile={setImuFile}
                          />
                          <Dropzone
                            label="ATT file"
                            hint="Roll, pitch and yaw."
                            accept=".csv,text/csv"
                            file={attFile}
                            onFile={setAttFile}
                          />
                          <Dropzone
                            label="BARO file"
                            hint="Barometer: altitude and pressure."
                            accept=".csv,text/csv"
                            file={baroFile}
                            onFile={setBaroFile}
                          />
                        </div>
                        <details className="small muted" style={{ margin: "0 0 1rem" }}>
                          <summary>Which columns must each file contain?</summary>
                          <ul>
                            <li>IMU: GyrX, GyrY, GyrZ, AccX, AccY, AccZ</li>
                            <li>ATT: Roll, Pitch, Yaw, DesRoll, DesPitch</li>
                            <li>BARO: Alt, Press (shown for context, not checked)</li>
                          </ul>
                        </details>
                        <FlightName value={flightId} onChange={setFlightId} />
                        <button
                          className="btn btn-primary btn-lg"
                          type="submit"
                          disabled={deepBusy || !imuFile || !attFile || !baroFile}
                        >
                          {imuFile && attFile && baroFile ? "Check my files" : "Add all three files to continue"}
                        </button>
                      </form>
                    )}

                    {deepKind === "zip" && (
                      <form onSubmit={submitDeepSeparate}>
                        <Dropzone
                          label="ZIP archive"
                          hint="The ZIP must contain the IMU, ATT and BARO CSV files."
                          accept=".zip,application/zip"
                          file={zipFile}
                          onFile={setZipFile}
                        />
                        <FlightName value={flightId} onChange={setFlightId} />
                        <button className="btn btn-primary btn-lg" type="submit" disabled={deepBusy || !zipFile}>
                          {zipFile ? "Check my files" : "Add a ZIP to continue"}
                        </button>
                      </form>
                    )}

                    {deepKind === "combined" && (
                      <div>
                        <div className="callout">
                          <p>
                            One file with every signal as a column. It is checked as soon as you add it, and anything
                            that is missing or unclear is explained below.
                          </p>
                        </div>
                        <Dropzone
                          label="Combined telemetry CSV"
                          accept=".csv,text/csv"
                          file={combinedFile}
                          onFile={onPickCombined}
                          disabled={deepBusy}
                        />
                        <label className="checkline">
                          <input type="checkbox" checked={llmConsent} onChange={(e) => setLlmConsent(e.target.checked)} />
                          <span>
                            Let AI suggest what unrecognised columns mean (optional). Only column names, a few sample
                            values and each column&apos;s range are sent. Suggestions are checked locally.
                          </span>
                        </label>
                        <FlightName value={flightId} onChange={setFlightId} />
                        {deepBusy && (
                          <p className="muted" role="status">
                            <span className="spinner" aria-hidden="true" /> Reading the file and checking its columns…
                          </p>
                        )}
                      </div>
                    )}

                    {deepKind === "combined" && auto && !auto.ok && (
                      <div className="alert alert-error" role="alert">
                        {(auto.errors ?? []).map((e) => (
                          <div key={e}>{e}</div>
                        ))}
                      </div>
                    )}

                    {deepKind === "combined" && auto?.ok && auto.assessment && (
                      <>
                        <AssessmentSummary a={auto.assessment} />
                        {auto.stage === "analyzed" && auto.flight_id && (
                          <p>
                            Analysis finished. <a href={`/flights/${encodeURIComponent(auto.flight_id)}`}>Open the report</a>.
                          </p>
                        )}
                        {auto.stage === "analysis_failed" && auto.flight_id && (
                          <div className="alert alert-error" role="alert">
                            The file was accepted but the analysis did not finish. {(auto.analysis?.errors ?? []).join(" ")}{" "}
                            <a href={`/flights/${encodeURIComponent(auto.flight_id)}`}>Open the flight</a>.
                          </div>
                        )}
                        {auto.stage === "not_stored" && (
                          <div className="alert alert-warn" role="alert">
                            {(auto.upload_errors ?? []).map((e) => (
                              <div key={e}>{e}</div>
                            ))}
                          </div>
                        )}
                        {auto.stage === "needs_input" && combinedFile && (
                          <UnitQuestions
                            a={auto.assessment}
                            targets={auto.targets ?? []}
                            assign={assign}
                            onChange={(col, canonical, unit) =>
                              setCol(col, { canonical: canonical === "__time__" ? "__time__" : "", unit })
                            }
                            onApply={() => runCombinedAuto(combinedFile, assign, llmConsent)}
                            busy={deepBusy}
                          />
                        )}
                        {auto.warnings && auto.warnings.length > 0 && (
                          <div className="alert alert-info">
                            {Array.from(new Set(auto.warnings))
                              .slice(0, 5)
                              .map((w) => (
                                <div key={w}>{w}</div>
                              ))}
                          </div>
                        )}
                        {auto.llm && auto.llm.status !== "not_requested" && auto.llm.status !== "nothing_to_ask" && (
                          <p className="muted small">AI-assisted column naming: {auto.llm.message}</p>
                        )}
                        {combinedFile && (
                          <MappingDetails
                            cols={auto.columns ?? []}
                            targets={auto.targets ?? []}
                            assign={assign}
                            setCol={setCol}
                            onRecheck={() => runCombinedAuto(combinedFile, assign, llmConsent)}
                            busy={deepBusy}
                          />
                        )}
                      </>
                    )}

                    {deepResult && !deepResult.ok && deepKind !== "combined" && (
                      <UploadProblems messages={deepResult.errors} />
                    )}
                    {deepResult?.warnings?.length ? (
                      <div className="alert alert-warn">
                        {deepResult.warnings.map((w) => (
                          <div key={w}>{w}</div>
                        ))}
                      </div>
                    ) : null}
                  </>
                )}
              </div>
            )}

            {/* ── Step 3: files checked — review and run ───────────────── */}
            {uploadSummary && deepResult?.flight_id && !analysisBusy && (
              <div>
                <UploadValidationSummary summary={uploadSummary} />

                {analysisError && (
                  <div className="alert alert-error" role="alert" style={{ marginTop: "0.75rem" }}>
                    {analysisError}
                  </div>
                )}

                <div className="row" style={{ marginTop: "1.25rem" }}>
                  <button
                    className="btn btn-primary btn-lg"
                    onClick={runAnalysis}
                    disabled={!uploadSummary.ready_for_analysis}
                  >
                    Run Deep Analysis
                  </button>
                  <button type="button" className="btn" onClick={resetDeep}>
                    Use different files
                  </button>
                </div>
                {!uploadSummary.ready_for_analysis && (
                  <p className="small" style={{ color: "var(--error)", marginTop: "0.5rem" }}>
                    One or more files have problems. Fix them and add the files again before running the analysis.
                  </p>
                )}
                <p className="muted small" style={{ marginTop: "0.75rem" }}>
                  Want to look first? <Link href={`/flights/${encodeURIComponent(deepResult.flight_id)}`}>Open the flight without analysing</Link>.
                </p>
              </div>
            )}
          </div>
        )}
      </main>
    </>
  );
}

function FlightName({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div className="file-field" style={{ maxWidth: 360 }}>
      <label htmlFor="flight_id">Flight name (optional)</label>
      <input
        id="flight_id"
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Leave empty to name it automatically"
      />
    </div>
  );
}
