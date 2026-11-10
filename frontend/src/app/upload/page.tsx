"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { AppNav } from "@/components/AppNav";
import { AssessmentSummary, MappingDetails, UnitQuestions } from "@/components/CombinedResult";
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

export default function NewAnalysisPage() {
  const router = useRouter();
  const [artifacts, setArtifacts] = useState<ArtifactsStatus | null>(null);
  const [mode, setMode] = useState<AnalysisMode>(null);

  // AI Analysis state
  const [aiFile, setAiFile] = useState<File | null>(null);
  const [aiConsent, setAiConsent] = useState(false);
  const [aiBusy, setAiBusy] = useState(false);
  const [aiError, setAiError] = useState<string | null>(null);

  // Deep Analysis state
  const [deepKind, setDeepKind] = useState<"csv" | "zip" | "combined">("csv");
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
    const form = e.currentTarget;
    try {
      if (deepKind === "zip") {
        const zip = (form.elements.namedItem("zip") as HTMLInputElement).files?.[0];
        if (!zip) throw new Error("Choose a ZIP archive.");
        await showValidationSummary(await api.uploadZip(zip, flightId || undefined));
      } else {
        const get = (n: string) =>
          (form.elements.namedItem(n) as HTMLInputElement).files?.[0];
        const imu = get("imu"),
          att = get("att"),
          baro = get("baro");
        if (!imu || !att || !baro)
          throw new Error("Choose all three files: IMU, ATT, and BARO.");
        await showValidationSummary(
          await api.uploadCsvs({ imu, att, baro, flightId: flightId || undefined })
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
    setMode(null);
    setDeepError(null);
    setDeepResult(null);
    setUploadSummary(null);
    setAnalysisError(null);
    setCombinedFile(null);
    setAuto(null);
    setAssign({});
  }

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <>
      <AppNav />
      <main>
        <div className="page-head">
          <h1>New analysis</h1>
          <p className="sub">Choose how you want to analyse your telemetry.</p>
        </div>

        {/* ── Mode selection ────────────────────────────────────────────── */}
        {mode === null && (
          <>
            <div className="analysis-choice-grid" role="group" aria-label="Choose analysis type">
              <button
                type="button"
                className="analysis-card"
                onClick={() => setMode("ai")}
                aria-label="Select AI Analysis"
              >
                <div className="analysis-card-icon" aria-hidden="true">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 8v4l3 3" />
                    <path d="M8.5 8.5 12 12" />
                  </svg>
                </div>
                <h2 className="analysis-card-title">AI Analysis</h2>
                <p className="analysis-card-desc">
                  Get an AI-powered overview of your flight using a single telemetry CSV.
                </p>
                <span className="analysis-card-cta">Select →</span>
              </button>

              <button
                type="button"
                className="analysis-card"
                onClick={() => setMode("deep")}
                aria-label="Select Deep Analysis"
                disabled={artifacts?.ready === false}
              >
                <div className="analysis-card-icon" aria-hidden="true">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M3 3h18v18H3z" />
                    <path d="M3 9h18" /><path d="M3 15h18" />
                    <path d="M9 3v18" /><path d="M15 3v18" />
                  </svg>
                </div>
                <h2 className="analysis-card-title">Deep Analysis</h2>
                <p className="analysis-card-desc">
                  Run a more detailed model-based analysis using supported telemetry files.
                  Separate IMU, attitude, and barometer files may be required.
                </p>
                {artifacts?.ready === false ? (
                  <span className="analysis-card-unavailable">
                    Detector unavailable — {(artifacts.missing || []).join(", ") || artifacts.error}
                  </span>
                ) : (
                  <span className="analysis-card-cta">Select →</span>
                )}
              </button>
            </div>

            <p className="muted small" style={{ marginTop: "1rem" }}>
              Not sure which to pick? Start with <strong>AI Analysis</strong> — it works with any
              telemetry CSV. Use <strong>Deep Analysis</strong> when you have properly formatted
              IMU, ATT, and BARO files and need model-based anomaly detection.
            </p>
          </>
        )}

        {/* ── AI Analysis workflow ──────────────────────────────────────── */}
        {mode === "ai" && (
          <div>
            <div className="mode-header">
              <button
                type="button"
                className="back-link"
                onClick={() => { setMode(null); setAiFile(null); setAiError(null); setAiConsent(false); }}
              >
                ← Back
              </button>
              <div>
                <h2 style={{ margin: 0 }}>AI Analysis</h2>
                <p className="muted small" style={{ margin: "0.2rem 0 0" }}>
                  Upload a single telemetry CSV. The system will inspect the available signals,
                  compute statistics, and generate a readable flight summary.
                </p>
              </div>
            </div>

            <div className="section">
              <div className="file-field" style={{ maxWidth: 480 }}>
                <label htmlFor="ai-file">Telemetry CSV</label>
                <input
                  id="ai-file"
                  type="file"
                  accept=".csv,text/csv"
                  onChange={(e) => { setAiFile(e.target.files?.[0] ?? null); setAiError(null); }}
                />
                {aiFile && (
                  <p className="muted small" style={{ margin: "0.3rem 0 0" }}>
                    {aiFile.name} · {(aiFile.size / 1024).toFixed(0)} KB
                  </p>
                )}
              </div>

              {aiFile && (
                <div className="ai-consent-block">
                  <label className="checkline">
                    <input
                      type="checkbox"
                      checked={aiConsent}
                      onChange={(e) => setAiConsent(e.target.checked)}
                    />
                    <span>
                      Allow an AI model to write the flight summary. Only a bounded statistical
                      summary of this file will be sent — column names, per-column statistics,
                      candidate events with timestamps, and down-sampled signal envelopes. The
                      full file is not sent. Without this, you still get computed statistics
                      and an event list.
                    </span>
                  </label>
                  <p className="muted small" style={{ marginTop: "0.5rem" }}>
                    This analysis uses only the signals present in your file. It does not use
                    the model-based detector and will not fabricate signals or model scores.
                    Limitations and unknowns are reported explicitly.
                  </p>
                </div>
              )}

              {aiError && <div className="alert alert-error" role="alert">{aiError}</div>}

              <button
                className="btn btn-primary"
                disabled={!aiFile || aiBusy}
                onClick={runAiAnalysis}
              >
                {aiBusy
                  ? "Analysing…"
                  : aiConsent
                  ? "Analyse with AI"
                  : aiFile
                  ? "Compute summary (no AI)"
                  : "Choose a file first"}
              </button>
            </div>
          </div>
        )}

        {/* ── Deep Analysis workflow ────────────────────────────────────── */}
        {mode === "deep" && (
          <div>
            <div className="mode-header">
              <button type="button" className="back-link" onClick={resetDeep}>
                ← Back
              </button>
              <div>
                <h2 style={{ margin: 0 }}>Deep Analysis</h2>
                <p className="muted small" style={{ margin: "0.2rem 0 0" }}>
                  Model-based analysis using the deployed anomaly detector. Requires telemetry
                  with the correct signals, units, and timestamps. Incompatible files will be
                  rejected with a clear explanation.
                </p>
              </div>
            </div>

            {/* ── Step 1: choose format and upload ───────────────────── */}
            {!uploadSummary && (
              <div className="section">
                <fieldset className="choice" style={{ border: 0, padding: 0 }} aria-label="Input format">
                  <legend className="muted small" style={{ marginBottom: "0.5rem" }}>
                    How is your telemetry stored?
                  </legend>
                  <label>
                    <input
                      type="radio"
                      name="deepKind"
                      checked={deepKind === "csv"}
                      onChange={() => { setDeepKind("csv"); setDeepResult(null); setDeepError(null); }}
                    />
                    <strong>Three separate CSVs</strong>
                    <span className="d">
                      IMU (gyroscope + accelerometer), ATT (roll, pitch, yaw), BARO (pressure,
                      temperature, altitude) — uploaded together.
                    </span>
                  </label>
                  <label>
                    <input
                      type="radio"
                      name="deepKind"
                      checked={deepKind === "zip"}
                      onChange={() => { setDeepKind("zip"); setDeepResult(null); setDeepError(null); }}
                    />
                    <strong>ZIP archive</strong>
                    <span className="d">A ZIP containing the IMU, ATT, and BARO CSVs.</span>
                  </label>
                  <label>
                    <input
                      type="radio"
                      name="deepKind"
                      checked={deepKind === "combined"}
                      onChange={() => {
                        setDeepKind("combined");
                        setDeepResult(null);
                        setDeepError(null);
                        setCombinedFile(null);
                        setAuto(null);
                        setAssign({});
                      }}
                    />
                    <strong>Combined CSV</strong>
                    <span className="d">
                      A single file with all signals as columns. The system will validate whether
                      it contains every required input and explain any incompatibility — the file
                      is never silently routed to another mode.
                    </span>
                  </label>
                </fieldset>

                <div className="file-field" style={{ maxWidth: 360 }}>
                  <label htmlFor="flight_id">Flight name (optional)</label>
                  <input
                    id="flight_id"
                    type="text"
                    value={flightId}
                    onChange={(e) => setFlightId(e.target.value)}
                    placeholder="generated if empty"
                  />
                </div>

                {deepError && <div className="alert alert-error" role="alert">{deepError}</div>}

                {/* Three CSV form */}
                {deepKind === "csv" && (
                  <form onSubmit={submitDeepSeparate}>
                    <p className="muted small" style={{ marginBottom: "0.75rem" }}>
                      All three files are required. The detector needs gyroscope, accelerometer,
                      attitude angles, and barometer data to run.
                    </p>
                    <div className="grid-2">
                      <div className="file-field">
                        <label htmlFor="imu">
                          IMU — gyroscope and accelerometer
                        </label>
                        <input id="imu" name="imu" type="file" accept=".csv,text/csv" required />
                        <span className="muted small">Required columns: GyrX, GyrY, GyrZ, AccX, AccY, AccZ</span>
                      </div>
                      <div className="file-field">
                        <label htmlFor="att">
                          ATT — roll, pitch, yaw
                        </label>
                        <input id="att" name="att" type="file" accept=".csv,text/csv" required />
                        <span className="muted small">Required columns: Roll, Pitch, Yaw, DesRoll, DesPitch</span>
                      </div>
                      <div className="file-field">
                        <label htmlFor="baro">
                          BARO — barometer (altitude and pressure)
                        </label>
                        <input id="baro" name="baro" type="file" accept=".csv,text/csv" required />
                        <span className="muted small">Required columns: Alt, Press — context only, not scored</span>
                      </div>
                    </div>
                    <button
                      className="btn btn-primary"
                      type="submit"
                      disabled={deepBusy || artifacts?.ready === false}
                    >
                      {deepBusy ? "Validating…" : "Upload and validate"}
                    </button>
                  </form>
                )}

                {/* ZIP form */}
                {deepKind === "zip" && (
                  <form onSubmit={submitDeepSeparate}>
                    <div className="file-field">
                      <label htmlFor="zip">ZIP archive containing IMU, ATT, and BARO CSVs</label>
                      <input
                        id="zip"
                        name="zip"
                        type="file"
                        accept=".zip,application/zip"
                        required
                      />
                    </div>
                    <button
                      className="btn btn-primary"
                      type="submit"
                      disabled={deepBusy || artifacts?.ready === false}
                    >
                      {deepBusy ? "Validating…" : "Upload and validate"}
                    </button>
                  </form>
                )}

                {/* Combined CSV */}
                {deepKind === "combined" && (
                  <div>
                    <div className="alert alert-info">
                      The file will be checked against the detector&apos;s exact input requirements —
                      required signals, units, timestamps, and preprocessing assumptions.
                    </div>
                    <label className="checkline">
                      <input
                        type="checkbox"
                        checked={llmConsent}
                        onChange={(e) => setLlmConsent(e.target.checked)}
                      />
                      <span>
                        Allow an AI model to suggest the meaning of unrecognised column names.
                        Only column names, up to 5 sample values, and min/max per column are sent.
                        Suggestions are validated locally and never decide units or compatibility.
                      </span>
                    </label>
                    <div className="file-field">
                      <label htmlFor="combined">
                        Combined telemetry CSV — validation starts as soon as you choose a file
                      </label>
                      <input
                        id="combined"
                        type="file"
                        accept=".csv,text/csv"
                        onChange={(e) => onPickCombined(e.target.files?.[0] ?? null)}
                      />
                    </div>
                    {deepBusy && (
                      <p className="muted" role="status">
                        Reading the file, checking the schema…
                      </p>
                    )}
                  </div>
                )}

                {/* Combined result */}
                {deepKind === "combined" && auto && !auto.ok && (
                  <div className="alert alert-error" role="alert">
                    {(auto.errors ?? []).map((e) => <div key={e}>{e}</div>)}
                  </div>
                )}

                {deepKind === "combined" && auto?.ok && auto.assessment && (
                  <>
                    <AssessmentSummary a={auto.assessment} />
                    {auto.stage === "analyzed" && auto.flight_id && (
                      <p>
                        Analysis finished.{" "}
                        <a href={`/flights/${encodeURIComponent(auto.flight_id)}`}>
                          Open the results
                        </a>.
                      </p>
                    )}
                    {auto.stage === "analysis_failed" && auto.flight_id && (
                      <div className="alert alert-error" role="alert">
                        The file was accepted but the analysis did not complete.{" "}
                        {(auto.analysis?.errors ?? []).join(" ")}{" "}
                        <a href={`/flights/${encodeURIComponent(auto.flight_id)}`}>Open the flight</a>.
                      </div>
                    )}
                    {auto.stage === "not_stored" && (
                      <div className="alert alert-warn" role="alert">
                        {(auto.upload_errors ?? []).map((e) => <div key={e}>{e}</div>)}
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
                        {Array.from(new Set(auto.warnings)).slice(0, 5).map((w) => <div key={w}>{w}</div>)}
                      </div>
                    )}
                    {auto.llm &&
                      auto.llm.status !== "not_requested" &&
                      auto.llm.status !== "nothing_to_ask" && (
                        <p className="muted small">AI-assisted mapping: {auto.llm.message}</p>
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
                  <div className="alert alert-error" role="alert">
                    {deepResult.errors.map((e) => <div key={e}>{e}</div>)}
                  </div>
                )}
                {deepResult?.warnings?.length ? (
                  <div className="alert alert-warn">
                    {deepResult.warnings.map((w) => <div key={w}>{w}</div>)}
                  </div>
                ) : null}
              </div>
            )}

            {/* ── Step 2: validation summary + run analysis ──────────── */}
            {uploadSummary && deepResult?.flight_id && (
              <div className="section">
                <UploadValidationSummary summary={uploadSummary} />

                {analysisError && (
                  <div className="alert alert-error" role="alert" style={{ marginTop: "0.75rem" }}>
                    {analysisError}
                  </div>
                )}

                <div className="row" style={{ marginTop: "1rem", gap: "0.75rem" }}>
                  <button
                    className="btn btn-primary"
                    onClick={runAnalysis}
                    disabled={analysisBusy || !uploadSummary.ready_for_analysis}
                  >
                    {analysisBusy ? "Running analysis…" : "Run Deep Analysis"}
                  </button>
                  <button
                    type="button"
                    className="btn"
                    onClick={resetDeep}
                  >
                    Upload different files
                  </button>
                  <a
                    className="btn"
                    href={`/flights/${encodeURIComponent(deepResult.flight_id)}`}
                  >
                    Open flight (skip analysis)
                  </a>
                </div>
                {!uploadSummary.ready_for_analysis && (
                  <p className="small" style={{ color: "var(--error)", marginTop: "0.5rem" }}>
                    One or more files have errors. Correct them before running analysis.
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </>
  );
}
