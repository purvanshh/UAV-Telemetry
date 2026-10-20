"use client";

import { FormEvent, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { AppNav } from "@/components/AppNav";
import { AssessmentSummary, MappingDetails, UnitQuestions } from "@/components/CombinedResult";
import {
  api,
  ApiError,
  ArtifactsStatus,
  Assignments,
  AutoResult,
  UploadResponse,
  setLastFlightId,
} from "@/lib/api";

type Mode = "separate" | "combined";

export default function NewAnalysisPage() {
  const router = useRouter();
  const [artifacts, setArtifacts] = useState<ArtifactsStatus | null>(null);
  const [mode, setMode] = useState<Mode>("separate");
  const [sepKind, setSepKind] = useState<"csv" | "zip">("csv");
  const [flightId, setFlightId] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadResponse | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [auto, setAuto] = useState<AutoResult | null>(null);
  const [assign, setAssign] = useState<Assignments>({});
  const [llmConsent, setLlmConsent] = useState(false);

  useEffect(() => {
    api.artifacts().then(setArtifacts).catch(() =>
      setArtifacts({ ready: false, missing: ["(service unreachable)"], model_dir: "", features: [], warnings: [] })
    );
  }, []);

  function finish(res: UploadResponse) {
    setResult(res);
    if (res.ok && res.flight_id) {
      setLastFlightId(res.flight_id);
      router.push(`/flights/${res.flight_id}`);
    }
  }

  async function submitSeparate(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);
    const form = e.currentTarget;
    try {
      if (sepKind === "zip") {
        const zip = (form.elements.namedItem("zip") as HTMLInputElement).files?.[0];
        if (!zip) throw new Error("Choose a ZIP archive.");
        finish(await api.uploadZip(zip, flightId || undefined));
      } else {
        const get = (n: string) => (form.elements.namedItem(n) as HTMLInputElement).files?.[0];
        const imu = get("imu"), att = get("att"), baro = get("baro");
        if (!imu || !att || !baro) throw new Error("Choose the IMU, ATT and BARO files.");
        finish(await api.uploadCsvs({ imu, att, baro, flightId: flightId || undefined }));
      }
    } catch (err) {
      setError(err instanceof ApiError || err instanceof Error ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function runAuto(f: File, a: Assignments, consent: boolean) {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await api.combinedAuto(f, a, consent, flightId || undefined);
      setAuto(r);
      if (r.stored && r.flight_id) {
        setLastFlightId(r.flight_id);
        if (r.stage === "analyzed") router.push(`/flights/${encodeURIComponent(r.flight_id)}`);
      }
    } catch (err) {
      setError(err instanceof ApiError || err instanceof Error ? err.message : "Could not read the file");
    } finally {
      setBusy(false);
    }
  }

  // Choosing a file starts the whole pipeline: schema analysis, validation and, if compatible, analysis.
  function onPickCombined(f: File | null) {
    setFile(f);
    setAuto(null);
    setAssign({});
    if (f) void runAuto(f, {}, llmConsent);
  }

  function setCol(name: string, patch: Partial<{ canonical: string; unit: string }>) {
    setAssign((cur) => {
      const next = { ...cur };
      const merged = { canonical: cur[name]?.canonical ?? "", unit: cur[name]?.unit ?? "", ...patch };
      if (!merged.canonical && !merged.unit) delete next[name];
      else next[name] = merged;
      return next;
    });
  }

  return (
    <>
      <AppNav />
      <main>
        <div className="page-head">
          <h1>New analysis</h1>
          <p className="sub">Add a flight recording. Nothing is analysed until you open the flight.</p>
        </div>

        {artifacts && !artifacts.ready && (
          <div className="alert alert-error" role="alert">
            The detector is not available ({(artifacts.missing || []).join(", ") || artifacts.error}).
          </div>
        )}

        <fieldset className="choice" style={{ border: 0, padding: 0 }}>
          <legend className="muted small" style={{ marginBottom: "0.4rem" }}>How is your telemetry stored?</legend>
          <label>
            <input type="radio" name="mode" checked={mode === "separate"} onChange={() => setMode("separate")} />
            <strong>Separate files</strong>
            <span className="d">Three CSVs (IMU, ATT, BARO) or one ZIP containing them.</span>
          </label>
          <label>
            <input type="radio" name="mode" checked={mode === "combined"} onChange={() => setMode("combined")} />
            <strong>One combined CSV</strong>
            <span className="d">A single file with all signals as columns on one time column.</span>
          </label>
        </fieldset>

        <div className="file-field" style={{ maxWidth: 360 }}>
          <label htmlFor="flight_id">Flight name (optional)</label>
          <input id="flight_id" type="text" value={flightId} onChange={(e) => setFlightId(e.target.value)} placeholder="generated if empty" />
        </div>

        {error && <div className="alert alert-error" role="alert">{error}</div>}

        {mode === "separate" && (
          <form onSubmit={submitSeparate} className="section">
            <div className="tabs" role="tablist">
              <button type="button" className={sepKind === "csv" ? "btn btn-primary" : "btn"} onClick={() => setSepKind("csv")}>Three CSV files</button>
              <button type="button" className={sepKind === "zip" ? "btn btn-primary" : "btn"} onClick={() => setSepKind("zip")}>ZIP archive</button>
            </div>
            {sepKind === "csv" ? (
              <div className="grid-2">
                <div className="file-field"><label htmlFor="imu">IMU file (gyro and accelerometer)</label><input id="imu" name="imu" type="file" accept=".csv,text/csv" required /></div>
                <div className="file-field"><label htmlFor="att">ATT file (roll, pitch, yaw)</label><input id="att" name="att" type="file" accept=".csv,text/csv" required /></div>
                <div className="file-field"><label htmlFor="baro">BARO file (pressure, temperature, altitude)</label><input id="baro" name="baro" type="file" accept=".csv,text/csv" required /></div>
              </div>
            ) : (
              <div className="file-field"><label htmlFor="zip">ZIP with IMU, ATT and BARO files</label><input id="zip" name="zip" type="file" accept=".zip,application/zip" required /></div>
            )}
            <button className="btn btn-primary" type="submit" disabled={busy || artifacts?.ready === false}>
              {busy ? "Checking…" : "Check and open flight"}
            </button>
          </form>
        )}

        {mode === "combined" && (
          <div className="section">
            <label className="checkline">
              <input type="checkbox" checked={llmConsent} onChange={(e) => setLlmConsent(e.target.checked)} />
              <span>
                Allow AI (GPT-4o mini) to suggest the meaning of columns the system does not recognise. If ticked,
                only those columns&apos; names, up to 5 sample values and min/max are sent to OpenAI — never the rest
                of the file. Suggestions are checked locally and never decide units or compatibility.
              </span>
            </label>
            <div className="file-field">
              <label htmlFor="combined">Combined telemetry CSV — analysis starts as soon as you choose a file</label>
              <input id="combined" type="file" accept=".csv,text/csv" onChange={(e) => onPickCombined(e.target.files?.[0] ?? null)} />
            </div>
            {busy && <p className="muted" role="status">Reading the file, checking the schema…</p>}
          </div>
        )}

        {mode === "combined" && auto && !auto.ok && (
          <div className="alert alert-error" role="alert">
            {(auto.errors ?? []).map((e) => <div key={e}>{e}</div>)}
          </div>
        )}

        {mode === "combined" && auto?.ok && auto.assessment && (
          <>
            <AssessmentSummary a={auto.assessment} />
            {auto.stage === "analyzed" && auto.flight_id && (
              <p>Analysis finished. <a href={`/flights/${encodeURIComponent(auto.flight_id)}`}>Open the results</a>.</p>
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
            {auto.stage === "needs_input" && file && (
              <UnitQuestions
                a={auto.assessment}
                targets={auto.targets ?? []}
                assign={assign}
                onChange={(col, canonical, unit) => setCol(col, { canonical: canonical === "__time__" ? "__time__" : "", unit })}
                onApply={() => runAuto(file, assign, llmConsent)}
                busy={busy}
              />
            )}
            {auto.warnings && auto.warnings.length > 0 && (
              <div className="alert alert-info">
                {Array.from(new Set(auto.warnings)).slice(0, 5).map((w) => <div key={w}>{w}</div>)}
              </div>
            )}
            {auto.llm && auto.llm.status !== "not_requested" && auto.llm.status !== "nothing_to_ask" && (
              <p className="muted small">AI-assisted mapping: {auto.llm.message}</p>
            )}
            {file && (
              <MappingDetails
                cols={auto.columns ?? []}
                targets={auto.targets ?? []}
                assign={assign}
                setCol={setCol}
                onRecheck={() => runAuto(file, assign, llmConsent)}
                busy={busy}
              />
            )}
          </>
        )}

        {result && !result.ok && mode === "separate" && (
          <div className="alert alert-error" role="alert">
            {result.errors.map((e) => <div key={e}>{e}</div>)}
          </div>
        )}
        {result?.warnings?.length ? (
          <div className="alert alert-warn">{result.warnings.map((w) => <div key={w}>{w}</div>)}</div>
        ) : null}
      </main>
    </>
  );
}
