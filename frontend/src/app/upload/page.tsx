"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { AppNav } from "@/components/AppNav";
import { Dropzone } from "@/components/Dropzone";
import { api, ApiError, AiStatus, PilotForm } from "@/lib/api";

const SYMPTOMS: [string, string][] = [
  ["", "Nothing in particular"],
  ["wobble", "Wobble or oscillation"],
  ["drift", "Drift (would not hold position)"],
  ["sudden_drop", "Sudden drop in height"],
  ["yaw_spin", "Yaw spin"],
  ["toilet_bowl", "Toilet-bowling (circling)"],
  ["slow_response", "Slow or weak response to sticks"],
  ["failsafe", "Failsafe triggered"],
  ["crash", "Crash"],
  ["motor_sound", "Motor sound or smell"],
  ["other", "Something else"],
];

export default function NewAnalysisPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [pilot, setPilot] = useState<PilotForm>({ symptom: "", time_ref: "since_takeoff", time_value: "", note: "" });
  const [consent, setConsent] = useState(true);
  const [ai, setAi] = useState<AiStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.aiStatus().then(setAi).catch(() => setAi({ configured: false, model: "" }));
  }, []);

  async function run() {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const v = await api.create(file, pilot.symptom ? pilot : null);
      const agent = consent && ai?.configured && !v.summary.refused;
      router.push(`/analysis/${encodeURIComponent(v.analysis_id)}${agent ? "?agent=1" : ""}`);
    } catch (e) {
      setError(e instanceof ApiError || e instanceof Error ? e.message : "Upload failed");
      setBusy(false);
    }
  }

  const set = (k: keyof PilotForm) => (e: { target: { value: string } }) => setPilot({ ...pilot, [k]: e.target.value });

  return (
    <>
      <AppNav />
      <main>
        <div className="page-head">
          <h1>Analyse a new flight</h1>
          <p className="sub">Agentic Analysis: code checks every ARJUNA fault code, then an AI agent investigates.</p>
        </div>
        <div style={{ maxWidth: "42rem" }}>
          <Dropzone
            label="Flight log"
            hint="An ARJUNA flight.ulg, a zipped flight folder (F00042_…/flight.ulg with metadata.json), or a telemetry CSV. A CSV is first converted to a standard ARJUNA log; the conversion report shows every mapping and assumption."
            accept=".ulg,.zip,.csv,application/zip,text/csv"
            file={file}
            onFile={(f) => {
              setFile(f);
              setError(null);
            }}
            disabled={busy}
          />

          <fieldset className="pilot-form" disabled={busy}>
            <legend>What did you see? (optional)</legend>
            <p className="small muted">
              The agent ranks every finding against your report and answers it directly. A rough time is fine; it searches
              about 30 s either side.
            </p>
            <label>
              Symptom
              <select value={pilot.symptom} onChange={set("symptom")}>
                {SYMPTOMS.map(([k, label]) => (
                  <option key={k} value={k}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            {pilot.symptom && (
              <>
                <div className="row">
                  <label>
                    When
                    <select value={pilot.time_ref} onChange={set("time_ref")}>
                      <option value="since_takeoff">seconds after take-off</option>
                      <option value="since_log_start">seconds after the log started</option>
                      <option value="before_landing">seconds before landing</option>
                      <option value="phase">during a phase</option>
                    </select>
                  </label>
                  {pilot.time_ref === "phase" ? (
                    <label>
                      Phase
                      <select value={pilot.time_value} onChange={set("time_value")}>
                        <option value="">choose</option>
                        <option value="takeoff">take-off</option>
                        <option value="climb">climb</option>
                        <option value="cruise">cruise</option>
                        <option value="landing">landing</option>
                      </select>
                    </label>
                  ) : (
                    <label>
                      Seconds
                      <input type="number" min={0} step={1} value={pilot.time_value} onChange={set("time_value")} placeholder="e.g. 120" />
                    </label>
                  )}
                </div>
                <label>
                  Note
                  <textarea rows={2} maxLength={1000} value={pilot.note} onChange={set("note")} placeholder="e.g. it wobbled, then flipped on landing" />
                </label>
              </>
            )}
          </fieldset>

          {ai?.configured ? (
            <label className="checkline">
              <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} disabled={busy} />
              <span>
                <strong>Run the agentic investigation ({ai.model}).</strong> The results of code tools (statistics, times,
                findings) are sent to OpenAI. The raw log, GPS coordinates and engineer labels never leave the server.
                Without this you still get every code finding.
              </span>
            </label>
          ) : (
            ai && <p className="small muted">The AI agent is not configured on the server; you will get the code findings only.</p>
          )}

          {error && <div className="alert alert-error" role="alert">{error}</div>}
          <button className="btn btn-primary btn-lg" disabled={!file || busy} onClick={run}>
            {busy ? "Reading the log and running the checks…" : file ? "Analyse this flight" : "Add a file to continue"}
          </button>
        </div>
      </main>
    </>
  );
}
