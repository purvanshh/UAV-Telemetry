"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  ArrowDownToLine,
  ArrowRight,
  Check,
  CircleCheck,
  CircleHelp,
  ClipboardCheck,
  Flame,
  Drone,
  Repeat2,
  RotateCw,
  ScanSearch,
  ShieldAlert,
  Timer,
  Vibrate,
  Waves,
  Wind,
  type LucideIcon,
} from "lucide-react";
import { DroneProcessing } from "@/components/drone/DroneProcessing";
import { Dropzone } from "@/components/Dropzone";
import { Alert, Button, Card, PageHeader, Segmented, Switch } from "@/components/ui";
import { api, ApiError, AiStatus, PilotForm } from "@/lib/api";

const SYMPTOMS: { id: string; label: string; icon: LucideIcon }[] = [
  { id: "wobble", label: "Wobbled", icon: Waves },
  { id: "drift", label: "Drifted", icon: Wind },
  { id: "sudden_drop", label: "Dropped suddenly", icon: ArrowDownToLine },
  { id: "yaw_spin", label: "Spun around", icon: RotateCw },
  { id: "toilet_bowl", label: "Circled in place", icon: Repeat2 },
  { id: "slow_response", label: "Slow to respond", icon: Timer },
  { id: "failsafe", label: "Failsafe triggered", icon: ShieldAlert },
  { id: "crash", label: "Crashed", icon: Flame },
  { id: "motor_sound", label: "Odd motor sound or smell", icon: Vibrate },
  { id: "other", label: "Something else", icon: CircleHelp },
];

const WHEN: { value: string; label: string }[] = [
  { value: "since_takeoff", label: "After take-off" },
  { value: "before_landing", label: "Before landing" },
  { value: "phase", label: "During…" },
];
const PHASES = [
  { id: "takeoff", label: "Take-off" },
  { id: "climb", label: "Climb" },
  { id: "cruise", label: "Cruise" },
  { id: "landing", label: "Landing" },
];

type Stage = "upload" | "checks" | "open";

export default function NewAnalysisPage() {
  const router = useRouter();
  const [purpose, setPurpose] = useState<"flight" | "preflight">("flight");
  const [file, setFile] = useState<File | null>(null);
  const [pilot, setPilot] = useState<PilotForm>({ symptom: "", time_ref: "since_takeoff", time_value: "", note: "" });
  const [review, setReview] = useState(true);
  const [ai, setAi] = useState<AiStatus | null>(null);
  const [stage, setStage] = useState<Stage | null>(null);
  const [sent, setSent] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.aiStatus().then(setAi).catch(() => setAi({ configured: false, model: "" }));
  }, []);

  const busy = stage !== null;
  const isFlight = purpose === "flight";

  async function run() {
    if (!file) return;
    setError(null);
    setSent(0);
    setStage("upload");
    try {
      const v = await api.create(file, isFlight && pilot.symptom ? pilot : null, purpose, (p) => {
        setSent(p);
        if (p >= 1) setStage("checks");
      });
      setStage("open");
      const agent = isFlight && review && ai?.configured && !v.summary.refused;
      router.push(`/analysis/${encodeURIComponent(v.analysis_id)}${agent ? "?agent=1" : ""}`);
    } catch (e) {
      setError(e instanceof ApiError || e instanceof Error ? e.message : "The upload failed.");
      setStage(null);
    }
  }

  const pick = (id: string) => setPilot((p) => ({ ...p, symptom: p.symptom === id ? "" : id, time_value: "" }));

  return (
    <main className="page page-narrow">
      <PageHeader
        title="Analyse a flight"
        lede="Upload the UAV flight log. You get a clear verdict, the problems in plain words, and what to check before the next flight."
      />

      <div className="stack" style={{ ["--gap" as string]: "1rem" }}>
        <Card title="1 · What is this recording?">
          <div className="choice-grid" role="radiogroup" aria-label="What is this recording?">
            {(
              [
                { id: "flight", icon: Drone, title: "A drone flight", desc: "See what happened and what to check." },
                { id: "preflight", icon: ClipboardCheck, title: "A pre-flight check", desc: "A short test flight or bench run: is it ready for a mission?" },
              ] as const
            ).map((c) => (
              <button
                key={c.id}
                type="button"
                role="radio"
                aria-checked={purpose === c.id}
                className="choice"
                disabled={busy}
                onClick={() => setPurpose(c.id)}
              >
                <span className="choice-icon" aria-hidden="true">
                  <c.icon size={19} />
                </span>
                <span className="choice-title">{c.title}</span>
                <span className="choice-desc">{c.desc}</span>
                <CircleCheck className="choice-check" size={18} aria-hidden="true" />
              </button>
            ))}
          </div>
        </Card>

        <Card title="2 · The flight log">
          <Dropzone
            accept=".ulg,.zip,.csv,application/zip,text/csv"
            formats={["flight.ulg", "flight folder .zip", "telemetry .csv"]}
            file={file}
            onFile={(f) => {
              setFile(f);
              setError(null);
            }}
            disabled={busy}
          />
          <p className="tiny muted" style={{ marginTop: "0.6rem" }}>
            A CSV is converted to the standard ARJUNA format first; the report shows exactly how.
          </p>
        </Card>

        {isFlight && (
          <Card
            title="3 · Anything you noticed?"
            actions={<span className="tiny muted">Optional</span>}
          >
            <p className="small muted" style={{ marginBottom: "0.85rem" }}>
              Tap what you saw. The report will answer it directly.
            </p>
            <div className="chips" role="group" aria-label="What you noticed">
              {SYMPTOMS.map((s) => (
                <button key={s.id} type="button" className="chip-btn" aria-pressed={pilot.symptom === s.id} disabled={busy} onClick={() => pick(s.id)}>
                  {pilot.symptom === s.id ? <Check aria-hidden="true" /> : <s.icon aria-hidden="true" />}
                  {s.label}
                </button>
              ))}
            </div>

            <div className={`disclosure-body${pilot.symptom ? " is-open" : ""}`} inert={!pilot.symptom}>
              <div>
                <div className="stack" style={{ ["--gap" as string]: "0.9rem", paddingTop: "1.1rem" }}>
                  <div className="field">
                    <span className="field-label">About when?</span>
                    <div className="row">
                      <Segmented
                        label="When"
                        value={pilot.time_ref}
                        onChange={(v) => setPilot((p) => ({ ...p, time_ref: v, time_value: "" }))}
                        options={WHEN}
                      />
                      {pilot.time_ref === "phase" ? (
                        <div className="chips">
                          {PHASES.map((ph) => (
                            <button
                              key={ph.id}
                              type="button"
                              className="chip-btn"
                              aria-pressed={pilot.time_value === ph.id}
                              onClick={() => setPilot((p) => ({ ...p, time_value: ph.id }))}
                            >
                              {ph.label}
                            </button>
                          ))}
                        </div>
                      ) : (
                        <label className="row" style={{ gap: "0.45rem" }}>
                          <input
                            className="input"
                            style={{ width: 96 }}
                            type="number"
                            inputMode="numeric"
                            min={0}
                            step={1}
                            placeholder="e.g. 90"
                            value={pilot.time_value}
                            onChange={(e) => setPilot((p) => ({ ...p, time_value: e.target.value }))}
                          />
                          <span className="small muted">seconds</span>
                        </label>
                      )}
                    </div>
                    <span className="field-hint">A rough guess is fine; it looks about 30 seconds either side.</span>
                  </div>
                  <label className="field">
                    <span className="field-label">Note</span>
                    <textarea
                      className="textarea"
                      rows={2}
                      maxLength={1000}
                      value={pilot.note}
                      placeholder="e.g. it wobbled, then flipped on landing"
                      onChange={(e) => setPilot((p) => ({ ...p, note: e.target.value }))}
                    />
                  </label>
                </div>
              </div>
            </div>
          </Card>
        )}

        {isFlight && ai?.configured && (
          <Card>
            <div className="switch-row">
              <div>
                <div className="row" style={{ gap: "0.45rem" }}>
                  <ScanSearch size={17} aria-hidden="true" className="muted" />
                  <strong>Detailed AI review</strong>
                </div>
                <p className="small muted" style={{ marginTop: "0.25rem" }}>
                  Explains what happened and why, in about a minute. Only computed results are sent to {ai.model}; the raw
                  log and GPS position never leave the server.
                </p>
              </div>
              <Switch checked={review} onChange={setReview} label="Detailed AI review" disabled={busy} />
            </div>
          </Card>
        )}

        {error && (
          <Alert tone="warn" title="That didn't work.">
            {error}
          </Alert>
        )}

        {busy ? (
          <Card>
            <div style={{ marginBottom: "1.1rem" }}>
              <DroneProcessing
                title={
                  stage === "upload"
                    ? `Uploading the flight log · ${Math.round(sent * 100)}%`
                    : stage === "checks"
                      ? purpose === "preflight"
                        ? "Running the pre-flight checks…"
                        : "Reading the telemetry and running every check…"
                      : "Opening the report…"
                }
              />
            </div>
            <div className="steps">
              <Step label={stage === "upload" ? `Uploading the log · ${Math.round(sent * 100)}%` : "Log uploaded"} state={stage === "upload" ? "active" : "done"} />
              {stage === "upload" && (
                <div className="progress" style={{ marginLeft: 34 }}>
                  <span style={{ width: `${Math.round(sent * 100)}%` }} />
                </div>
              )}
              <Step
                label={purpose === "preflight" ? "Running the pre-flight checks" : "Reading the flight and running every check"}
                state={stage === "checks" ? "active" : stage === "open" ? "done" : "waiting"}
              />
              <Step label="Opening the report" state={stage === "open" ? "active" : "waiting"} />
            </div>
          </Card>
        ) : (
          <Button variant="primary" size="lg" block icon={file ? ArrowRight : undefined} disabled={!file} onClick={() => void run()}>
            {!file ? "Add a flight log to continue" : isFlight ? "Analyse this flight" : "Run the pre-flight check"}
          </Button>
        )}
      </div>
    </main>
  );
}

function Step({ label, state }: { label: string; state: "done" | "active" | "waiting" }) {
  return (
    <div className={`step${state === "active" ? " is-active" : state === "done" ? " is-done" : ""}`}>
      <span className="step-dot" aria-hidden="true">
        {state === "done" ? <Check /> : state === "active" ? <span className="spinner" style={{ width: 11, height: 11 }} /> : null}
      </span>
      <span>{label}</span>
    </div>
  );
}
