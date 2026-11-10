"use client";

import type { Assessment, Assignments, AutoResult, MappedColumn } from "@/lib/api";

const ICON: Record<string, string> = { ok: "✓", warn: "!", fail: "✕", skip: "–" };
const INPUT_TEXT: Record<string, string> = {
  ok: "Available",
  missing: "Not in this file",
  unit_unresolved: "Unit not established",
  ambiguous: "Ambiguous",
};
const SRC: Record<string, string> = {
  deterministic_alias: "dataset column name",
  dataset_profile: "dataset profile",
  user_confirmed: "your input",
  llm_proposed: "AI suggestion",
  none: "—",
};
const USTAT: Record<string, string> = {
  dataset_spec: "known (dataset definition)",
  profile: "stated by dataset profile",
  user_confirmed: "confirmed by you",
  profile_hypothesis: "profile hypothesis — not applied until verified",
  header_hint: "guessed from name (not used)",
  llm_stated: "AI-stated (not trusted)",
  unknown: "unknown",
};

/** Compact schema assessment. Every line comes from the backend's real validation results. */
export function AssessmentSummary({ a }: { a: Assessment }) {
  const tone = a.outcome === "ready" ? "alert-ok" : a.outcome === "needs_input" ? "alert-warn" : "alert-error";
  return (
    <section className="section" aria-labelledby="assess-h">
      <div className={`alert ${tone}`} role={a.outcome === "ready" ? "status" : "alert"}>
        <strong id="assess-h">{a.headline}</strong>
      </div>
      {a.format_summary && (
        <>
          <h3>Dataset format recognised</h3>
          <ul className="checklist">
            {a.format_summary.map((f) => (
              <li key={f.label} className={f.status === "ok" ? "check-ok" : f.status === "fail" ? "check-fail" : f.status === "info" ? "" : "check-warn"}>
                <span className="check-icon" aria-hidden="true">
                  {f.status === "ok" ? "✓" : f.status === "fail" ? "✕" : f.status === "info" ? "•" : "?"}
                </span>
                <span>
                  <strong>{f.label}:</strong> {f.text}
                </span>
              </li>
            ))}
          </ul>
          <p className="muted small">
            “?” marks a hypothesis that is shown but not applied until it is verified; an unverified fact is never
            treated as validated.
          </p>
        </>
      )}
      <ul className="checklist">
        {a.checks.map((c) => (
          <li key={c.id} className={`check-${c.status}`}>
            <span className="check-icon" aria-hidden="true">{ICON[c.status]}</span>
            <span className="sr-only">{c.status}: </span>
            {c.text}
          </li>
        ))}
      </ul>
      {a.outcome !== "ready" && !a.format_summary && a.detector_inputs.length > 0 && (
        <>
          <h3>What the deployed detector needs</h3>
          <ul className="checklist">
            {a.detector_inputs.map((g) => (
              <li key={g.key} className={g.status === "ok" ? "check-ok" : g.role !== "scored" ? "check-warn" : "check-fail"}>
                <span className="check-icon" aria-hidden="true">{g.status === "ok" ? "✓" : "✕"}</span>
                {g.label}: {INPUT_TEXT[g.status]}
                {g.role !== "scored" && g.status !== "ok" ? (g.role === "optional" ? " (optional for this layout, not scored)" : " (needed by the pipeline, not scored)") : ""}
              </li>
            ))}
          </ul>
        </>
      )}
      {a.needed.length > 0 && (
        <>
          <h3>What is missing</h3>
          <ul>{a.needed.map((n) => <li key={n}>{n}</li>)}</ul>
        </>
      )}
      {a.next_steps && a.next_steps.length > 0 && (
        <>
          <h3>Next steps</h3>
          <ol>
            {a.next_steps.map((n) => (
              <li key={n.kind}>
                <strong>
                  {n.kind === "confirm" ? "Confirm (data supports it)" : n.kind === "tilt_test" ? "Needs a physical check" : n.kind === "document" ? "Needs documentation" : "Record in the log"}
                </strong>
                : {n.text}
              </li>
            ))}
          </ol>
        </>
      )}
      {a.pipeline_note && <p className="muted small">{a.pipeline_note}</p>}
      {a.profile && (
        <p className="muted small">
          Layout profile <span className="mono">{a.profile.id}</span>: {a.profile.documentation_status}
        </p>
      )}
    </section>
  );
}

/** Only unresolved facts are asked for: one unit selector per column, nothing else. */
export function UnitQuestions({
  a,
  targets,
  assign,
  onChange,
  onApply,
  busy,
}: {
  a: Assessment;
  targets: NonNullable<AutoResult["targets"]>;
  assign: Assignments;
  onChange: (column: string, canonical: string, unit: string) => void;
  onApply: () => void;
  busy: boolean;
}) {
  const TIME = ["s", "ms", "us", "ns"];
  if (!a.unresolved.length) return null;
  return (
    <section className="section" aria-labelledby="q-h">
      <h3 id="q-h">Please confirm</h3>
      <p className="muted small">
        These facts cannot be established from the file or a dataset definition. Everything else was decided automatically.
      </p>
      {a.unresolved.map((u) => {
        const t = targets.find((x) => x.canonical === u.canonical);
        const opts = u.canonical === "__time__" ? TIME : t?.accepted_units ?? [];
        return (
          <div className="file-field" key={u.column}>
            <label htmlFor={`u-${u.column}`}>
              <span className="mono">{u.column}</span> — {u.label}. {u.detail}
            </label>
            <select
              id={`u-${u.column}`}
              value={assign[u.column]?.unit ?? ""}
              onChange={(e) => onChange(u.column, u.canonical, e.target.value)}
            >
              <option value="">Choose the unit…</option>
              {opts.map((o) => <option key={o}>{o}</option>)}
            </select>
          </div>
        );
      })}
      <button
        className="btn btn-primary"
        disabled={busy || a.unresolved.some((u) => !assign[u.column]?.unit)}
        onClick={onApply}
      >
        {busy ? "Analysing…" : "Apply and analyse"}
      </button>
    </section>
  );
}

/** Full mapping for engineers; also where an automatic decision can be corrected. */
export function MappingDetails({
  cols,
  targets,
  assign,
  setCol,
  onRecheck,
  busy,
}: {
  cols: MappedColumn[];
  targets: NonNullable<AutoResult["targets"]>;
  assign: Assignments;
  setCol: (name: string, patch: Partial<{ canonical: string; unit: string }>) => void;
  onRecheck: () => void;
  busy: boolean;
}) {
  return (
    <details className="appendix">
      <summary>Advanced: full column mapping and corrections</summary>
      <div>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Source column</th><th>Meaning</th><th>Source unit → canonical</th><th>Unit evidence</th>
                <th>Mapped by</th><th>Status</th>
              </tr>
            </thead>
            <tbody>
              {cols.map((c) => {
                const cur = assign[c.name];
                const t = targets.find((x) => x.canonical === (cur?.canonical || c.canonical));
                return (
                  <tr key={c.name}>
                    <td className="mono">{c.name}</td>
                    <td>
                      {c.role === "time" ? (
                        "Time"
                      ) : c.status === "ignored" ? (
                        <span className="muted">Not used by the detector</span>
                      ) : (
                        <select
                          aria-label={`Meaning of ${c.name}`}
                          value={cur?.canonical ?? c.canonical ?? ""}
                          onChange={(e) => setCol(c.name, { canonical: e.target.value })}
                        >
                          <option value="">Not used</option>
                          {targets.map((x) => <option key={x.canonical} value={x.canonical}>{x.label}</option>)}
                        </select>
                      )}
                      {c.ambiguity && <div className="small muted">{c.ambiguity}</div>}
                      {c.handling && c.status === "ignored" && <div className="small muted">{c.handling}</div>}
                      {c.unit_hypothesis && c.status !== "recognized" && (
                        <div className="small muted">
                          {c.evidence_level === "supported" ? "Supported by measurements, awaiting confirmation" : "Hypothesis, unverified"}:{" "}
                          {c.unit_hypothesis}{c.evidence ? ` — ${c.evidence}` : ""}
                        </div>
                      )}
                    </td>
                    <td>
                      {c.role === "signal" || c.role === "time" ? (
                        <>
                          <input
                            type="text"
                            aria-label={`Unit of ${c.name}`}
                            placeholder={c.unit ?? "unit"}
                            value={cur?.unit ?? ""}
                            onChange={(e) => setCol(c.name, { unit: e.target.value, canonical: cur?.canonical ?? c.canonical ?? "" })}
                            style={{ width: "6rem" }}
                          />{" "}
                          <span className="small muted">
                            {c.unit ?? "?"} → {c.expected_unit ?? t?.expected_unit ?? "?"}
                            {c.conversion ? " (converted)" : ""}
                          </span>
                        </>
                      ) : "—"}
                    </td>
                    <td className="small">{USTAT[c.unit_status] ?? "—"}</td>
                    <td className="small">{SRC[c.method] ?? c.method}</td>
                    <td><span className="chip">{c.status.replace("_", " ")}</span></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <button className="btn" onClick={onRecheck} disabled={busy} style={{ marginTop: "0.75rem" }}>
          {busy ? "Checking…" : "Re-check with my corrections"}
        </button>
      </div>
    </details>
  );
}
