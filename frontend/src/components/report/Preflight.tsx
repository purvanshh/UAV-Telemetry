import { ArrowRight, ClipboardCheck } from "lucide-react";
import { Card, Pill } from "@/components/ui";
import type { Preflight } from "@/lib/api";
import { CHECK_ICON, CHECK_TONE, CHECK_WORD, toneClass } from "@/lib/status";

const ORDER = ["pass", "check", "fail", "unknown"] as const;
const LEGEND: Record<string, string> = { pass: "OK", check: "to check", fail: "problem(s)", unknown: "not recorded" };
const COLOR: Record<string, string> = { pass: "var(--go)", check: "var(--caution)", fail: "var(--warn)", unknown: "var(--line-strong)" };

/** The go / no-go list of a pre-flight check. */
export function PreflightChecks({ pf }: { pf: Preflight }) {
  const n = Object.fromEntries(ORDER.map((s) => [s, pf.checks.filter((c) => c.status === s).length])) as Record<string, number>;
  return (
    <Card
      title="Pre-flight checks"
      icon={ClipboardCheck}
      pad={false}
      actions={<span className="tiny muted">{pf.checks.length} checks</span>}
    >
      <div className="card-body" style={{ paddingBottom: "0.9rem" }}>
        <div className="pf-summary" role="img" aria-label={ORDER.map((s) => `${n[s]} ${CHECK_WORD[s]}`).join(", ")}>
          {ORDER.filter((s) => n[s]).map((s) => (
            <span key={s} style={{ flexGrow: n[s], background: COLOR[s] }} />
          ))}
        </div>
        <div className="pf-legend">
          {ORDER.filter((s) => n[s]).map((s) => (
            <span key={s}>
              <i style={{ background: COLOR[s] }} />
              {n[s]} {LEGEND[s]}
            </span>
          ))}
        </div>
      </div>
      <ul className="plain pf-list" style={{ borderTop: "1px solid var(--line)" }}>
        {pf.checks.map((c, i) => {
          const tone = CHECK_TONE[c.status];
          const Icon = CHECK_ICON[c.status];
          return (
            <li key={c.id} className={`pf-item ${toneClass(tone)}${c.status === "fail" ? " is-fail" : ""} rise`} style={{ ["--i" as string]: i }}>
              <span className="pf-icon" aria-hidden="true">
                <Icon />
              </span>
              <div style={{ minWidth: 0 }}>
                <div className="pf-label">{c.label}</div>
                <div className="pf-detail">{c.detail}</div>
                {c.action && (c.status === "fail" || c.status === "check") && (
                  <div className="pf-action">
                    <ArrowRight size={14} aria-hidden="true" /> {c.action}
                  </div>
                )}
                {c.provisional && <div className="tiny muted" style={{ marginTop: "0.25rem" }}>Limit not yet approved by the engineering team.</div>}
              </div>
              <Pill tone={tone}>{CHECK_WORD[c.status]}</Pill>
            </li>
          );
        })}
      </ul>
    </Card>
  );
}
