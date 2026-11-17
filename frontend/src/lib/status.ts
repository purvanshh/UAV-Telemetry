import {
  CircleCheck,
  CircleHelp,
  CircleMinus,
  Info,
  OctagonAlert,
  TriangleAlert,
  type LucideIcon,
} from "lucide-react";

/** Flight-instrument convention: go = normal, caution = amber, warn = red, info = blue, neutral = not known. */
export type Tone = "go" | "caution" | "warn" | "info" | "neutral";

export const TONE_ICON: Record<Tone, LucideIcon> = {
  go: CircleCheck,
  caution: TriangleAlert,
  warn: OctagonAlert,
  info: Info,
  neutral: CircleHelp,
};

/** One table per kind of verdict, so the same word always gets the same colour and icon everywhere. */
export const PILOT_TONE: Record<string, Tone> = {
  critical: "warn",
  warning: "caution",
  clear: "go",
  not_a_flight: "info",
  unreadable: "neutral",
};

export const PREFLIGHT_TONE: Record<string, Tone> = {
  ready: "go",
  check: "caution",
  not_ready: "warn",
  partial: "neutral",
  unreadable: "neutral",
};

export const CHECK_TONE: Record<string, Tone> = { pass: "go", check: "caution", fail: "warn", unknown: "neutral" };
export const CHECK_ICON: Record<string, LucideIcon> = {
  pass: CircleCheck,
  check: TriangleAlert,
  fail: OctagonAlert,
  unknown: CircleMinus,
};
export const CHECK_WORD: Record<string, string> = { pass: "OK", check: "Check", fail: "Problem", unknown: "Not recorded" };

export const COMPARE_TONE: Record<string, Tone> = {
  fixed: "go",
  no_problems: "go",
  partly_fixed: "caution",
  not_fixed: "warn",
  worse: "warn",
  not_comparable: "neutral",
};

export const SEVERITY_TONE: Record<string, Tone> = { critical: "warn", warning: "caution" };
export const SEVERITY_WORD: Record<string, string> = { critical: "Serious", warning: "Check" };

export function toneClass(t: Tone): string {
  return `tone-${t}`;
}
