import type { AnalysisMeta } from "@/lib/api";
import { PREFLIGHT_TONE, type Tone } from "@/lib/status";

/** The one-line verdict for a stored analysis, as shown in lists and pickers. Never says "safe". */
export function metaVerdict(m: AnalysisMeta): { tone: Tone; text: string } {
  if (m.refused) return { tone: "neutral", text: "Could not read this log" };
  if (m.purpose === "preflight") {
    const text: Record<string, string> = {
      ready: "Ready for a mission",
      check: "Check before the mission",
      not_ready: "Not ready for a mission",
      partial: "Cannot tell",
    };
    const s = m.preflight_status ?? "";
    return { tone: PREFLIGHT_TONE[s] ?? "neutral", text: text[s] ?? "Pre-flight check" };
  }
  if (m.classification && m.classification !== "flight") return { tone: "info", text: "Not a full flight" };
  if (m.worst === "critical") return { tone: "warn", text: "Do not fly until checked" };
  if (m.worst === "warning") return { tone: "caution", text: "Check before next flight" };
  return { tone: "go", text: "No problems found" };
}

export function metaKind(m: AnalysisMeta): string {
  if (m.purpose === "preflight") return "Pre-flight check";
  return m.converted_from === "csv" ? "CSV, converted" : "ARJUNA log";
}

export function metaTitle(m: AnalysisMeta): string {
  return m.converted_from === "csv" ? m.filename : m.flight_id ?? m.filename;
}
