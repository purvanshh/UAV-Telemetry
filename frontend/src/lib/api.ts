const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

async function parseError(res: Response): Promise<never> {
  let detail = res.statusText;
  try {
    const data = await res.json();
    if (typeof data?.detail === "string") detail = data.detail;
    else if (Array.isArray(data?.detail))
      detail = data.detail.map((d: { msg?: string }) => d.msg || JSON.stringify(d)).join("; ");
    else if (data?.detail) detail = JSON.stringify(data.detail);
  } catch {
    /* ignore */
  }
  throw new ApiError(res.status, detail);
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { cache: "no-store" });
  if (!res.ok) await parseError(res);
  return res.json() as Promise<T>;
}

async function sendJson<T>(
  path: string,
  method: string,
  body?: unknown
): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    cache: "no-store",
  });
  if (!res.ok) await parseError(res);
  return res.json() as Promise<T>;
}

// ---- Agentic Analysis ------------------------------------------------------------------------------------------

export type Severity = "critical" | "warning";

export type Evidence = {
  topic: string;
  field: string;
  instance: number;
  stat: string;
  value: number | null;
  unit: string | null;
  t?: number | null;
  at?: string;
  [k: string]: unknown;
};

export type Finding = {
  id: string;
  code: string;
  severity: Severity;
  title: string;
  t_start: number | null;
  t_end: number | null;
  when: string;
  window: "interval" | "whole_flight";
  detector: string;
  confidence: "high" | "medium" | "low";
  basis: string;
  evidence: Evidence[];
};

export type Coverage = Record<string, { status: "evaluated" | "partial" | "not_evaluable"; checks: string[]; reasons: string[] }>;

export type Segment = { value: string; t_start: number; t_end: number; when: string };

export type ArjunaSummary = {
  kind: "ulog";
  analysis_version: string;
  refused: boolean;
  file: { name: string; sha256: string; size_bytes: number; folder_files?: string[] };
  identity: Record<string, string | number | undefined>;
  compatibility: { ok: boolean; problems: string[]; notes: string[]; schema_version: string | null; log_profile: string | null };
  integrity: { result: string; label?: string; detail?: string; converted_log?: boolean; blocks_checked?: number; block_count?: number };
  classification?: { kind: "flight" | "ground_only" | "bench_test"; reason: string; airborne_s: number };
  timeline?: {
    duration_s: number;
    airborne_s: number;
    air_intervals: { t_start: number; t_end: number; when: string }[];
    takeoff_t?: number;
    landing_t?: number;
    modes?: Segment[];
    events?: { t: number; at: string; event: string; severity: string; category: string; mode_before?: string; mode_after?: string }[];
    dropouts?: { t: number; duration_ms: number }[];
    important_messages?: { t: number; at: string; level: string; text: string }[];
    message_counts?: Record<string, number>;
  };
  topics?: {
    topics: { topic: string; instance: number; samples: number; rate_hz: number | null; fields: string[]; unavailable_fields: string[] }[];
    missing_standard_topics: string[];
    topics_not_produced: string[];
  };
  findings: Finding[];
  coverage?: Coverage;
  metrics?: Record<string, unknown>;
  physics?: {
    hover_balance?: {
      status: string;
      reason?: string;
      steady_hover_s?: number;
      motors?: string[];
      mean_cmd?: number[];
      deviation_pct?: number[];
      worst_motor?: string;
      worst_deviation_pct?: number;
      interpretation?: string;
      flag?: boolean;
    };
    thrust_residual?: {
      status: string;
      reason?: string;
      model?: string;
      voltage_compensation?: string;
      residual_robust_spread_mps2?: number;
      sustained_departures?: { t_start: number; t_end: number; when: string; direction: string; median_residual_mps2: number }[];
      threshold_note?: string;
    };
    dynamics?: { status: string; reason?: string; rate_hz?: number; eta?: number[]; relative_to_others?: number[]; motors?: string[]; flag_motor?: string | null; method?: string };
  };
  thresholds?: Record<string, number>;
  fault_codes?: Record<string, string>;
  verdict?: { worst: Severity | null; critical: number; warning: number; codes: string[] };
  metadata?: Record<string, unknown> | null;
  conversion?: Conversion;
  pilot?: PilotView;
  engineer_labels?: {
    reviewed?: boolean;
    reviewed_by?: string;
    method?: string;
    split?: string;
    labels: { code: string; severity: string; t_start: number; t_end: number; when: string; note?: string; source?: string }[];
  } | null;
};

export type PilotProblem = {
  finding_id: string;
  severity: Severity;
  code: string;
  name: string;
  headline: string;
  when: string;
  meaning: string;
  action: string;
  likely_false_alarm: boolean;
  confidence: string;
  also: string[];
};

export type PilotView = {
  status: "critical" | "warning" | "clear" | "not_a_flight" | "unreadable";
  status_text: string;
  status_detail: string;
  problems: PilotProblem[];
  critical_checks: string[];
  checklist: string[];
  not_checked: string[];
  integrity_warning?: string | null;
};

export type Conversion = {
  converter_version: string;
  source_file: string;
  format: { id: string; label: string; basis: string };
  time: { column: string; unit: string; how: string; duration_s: number; sample_rate_hz: number | null };
  rows_in: number;
  rows_written: number;
  rows_without_valid_time: number;
  duplicate_timestamps_dropped: number;
  mapped: { from: string; to: string[]; unit_in: string; conversion: string }[];
  derived: { record: string; how: string; caveat: string }[];
  dropped: { column: string; reason: string }[];
  assumptions: string[];
  air_rule: string;
  min_airborne_s: number;
  motor_layout: string;
  topics_written: string[];
};

export type FindingReview = {
  finding_id: string;
  assessment: "confirmed" | "likely_false_alarm" | "needs_review";
  explanation: string;
  evidence_refs: string[];
  suggested_cause: string;
  check_next: string;
  reviewed_by_agent: boolean;
};

export type AgentTrace = { ref: string; tool: string; args: Record<string, unknown>; ok: boolean; chars: number; error?: string | null };

export type AgentRun = {
  status: "ok" | "partial" | "error" | "unavailable" | "incomplete";
  label: string;
  message: string;
  provenance: {
    model: string;
    prompt_version: string;
    rounds?: number;
    tool_calls?: number;
    latency_ms?: number;
    tokens?: { prompt: number; completion: number };
    finished_at?: string;
    pilot_report?: PilotInput | null;
  };
  failure?: { reason: string; detail?: string } | null;
  trace: AgentTrace[];
  report: {
    headline: string;
    summary: string;
    finding_reviews: FindingReview[];
    root_cause_chain: { finding_id: string; role: "root_cause" | "consequence" | "independent"; link: string }[];
    additional_observations: {
      title: string;
      detail: string;
      evidence_refs: string[];
      t_start: number | null;
      t_end: number | null;
      suggested_code: string | null;
    }[];
    pilot_report_answer: string;
    not_checked: string[];
    recommendations: { action: string; reason: string; finding_ids: string[] }[];
    rejected: { where: string; reason: string; text: string }[];
    unreviewed_findings: string[];
  } | null;
};

export type PilotInput = {
  symptom: string;
  symptom_label?: string;
  time_ref: string | null;
  time_value: number | string | null;
  note: string;
};

export type AnalysisMeta = {
  analysis_id: string;
  created_at: string;
  filename: string;
  kind: "ulog";
  converted_from?: "csv" | null;
  source_profile?: string | null;
  flight_id?: string | null;
  vehicle_id?: string | null;
  refused?: boolean;
  integrity?: string | null;
  classification?: string | null;
  duration_s?: number | null;
  airborne_s?: number | null;
  worst: Severity | null;
  critical?: number;
  warning?: number;
  codes?: string[];
  pilot_report?: PilotInput | null;
  agent?: { status: string; at: string; message?: string; model?: string } | null;
};

export type AnalysisView = {
  analysis_id: string;
  meta: AnalysisMeta;
  summary: ArjunaSummary;
  report: AgentRun | null;
  label: string;
  consent_text: string;
  symptoms: Record<string, string>;
  current_version: string;
  last_attempt?: AgentRun;
};

export type ChartPack = {
  finding_id?: string;
  window: [number, number];
  highlight?: [number, number];
  time_basis: string;
  charts: { title: string; unit: string | null; series: { label: string; t: number[]; y: number[] }[] }[];
};

export type AiStatus = { configured: boolean; model: string };

export type PilotForm = { symptom: string; time_ref: string; time_value: string; note: string };

export const api = {
  health: () => getJson<{ status: string }>("/api/health"),
  aiStatus: () => getJson<AiStatus>("/api/ai/status"),
  list: () => getJson<AnalysisMeta[]>("/api/analyses"),
  get: (id: string) => getJson<AnalysisView>(`/api/analyses/${encodeURIComponent(id)}`),
  async create(file: File, pilot?: PilotForm | null): Promise<AnalysisView> {
    const fd = new FormData();
    fd.append("file", file);
    if (pilot?.symptom) {
      fd.append("symptom", pilot.symptom);
      if (pilot.time_ref) fd.append("time_ref", pilot.time_ref);
      if (pilot.time_value) fd.append("time_value", pilot.time_value);
      if (pilot.note) fd.append("note", pilot.note);
    }
    const res = await fetch(`${API_URL}/api/analyses`, { method: "POST", body: fd });
    if (!res.ok) await parseError(res);
    return res.json() as Promise<AnalysisView>;
  },
  runAgent: (id: string) => sendJson<AnalysisView>(`/api/analyses/${encodeURIComponent(id)}/agent`, "POST", { consent: true }),
  findingContext: (id: string, fid: string) =>
    getJson<ChartPack>(`/api/analyses/${encodeURIComponent(id)}/findings/${encodeURIComponent(fid)}/context`),
  series: (id: string, sigs: string[], t0?: number | null, t1?: number | null) => {
    const q = new URLSearchParams();
    sigs.forEach((s) => q.append("sig", s));
    if (t0 != null) q.set("t_start", String(t0));
    if (t1 != null) q.set("t_end", String(t1));
    return getJson<ChartPack>(`/api/analyses/${encodeURIComponent(id)}/series?${q}`);
  },
  downloadUrl: (id: string, what: "ulg" | "original") =>
    `${API_URL}/api/analyses/${encodeURIComponent(id)}/download/${what}`,
  exportUrl: (id: string) => `${API_URL}/api/analyses/${encodeURIComponent(id)}/export`,
  remove: (id: string) => sendJson<{ status: string }>(`/api/analyses/${encodeURIComponent(id)}`, "DELETE"),
};
