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

export type ArtifactsStatus = {
  ready: boolean;
  missing: string[];
  model_dir: string;
  variant?: string | null;
  threshold?: number | null;
  features: string[];
  n_features?: number | null;
  win_samples?: number | null;
  stride_samples?: number | null;
  warnings: string[];
  error?: string | null;
};

export type UploadResponse = {
  ok: boolean;
  flight_id?: string | null;
  status?: string | null;
  source?: string | null;
  filenames: Record<string, string>;
  file_sizes: Record<string, number>;
  warnings: string[];
  errors: string[];
  validation?: Record<string, unknown> | null;
  duplicate_of?: string | null;
  duplicate_kind?: string | null;
};

export type UploadSummaryStream = {
  stream: string;
  role: string;
  ok: boolean;
  filename: string | null;
  file_size_bytes: number | null;
  rows: number;
  columns: number;
  errors: string[];
  warnings: string[];
  timestamp: {
    column: string;
    unit: string | null;
    unit_source: string | null;
    est_hz: number | null;
    duration_s: number | null;
    duplicate_timestamps: number;
    non_monotonic_timestamps: number;
    missing_timestamps: number;
  } | null;
  missing_required: string[];
  range_flags: string[];
};

export type UploadSummary = {
  flight_id: string;
  source: string;
  filenames: Record<string, string>;
  file_sizes: Record<string, number>;
  upload_warnings: string[];
  streams: UploadSummaryStream[];
  all_errors: string[];
  ready_for_analysis: boolean;
  hints: string[];
  analysis_note: string;
};

export type AnalyzeResponse = {
  ok: boolean;
  flight_id: string;
  status: string;
  errors: string[];
  warnings: string[];
  n_windows: number;
  n_flagged: number;
  flagged_rate?: number | null;
  n_events: number;
  threshold: number;
  variant: string;
  features: string[];
  score_mean?: number | null;
  score_p50?: number | null;
  score_p95?: number | null;
  score_max?: number | null;
  coverage_status?: string | null;
  data_quality?: Record<string, unknown> | null;
  verdict?: Verdict | null;
};

export type FlightListItem = {
  flight_id: string;
  status: string;
  source?: string | null;
  created_at?: string | null;
  filenames: Record<string, string>;
  verdict_state?: VerdictState | null;
  verdict_headline?: string | null;
};

export type VerdictState =
  | "anomalies_detected"
  | "no_alerts_adequate_coverage"
  | "no_alerts_low_coverage"
  | "analysis_failed"
  | "insufficient_data"
  | "not_analyzed";

export type Verdict = {
  state: VerdictState;
  headline: string;
  tone: "attention" | "neutral" | "caution" | "error";
  detail: string;
  caveats: string[];
  errors: string[];
};

export type OverviewEvent = {
  event_id: string;
  start_rel_s: number;
  end_rel_s: number;
  start_label: string;
  end_label: string;
  duration_s: number;
  session_id: number;
  headline: string;
  channels: string[];
  has_signal_checks: boolean;
};

export type Unassessed = {
  start_rel_s: number;
  end_rel_s: number;
  duration_s: number;
  reason: string;
  text: string;
};

export type Overview = {
  flight_id: string;
  created_at?: string | null;
  analyzed_at?: string | null;
  source?: string | null;
  filenames: Record<string, string>;
  upload_warnings: string[];
  input?: {
    filename?: string;
    rows?: number;
    profile_id?: string | null;
    mapping_version?: string;
    llm?: { status?: string; model?: string } | null;
    columns: {
      source: string;
      canonical: string | null;
      unit: string | null;
      expected_unit: string | null;
      unit_status: string | null;
      method: string | null;
      status: string | null;
      converted: boolean;
      ignored: boolean;
    }[];
  } | null;
  verdict: Verdict;
  events: OverviewEvent[];
  unassessed: Unassessed[];
  coverage: {
    status?: string | null;
    analysed_fraction?: number | null;
    known: boolean;
    scored_windows?: number | null;
    candidate_windows?: number | null;
    missing_signals: string[];
  } | null;
  duration: { flight_s?: number | null; analysed_s?: number | null; aligned_s?: number | null };
  summary_text: string;
  warnings: string[];
};

export type TimelineItem = {
  type: "event" | "unassessed";
  id: string;
  start_rel_s: number;
  end_rel_s: number;
  duration_s: number;
  session_id: number | null;
  title: string;
  channels: string[];
  signal_checks: string[];
};

export type Timeline = { time_reference: string; total_s: number | null; items: TimelineItem[] };

export type ChannelEvidence = {
  signal: string;
  label: string;
  pilot_label?: string;
  technical_name?: string;
  unit: string | null;
  during: { min: number | null; max: number | null; mean: number | null; n: number } | null;
  before: { p05: number | null; median: number | null; p95: number | null; span_s: number } | null;
};

export type EventDetail = {
  event_id: string;
  session_id: number;
  start_label: string;
  end_label: string;
  start_rel_s: number;
  end_rel_s: number;
  duration_s: number;
  what_observed: string;
  when: string;
  channels: ChannelEvidence[];
  signal_checks: { kind: string; signal: string; text: string; time_rel_s: number }[];
  signal_checks_note: string;
  detector_note: string;
  uncertain: string[];
  suggested_action: string;
  nearby_unassessed: { start_s: number; end_s: number; reason: string }[];
  flagged_window_count?: number;
  affected_signals?: string[];
};

export type EventContext = {
  event_id: string;
  start_rel_s: number;
  end_rel_s: number;
  window_start_rel_s: number;
  window_end_rel_s: number;
  time: number[];
  channels: { signal: string; label: string; technical_name?: string; unit: string | null; values: (number | null)[] }[];
  segment_limited: boolean;
};

/** Review categories. These order attention; they are not safety or severity ratings. */
export type PilotReviewLabel = "Review recommended" | "Worth checking" | "No notable change detected";

export type PilotWhen = {
  display: string;
  approximate: string;
  start_row: number | null;
  end_row: number | null;
  start_raw?: number | null;
  end_raw?: number | null;
  start_rel_s?: number | null;
  end_rel_s?: number | null;
};

export type PilotAttention = {
  event_id: string | null;
  review_label: PilotReviewLabel;
  priority: string;
  title: string;
  when: PilotWhen;
  what_happened: string;
};

export type PilotEvent = {
  event_id: string;
  priority: string;
  review_label: PilotReviewLabel;
  category: string;
  title: string;
  what_happened: string;
  when: PilotWhen;
  why_review: string;
  what_to_check: string[];
  source_rows: { start: number | null; end: number | null };
  technical_why: string;
};

export type PilotReport = {
  mode: "ai_analysis" | "deep_analysis";
  mode_label: string;
  mode_note: string;
  detection_note: string;
  summary: string;
  attention: PilotAttention[];
  key_observations: string[];
  next_steps: { event_id: string; text: string }[];
  limitation: string;
  other_intervals: { event_id: string | null; review_label: PilotReviewLabel; title: string; when: string }[];
  what_system_cannot_determine: string[];
  coverage?: {
    analysed_s: number | null;
    flight_s: number | null;
    status: string | null;
    analysed_fraction: number | null;
  };
};

export type OperatorReport = {
  overview: Overview;
  events: EventDetail[];
  pilot_report?: PilotReport;
  pilot_events?: Record<string, PilotEvent>;
  unassessed: Unassessed[];
  methodology: string;
  model_version: Record<string, unknown>;
  disclaimer: string;
  technical_appendix: Record<string, unknown>;
};

export type Explanation = {
  status: string;
  message?: string;
  label?: string;
  disclosure?: string;
  model?: string;
  configured?: boolean;
  events: {
    event_id: string;
    observed_facts: string[];
    interpretation: string[];
    uncertainty: string[];
    suggested_review: string[];
  }[];
};

export type MappedColumn = {
  name: string;
  status: "recognized" | "needs_confirmation" | "ambiguous" | "rejected" | "unknown" | "ignored";
  role: "time" | "signal" | "unmapped" | "ignored";
  unresolved?: string | null;
  semantics?: string | null;
  profile_role?: string | null;
  semantic_status?: string | null;
  unit_hypothesis?: string | null;
  evidence?: string | null;
  evidence_level?: "verified" | "supported" | "unverified" | null;
  handling?: string | null;
  canonical: string | null;
  group: string | null;
  unit: string | null;
  expected_unit: string | null;
  unit_status: string;
  required: boolean;
  method: string;
  ambiguity: string | null;
  notes: string[];
  samples: (string | number)[];
  conversion?: { scale: number; offset: number } | null;
};

export type AssessmentCheck = { id: string; status: "ok" | "warn" | "fail" | "skip"; text: string };

export type Assessment = {
  outcome: "ready" | "needs_input" | "incompatible" | "invalid";
  compatible: boolean;
  headline: string;
  checks: AssessmentCheck[];
  detector_inputs: {
    key: string;
    label: string;
    role: "scored" | "pipeline" | "optional";
    status: "ok" | "missing" | "unit_unresolved" | "ambiguous";
    sources: Record<string, string>;
  }[];
  missing: { canonical: string; label: string; unit: string; group: string }[];
  unresolved: {
    column: string;
    canonical: string;
    label: string;
    issue: string;
    detail: string | null;
    expected_unit: string | null;
    required: boolean;
  }[];
  needed: string[];
  counts: Record<string, number>;
  profile: { id: string; documentation_status: string } | null;
  pipeline_note?: string | null;
  next_steps?: { kind: "confirm" | "tilt_test" | "document" | "log_signals"; text: string; facts: Record<string, string[]> }[];
  format_summary?: { label: string; status: "ok" | "pending" | "missing" | "fail" | "info"; text: string }[] | null;
  profile_match?: {
    kind: "exact" | "compatible" | "partial" | "none";
    profile: string | null;
    file_fingerprint: string;
    missing_required: string[];
    missing_optional: string[];
    extra_columns: string[];
  } | null;
};

export type AutoResult = CombinedPreview & {
  stage: "analyzed" | "analysis_failed" | "needs_input" | "incompatible" | "invalid" | "not_stored";
  stored: boolean;
  flight_id: string | null;
  upload_errors?: string[];
  duplicate_of?: string | null;
  analysis?: { ok: boolean; status: string; n_events: number; n_windows: number; errors: string[]; verdict?: Verdict | null };
};

export type CombinedPreview = {
  assessment?: Assessment;
  ok: boolean;
  errors?: string[];
  filename: string;
  size_bytes?: number;
  rows?: number;
  columns?: MappedColumn[];
  counts?: Record<string, number>;
  sample_rows?: Record<string, unknown>[];
  mapping?: { version: string; missing_required: string[]; missing_required_labels: string[]; blockers: string[] };
  timestamp?: {
    checked: boolean;
    unit?: string;
    median_dt_s?: number | null;
    est_hz?: number | null;
    duration_s?: number | null;
    n_duplicates?: number;
    n_nonmonotonic_steps?: number;
    errors?: string[];
    warnings?: string[];
  };
  scoreable: boolean;
  blockers?: string[];
  warnings?: string[];
  targets?: {
    canonical: string;
    group: string;
    label: string;
    expected_unit: string;
    required: boolean;
    accepted_units: string[];
  }[];
  llm?: { status: string; message: string; disclosure: string; columns_sent: number; model?: string | null };
  scoreable_note?: string;
};

export type DirectWindow = { available: boolean; n: number; median: number | null; min: number | null; max: number | null; sigma?: number | null; jitter?: number | null };

export type DirectMeasurement = {
  channel: string;
  group: string | null;
  role: string | null;
  semantics: string;
  kinds: string[];
  before: DirectWindow;
  during: DirectWindow;
  after: DirectWindow;
  before_median: number | null;
  during_min: number | null;
  during_max: number | null;
  during_median: number | null;
  after_median: number | null;
  max_step: number | null;
  swing: number | null;
  shift: number | null;
  shift_sigma: number | null;
  direction: "higher" | "lower" | "unchanged" | null;
  persistence: string | null;
  level_held_until_row: number | null;
  onset_row: number;
  onset_raw: number | null;
};

export type DirectAssociation = {
  type: "simultaneous" | "sequential" | "group" | "correlation";
  basis: string;
  physical_relationship: string;
  channels?: string[];
  a?: string;
  b?: string;
  r?: number;
};

export type DirectCategory = "data_quality" | "state_transition" | "statistical_irregularity";

export type DirectEvent = {
  id: string;
  category: DirectCategory;
  priority: "review_first" | "review" | "minor";
  kinds: string[];
  channels: string[];
  n_channels_involved: number;
  start_index: number;
  end_index: number;
  n_samples: number;
  start_row: number;
  end_row: number;
  start_raw: number | null;
  end_raw: number | null;
  span_raw: number | null;
  span_s_assumed: { value: number; basis: string } | null;
  why_selected: string;
  measurements: DirectMeasurement[];
  associations: DirectAssociation[];
  concurrent_with: { event_id: string; category: DirectCategory; relation: string }[];
};

export type DirectOperatorEvent = {
  title: string;
  when: string;
  what_changed: string[];
  same_time: { kind: string; text: string }[];
  can_conclude: string[];
  unknown: string[];
  review_next: string[];
  why_selected: string;
};

export type DirectOperatorReport = {
  headline: string;
  flight_summary: string[];
  events: Record<string, DirectOperatorEvent>;
  recommended_review: { event_id: string; action: string; why: string }[];
  data_quality_material: { issue: string; detail: string }[];
  essential_limitations: string[];
  not_established: string;
  pilot_report?: PilotReport;
  pilot_events?: Record<string, PilotEvent>;
};

export type DirectSummary = {
  filename: string;
  input_fingerprint: string;
  analysis_version?: string;
  operator?: DirectOperatorReport;
  engine?: Record<string, number>;
  profile: { id: string | null; version: string | null; match: string; documentation_status: string | null };
  timing: {
    time_column: string | null;
    time_basis: { column: string | null; unit_hypothesis: string | null; status: string; source: string; evidence: string };
    median_step_raw?: number | null;
    span_raw?: number | null;
    first_raw?: number | null;
    last_raw?: number | null;
    duration_s?: { value: number; basis: string };
    sampling_hz?: { value: number; basis: string };
    n_gaps?: number;
    repeated_timestamps?: number;
    out_of_order_steps?: number;
    rows_without_valid_timestamp?: number;
    note?: string;
  };
  channels: Record<string, { name: string; group: string; role: string | null; unit: string | null; unit_hypothesis: string | null; unit_status: string; meaning_status: string }>;
  stats: Record<string, Record<string, number | boolean | string | null>>;
  events: DirectEvent[];
  event_overflow: number;
  peaks: { id: string; channel: string; row: number; raw_time: number | null; value: number; channel_median: number; departure: number }[];
  phases: { flag: string; value: number; start_row: number; end_row: number; start_raw: number | null; end_raw: number | null; n: number }[];
  flag_transitions: { channel: string; from: number; to: number; index: number }[];
  correlations: { pairs: { a: string; b: string; r: number }[]; rate_pairs: { a: string; b: string; r: number }[]; note: string };
  data_quality: {
    rows: number;
    rows_analysed: number;
    columns: number;
    numeric_columns: number;
    non_numeric_columns: string[];
    constant_channels: string[];
    channels_with_missing: Record<string, { missing: number; non_finite: number }>;
    notes: string[];
  };
  unassessed_periods: { kinds: string[]; channels: string[]; start_row: number; end_row: number; start_raw: number | null; end_raw: number | null }[];
  limitations: string[];
  not_implemented: string[];
};

export type EvidenceRef = { id: string; value: number | string };

export type DirectReport = {
  label: string;
  mode: string;
  status: "ok" | "partial" | "not_requested" | "unavailable" | "error" | "invalid_output";
  message: string;
  executive_summary: string;
  summary_source?: "ai" | "deterministic";
  events: {
    event_id: string;
    category: DirectCategory;
    title: string;
    summary: string;
    same_time: string;
    conclusion: string;
    unknown: string;
    review: string;
    supporting_channels: string[];
    measurements: EvidenceRef[];
    source: { start_row: number; end_row: number; start_raw: number | null; end_raw: number | null };
  }[];
  data_quality: { issue: string; detail: string }[];
  recommendations: { action: string; rationale: string }[];
  unknowns: string[];
  limitations: string[];
  rejected: { section: string; reason: string; text: string }[];
  normalized?: Record<string, string>[];
  failure: { reason: string; detail?: string; fields?: string[]; attempts?: number; kind?: string; finish_reason?: string | null } | null;
  provenance: Record<string, unknown>;
  reused?: boolean;
  reused_from?: string;
};

export type DirectView = {
  analysis_id: string;
  label: string;
  stage?: string;
  stale_analysis?: boolean;
  meta: { filename: string; created_at: string; rows: number; columns: number; events: number; ai_attempt?: { status: string; message?: string; reason?: string | null; fields?: string[]; attempts?: number | null } | null };
  summary: DirectSummary;
  report: DirectReport | null;
  consent_text: string;
  separation_note: string;
};

export type DirectEventContext = {
  event_id: string;
  axis: string;
  x_is: "raw_ticks" | "row";
  regions: {
    baseline: { start: number; end: number; available: boolean };
    event: { start: number; end: number };
    post: { start: number; end: number; available: boolean };
  };
  channels: {
    signal: string;
    label: string;
    technical_name?: string;
    group: string | null;
    semantics: string | null;
    time: number[];
    values: (number | null)[];
    before: DirectWindow;
    during: DirectWindow;
    after: DirectWindow;
    kinds: string[];
    persistence: string | null;
    shift_sigma: number | null;
  }[];
  window_start: number;
  window_end: number;
  segment_limited: boolean;
  baseline_samples: number;
};

export type Assignments = Record<string, { canonical: string; unit: string }>;

export type FlightDetail = {
  flight_id: string;
  status: string;
  source?: string | null;
  filenames: Record<string, string>;
  file_sizes: Record<string, number>;
  warnings: string[];
  validation?: Record<string, unknown> | null;
  created_at?: string | null;
  analyzed_at?: string | null;
  analysis?: AnalyzeResponse | Record<string, unknown> | null;
  has_report: boolean;
  has_ai_diagnostic: boolean;
  event_count: number;
  verdict?: Verdict | null;
};

const LAST_FLIGHT_KEY = "uav_sentinel_last_flight_id";

export function getLastFlightId(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(LAST_FLIGHT_KEY);
}

export function setLastFlightId(id: string): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(LAST_FLIGHT_KEY, id);
}

export function clearLastFlightId(id?: string): void {
  if (typeof window === "undefined") return;
  if (!id || window.localStorage.getItem(LAST_FLIGHT_KEY) === id) {
    window.localStorage.removeItem(LAST_FLIGHT_KEY);
  }
}

export type AnomalyEvent = {
  event_id: string;
  session_id: number;
  start_time_s: number;
  end_time_s: number;
  duration_s: number;
  flagged_window_count: number;
  peak_score: number;
  mean_flagged_score: number;
  window_indices: number[];
  scores: number[];
  per_feature_peak: Record<string, number>;
  per_feature_mean: Record<string, number>;
};

export type AiStatus = { configured: boolean; model: string };

export type Diagnostic = {
  flight_id: string;
  summary: string;
  findings: Array<{
    event_id: string;
    possible_causes: Array<{
      hypothesis: string;
      confidence: string;
      supporting_evidence: string[];
      contradicting_evidence: string[];
    }>;
    observed_facts: string[];
    unknowns: string[];
    recommended_checks: string[];
  }>;
  overall_limitations: string[];
  insufficient_evidence: boolean;
  model_name?: string | null;
  notes?: string | null;
};

export function exportUrl(flightId: string, kind: "json" | "windows.csv" | "events.csv") {
  return `${API_URL}/api/flights/${encodeURIComponent(flightId)}/exports/${kind}`;
}

export const api = {
  health: () => getJson<{ status: string }>("/api/health"),
  artifacts: () => getJson<ArtifactsStatus>("/api/artifacts/status"),
  aiStatus: () => getJson<AiStatus>("/api/ai/status"),
  listFlights: () => getJson<FlightListItem[]>("/api/flights"),
  getFlight: (id: string) => getJson<FlightDetail>(`/api/flights/${encodeURIComponent(id)}`),
  deleteFlight: (id: string) =>
    sendJson<{ status: string }>(`/api/flights/${encodeURIComponent(id)}`, "DELETE"),
  analyze: (id: string, max_gap_s = 1.5) =>
    sendJson<AnalyzeResponse>(`/api/flights/${encodeURIComponent(id)}/analyze`, "POST", {
      max_gap_s,
    }),
  events: (id: string) =>
    getJson<AnomalyEvent[]>(`/api/flights/${encodeURIComponent(id)}/events`),
  report: (id: string) => getJson<Record<string, unknown>>(`/api/flights/${encodeURIComponent(id)}/report`),
  chart: (id: string, name: string, eventId?: string) => {
    const q = eventId ? `?event_id=${encodeURIComponent(eventId)}` : "";
    return getJson<Record<string, unknown>>(
      `/api/flights/${encodeURIComponent(id)}/charts/${name}${q}`
    );
  },
  evidenceTrace: (id: string) =>
    getJson<{ trace: Record<string, unknown>; formatted: string }>(
      `/api/flights/${encodeURIComponent(id)}/evidence-trace`
    ),
  getDiagnostic: (id: string) =>
    getJson<Diagnostic>(`/api/flights/${encodeURIComponent(id)}/ai/diagnostic`),
  runDiagnostic: (id: string, selected_event_ids?: string[]) =>
    sendJson<{ diagnostic: Diagnostic; trace?: Record<string, unknown>; usage?: Record<string, unknown> }>(
      `/api/flights/${encodeURIComponent(id)}/ai/diagnostic`,
      "POST",
      { selected_event_ids: selected_event_ids ?? null }
    ),
  getChat: (id: string) =>
    getJson<{ flight_id: string; messages: { role: string; content: string }[]; selected_event_id?: string | null }>(
      `/api/flights/${encodeURIComponent(id)}/ai/chat`
    ),
  postChat: (id: string, message: string, selected_event_id?: string | null) =>
    sendJson<{
      answer: { answer: string; caveats: string[]; insufficient_evidence: boolean };
      chat: { messages: { role: string; content: string }[] };
    }>(`/api/flights/${encodeURIComponent(id)}/ai/chat`, "POST", {
      message,
      selected_event_id: selected_event_id ?? null,
    }),
  clearChat: (id: string) =>
    sendJson(`/api/flights/${encodeURIComponent(id)}/ai/chat`, "DELETE"),
  uploadCsvs: async (files: {
    imu: File;
    att: File;
    baro: File;
    flightId?: string;
  }): Promise<UploadResponse> => {
    const fd = new FormData();
    fd.append("imu", files.imu);
    fd.append("att", files.att);
    fd.append("baro", files.baro);
    if (files.flightId) fd.append("flight_id", files.flightId);
    const res = await fetch(`${API_URL}/api/flights/upload`, { method: "POST", body: fd });
    if (!res.ok) await parseError(res);
    return res.json();
  },
  overview: (id: string) => getJson<Overview>(`/api/flights/${encodeURIComponent(id)}/overview`),
  uploadSummary: (id: string) =>
    getJson<UploadSummary>(`/api/flights/${encodeURIComponent(id)}/upload-summary`),
  timeline: (id: string) => getJson<Timeline>(`/api/flights/${encodeURIComponent(id)}/timeline`),
  eventDetail: (id: string, eventId: string) =>
    getJson<EventDetail>(
      `/api/flights/${encodeURIComponent(id)}/events/${encodeURIComponent(eventId)}/detail`
    ),
  eventContext: (id: string, eventId: string) =>
    getJson<EventContext>(
      `/api/flights/${encodeURIComponent(id)}/events/${encodeURIComponent(eventId)}/context`
    ),
  operatorReport: (id: string) =>
    getJson<OperatorReport>(`/api/flights/${encodeURIComponent(id)}/operator-report`),
  telemetry: (id: string, limit = 200, start_s?: number, end_s?: number) => {
    const q = new URLSearchParams({ limit: String(limit) });
    if (start_s !== undefined) q.set("start_s", String(start_s));
    if (end_s !== undefined) q.set("end_s", String(end_s));
    return getJson<{ columns: string[]; rows: Record<string, number | null>[]; total_rows: number; returned: number }>(
      `/api/flights/${encodeURIComponent(id)}/telemetry?${q}`
    );
  },
  getExplanation: (id: string) =>
    getJson<Explanation>(`/api/flights/${encodeURIComponent(id)}/explanation`),
  createExplanation: (id: string) =>
    sendJson<Explanation>(`/api/flights/${encodeURIComponent(id)}/explanation`, "POST", {
      consent: true,
    }),
  combinedPreview: async (file: File, assignments: Assignments, llmConsent: boolean) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("assignments", JSON.stringify(assignments));
    fd.append("llm_consent", String(llmConsent));
    const res = await fetch(`${API_URL}/api/flights/combined/preview`, { method: "POST", body: fd });
    if (!res.ok) await parseError(res);
    return res.json() as Promise<CombinedPreview>;
  },
  directAnalyze: async (file: File, consent: boolean) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("llm_consent", String(consent));
    const res = await fetch(`${API_URL}/api/direct/analyze`, { method: "POST", body: fd });
    if (!res.ok) await parseError(res);
    return res.json() as Promise<DirectView>;
  },
  directGet: (id: string) => getJson<DirectView>(`/api/direct/${encodeURIComponent(id)}`),
  directList: () =>
    getJson<{ analysis_id: string; filename: string; created_at: string; rows: number; events: number; has_ai_report: boolean }[]>("/api/direct"),
  directAi: (id: string) =>
    sendJson<DirectView>(`/api/direct/${encodeURIComponent(id)}/ai`, "POST", { consent: true }),
  directContext: (id: string, eventId: string) =>
    getJson<DirectEventContext>(`/api/direct/${encodeURIComponent(id)}/events/${encodeURIComponent(eventId)}/context`),
  directRows: (id: string, limit = 200) =>
    getJson<{ columns: string[]; rows: Record<string, number | null>[]; total_rows: number; returned: number }>(
      `/api/direct/${encodeURIComponent(id)}/rows?limit=${limit}`
    ),
  directDelete: (id: string) => sendJson<{ status: string }>(`/api/direct/${encodeURIComponent(id)}`, "DELETE"),
  directExportUrl: (id: string) => `${API_URL}/api/direct/${encodeURIComponent(id)}/export`,
  combinedAuto: async (file: File, assignments: Assignments, llmConsent: boolean, flightId?: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("assignments", JSON.stringify(assignments));
    fd.append("llm_consent", String(llmConsent));
    if (flightId) fd.append("flight_id", flightId);
    const res = await fetch(`${API_URL}/api/flights/combined/auto`, { method: "POST", body: fd });
    if (!res.ok) await parseError(res);
    return res.json() as Promise<AutoResult>;
  },
  combinedUpload: async (file: File, assignments: Assignments, flightId?: string) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("assignments", JSON.stringify(assignments));
    if (flightId) fd.append("flight_id", flightId);
    const res = await fetch(`${API_URL}/api/flights/combined/upload`, { method: "POST", body: fd });
    if (!res.ok) await parseError(res);
    return res.json() as Promise<UploadResponse>;
  },
  uploadZip: async (zip: File, flightId?: string): Promise<UploadResponse> => {
    const fd = new FormData();
    fd.append("zip_file", zip);
    if (flightId) fd.append("flight_id", flightId);
    const res = await fetch(`${API_URL}/api/flights/upload`, { method: "POST", body: fd });
    if (!res.ok) await parseError(res);
    return res.json();
  },
};
