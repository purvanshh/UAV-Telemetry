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
};

export type FlightListItem = {
  flight_id: string;
  status: string;
  source?: string | null;
  created_at?: string | null;
  filenames: Record<string, string>;
};

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
  uploadZip: async (zip: File, flightId?: string): Promise<UploadResponse> => {
    const fd = new FormData();
    fd.append("zip_file", zip);
    if (flightId) fd.append("flight_id", flightId);
    const res = await fetch(`${API_URL}/api/flights/upload`, { method: "POST", body: fd });
    if (!res.ok) await parseError(res);
    return res.json();
  },
};
