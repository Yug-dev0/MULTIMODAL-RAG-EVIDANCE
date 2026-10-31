import { auth, IS_MOCK_AUTH } from "../firebase";

// Exported: EvidenceCard renders citation thumbnails with <img src={`${BASE_URL}...`}>.
export const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export type JobStage = "upload" | "extract" | "understand" | "link" | "index" | "done";

export interface JobStatus {
  job_id: string;
  status: "pending" | "running" | "done" | "failed";
  source_id: string | null;
  error: string | null;
  filename?: string | null;
  source_type?: string | null;
  submitted_at?: number;
  stage?: JobStage;
  stage_timings_ms?: Record<string, number>;
  session_id?: string | null;
}

export interface Citation {
  evidence_id: string;
  modality: string;
  source: string;
  content: string;
  timestamp_start: number | null;
  timestamp_end: number | null;
  page_number: number | null;
  thumbnail: string | null;
  confidence: number;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  events_used: string[];
}

export interface CompareResponse {
  multimodal: QueryResponse;
  baseline: { answer: string };
}

export interface StatsSummary {
  evidence_units: number;
  knowledge_events: number;
  relationships: number;
  sources: number;
  modality_distribution: Record<string, number>;
  total_media_seconds: number;
  jobs_running: number;
  jobs_done: number;
  jobs_failed: number;
  jobs_total: number;
  processing_complete_pct: number;
}

export interface ActivityItem {
  job_id: string;
  filename: string | null;
  source_type: string | null;
  status: string;
  modalities: string[];
  evidence_count: number;
  event_count: number;
  submitted_at: number;
  error: string | null;
}

export interface TopEntity {
  id: string;
  canonical_name: string;
  entity_type: string;
  mention_count: number;
}

export interface SourceRow {
  source_id: string;
  source_type: string;
  evidence_count: number;
  first_seen: string | null;
  duration_seconds: number | null;
  raw_locator: string;
}

export interface EventRow {
  id: string;
  event_type: string;
  title: string;
  summary: string;
  timestamp_start: number | null;
  timestamp_end: number | null;
  confidence: number;
  created_at: string | null;
  evidence_count?: number;
  relationship_count?: number;
  entity_count?: number;
}

export interface EventEvidenceRow {
  id: string;
  content_text: string;
  modality: string;
  content_type: string;
  raw_locator: string;
  timestamp_start: number | null;
  timestamp_end: number | null;
  page_number: number | null;
  confidence: number;
}

export interface EventRelationshipRow {
  id: string;
  source_id: string;
  target_id: string;
  source_label: string;
  target_label: string;
  relationship_type: string;
  confidence: number;
}

export interface EventEntityRow {
  id: string;
  canonical_name: string;
  entity_type: string;
}

export interface EventDetail {
  id: string;
  event_type: string;
  title: string;
  summary: string;
  timestamp_start: number | null;
  timestamp_end: number | null;
  confidence: number;
  created_at: string | null;
  evidence: EventEvidenceRow[];
  relationships: EventRelationshipRow[];
  entities: EventEntityRow[];
}

export interface EvidenceRow {
  id: string;
  content_text: string;
  modality: string;
  source_type: string;
  raw_locator: string;
  timestamp_start: number | null;
  timestamp_end: number | null;
  page_number: number | null;
  confidence: number;
  /** Semantic match against the searched question; null when browsing. Not
   *  the same thing as `confidence`, which rates extraction quality only. */
  relevance: number | null;
  extracted_at: string | null;
  session_id: string | null;
  session_title: string | null;
}

/** A question that was asked, with the evidence the answer actually cited.
 *  This is the grouping the Evidence page shows by default -- grouping by
 *  chat session mixed one question's evidence in with everything else that
 *  had ever been ingested into that session. */
export interface QueryEvidenceGroup {
  query: string;
  asked_at: string;
  session_id: string;
  session_title: string;
  evidence: EvidenceRow[];
}

export interface RecentQuery {
  query: string;
  mode: "query" | "compare";
  citation_count: number;
  modality_count: number;
  timestamp: number;
}

export interface EvalSessionSummary {
  comparisons_run: number;
  avg_multimodal_citations: number;
  avg_baseline_citations: number;
  queries_with_cross_modal_evidence: number;
  queries_baseline_had_no_provenance: number;
}

export interface ChatSessionSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  source_count: number;
}

export interface ChatMessageRow {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  created_at: string;
}

export interface ChatQueryResponse {
  answer: string;
  citations: Citation[];
  events_used: string[];
  session_title: string;
}

// Retrieve the active authorization token dynamically for API requests
async function getAuthToken(): Promise<string | null> {
  if (IS_MOCK_AUTH) {
    let stored: string | null = null;
    try {
      if (typeof window !== "undefined" && window.localStorage) {
        stored = window.localStorage.getItem("mock_user");
      } else if (typeof localStorage !== "undefined" && localStorage) {
        stored = localStorage.getItem("mock_user");
      }
    } catch {
      stored = null;
    }
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        return `mock-token-${parsed.email}`;
      } catch {
        return null;
      }
    }
    return null;
  }
  if (!auth || !auth.currentUser) return null;
  return auth.currentUser.getIdToken();
}

async function getJson<T>(path: string): Promise<T> {
  const token = await getAuthToken();
  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  const res = await fetch(`${BASE_URL}${path}`, { headers });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

async function ingest(file: File, sourceType: string, sessionId?: string): Promise<{ job_id: string; session_id: string }> {
  const token = await getAuthToken();
  const formData = new FormData();
  formData.append("file", file);
  formData.append("source_type", sourceType);
  if (sessionId) formData.append("session_id", sessionId);

  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${BASE_URL}/ingest`, {
    method: "POST",
    headers,
    body: formData
  });

  if (!res.ok) throw new Error(`ingest failed: ${res.status}`);
  return res.json();
}

async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const token = await getAuthToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

async function createChatSession(): Promise<ChatSessionSummary> {
  return postJson<ChatSessionSummary>("/sessions");
}

async function listChatSessions(): Promise<ChatSessionSummary[]> {
  return getJson<ChatSessionSummary[]>("/sessions");
}

async function getChatMessages(sessionId: string): Promise<ChatMessageRow[]> {
  return getJson<ChatMessageRow[]>(`/sessions/${sessionId}/messages`);
}

async function sendChatQuery(sessionId: string, queryText: string, topK = 10): Promise<ChatQueryResponse> {
  return postJson<ChatQueryResponse>(`/sessions/${sessionId}/query`, { query: queryText, top_k: topK });
}

async function getIngestStatus(jobId: string): Promise<JobStatus> {
  return getJson<JobStatus>(`/ingest/${jobId}`);
}

async function getJobs(): Promise<JobStatus[]> {
  return getJson<JobStatus[]>("/jobs");
}

async function query(queryText: string, topK = 10): Promise<QueryResponse> {
  const token = await getAuthToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json"
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${BASE_URL}/query`, {
    method: "POST",
    headers,
    body: JSON.stringify({ query: queryText, top_k: topK }),
  });
  if (!res.ok) throw new Error(`query failed: ${res.status}`);
  return res.json();
}

async function compare(queryText: string): Promise<CompareResponse> {
  const token = await getAuthToken();
  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${BASE_URL}/compare?query=${encodeURIComponent(queryText)}`, {
    headers
  });
  if (!res.ok) throw new Error(`compare failed: ${res.status}`);
  return res.json();
}

async function getStatsSummary(): Promise<StatsSummary> {
  return getJson<StatsSummary>("/stats/summary");
}

async function getRecentActivity(limit = 10): Promise<ActivityItem[]> {
  return getJson<ActivityItem[]>(`/activity/recent?limit=${limit}`);
}

async function getTopEntities(limit = 10): Promise<TopEntity[]> {
  return getJson<TopEntity[]>(`/entities/top?limit=${limit}`);
}

async function getSources(): Promise<SourceRow[]> {
  return getJson<SourceRow[]>("/sources");
}

async function getEvents(limit = 100): Promise<EventRow[]> {
  return getJson<EventRow[]>(`/events?limit=${limit}`);
}

async function getEventDetail(eventId: string): Promise<EventDetail> {
  return getJson<EventDetail>(`/events/${eventId}`);
}

async function getEvidence(limit = 200, q?: string): Promise<EvidenceRow[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (q && q.trim()) params.set("q", q.trim());
  return getJson<EvidenceRow[]>(`/evidence?${params.toString()}`);
}

async function getEvidenceByQuery(limit = 50): Promise<QueryEvidenceGroup[]> {
  return getJson<QueryEvidenceGroup[]>(`/evidence/by-query?limit=${limit}`);
}

async function getRecentQueries(limit = 10): Promise<RecentQuery[]> {
  return getJson<RecentQuery[]>(`/queries/recent?limit=${limit}`);
}

async function getEvalSessionSummary(): Promise<EvalSessionSummary> {
  return getJson<EvalSessionSummary>("/eval/session-summary");
}

async function deleteChatSession(sessionId: string): Promise<{ status: string }> {
  const token = await getAuthToken();
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${BASE_URL}/sessions/${sessionId}`, {
    method: "DELETE",
    headers,
  });
  if (!res.ok) throw new Error(`delete session failed: ${res.status}`);
  return res.json();
}

async function deleteSource(sourceId: string): Promise<{ status: string }> {
  const token = await getAuthToken();
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${BASE_URL}/sources/${sourceId}`, {
    method: "DELETE",
    headers,
  });
  if (!res.ok) throw new Error(`delete source failed: ${res.status}`);
  return res.json();
}

export const apiClient = {
  ingest, getIngestStatus, getJobs, query, compare,
  getStatsSummary, getRecentActivity, getTopEntities, getSources, deleteSource, getEvents, getEventDetail, getEvidence,
  getEvidenceByQuery, getRecentQueries, getEvalSessionSummary,
  createChatSession, listChatSessions, getChatMessages, sendChatQuery, deleteChatSession,
};
