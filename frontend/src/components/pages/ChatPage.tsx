import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient, ChatMessageRow, ChatSessionSummary, Citation, JobStatus } from "../../api/client";
import { EvidenceCard } from "../EvidenceCard";
import { EvidenceDetail } from "../EvidencePreview";
import { detectSourceType } from "../../lib/sourceType";

const EXAMPLES = ["Show me the caching architecture", "Who discussed Redis?", "Where was the diagram shown?"];

export interface ChatOpenSignal { query: string; id: number; sessionId?: string; }

interface Props {
  jobs: JobStatus[];
  submit: (file: File, sourceType: string, sessionId?: string) => Promise<{ job_id: string; session_id: string }>;
  openSignal?: ChatOpenSignal | null;
}

function timeAgo(iso: string): string {
  const s = Math.round((Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

export function ChatPage({ jobs, submit, openSignal }: Props) {
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessageRow[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [attachedJobIds, setAttachedJobIds] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const threadEndRef = useRef<HTMLDivElement>(null);
  const lastOpenSignalId = useRef<number | null>(null);
  const refreshedForJob = useRef<Set<string>>(new Set());
  // Mirrors activeSessionId synchronously for reads inside async handlers.
  const activeSessionIdRef = useRef<string | null>(null);
  // A file upload and a sent message can both fire in the same tick
  // (attach-then-immediately-ask). If both see "no active session" and each
  // independently creates one, the upload lands in session A while the
  // question is asked against a brand-new, sourceless session B. Tracking
  // the in-flight creation promise lets a second caller await the same one
  // instead of starting its own.
  const creatingSessionRef = useRef<Promise<string> | null>(null);

  const setActiveSession = useCallback((id: string | null) => {
    activeSessionIdRef.current = id;
    setActiveSessionId(id);
  }, []);

  const ensureSessionId = useCallback(async (): Promise<string> => {
    if (activeSessionIdRef.current) return activeSessionIdRef.current;
    if (!creatingSessionRef.current) {
      creatingSessionRef.current = apiClient.createChatSession().then((created) => {
        setActiveSession(created.id);
        return created.id;
      }).finally(() => {
        creatingSessionRef.current = null;
      });
    }
    return creatingSessionRef.current;
  }, [setActiveSession]);

  const refreshSessions = useCallback(() => {
    apiClient.listChatSessions().then(setSessions).catch(() => {});
  }, []);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  useEffect(() => {
    threadEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  const openSession = useCallback((id: string | null) => {
    setActiveSession(id);
    setError(null);
    setAttachedJobIds([]);
    setSelectedCitation(null);
    if (id) {
      apiClient.getChatMessages(id).then(setMessages).catch(() => setMessages([]));
    } else {
      setMessages([]);
    }
  }, [setActiveSession]);

  useEffect(() => {
    if (openSignal && openSignal.id !== lastOpenSignalId.current) {
      lastOpenSignalId.current = openSignal.id;
      if (openSignal.sessionId) {
        openSession(openSignal.sessionId);
      } else {
        openSession(null);
      }
      setInput(openSignal.query || "");
      if (openSignal.query && openSignal.query.trim()) {
        void sendMessage(openSignal.query.trim());
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [openSignal]);

  const attachedJobs = jobs.filter(
    (j) => j.session_id === activeSessionId || attachedJobIds.includes(j.job_id)
  );

  useEffect(() => {
    for (const job of attachedJobs) {
      if (job.status === "done" && !refreshedForJob.current.has(job.job_id)) {
        refreshedForJob.current.add(job.job_id);
        refreshSessions();
      }
    }
  }, [attachedJobs, refreshSessions]);

  async function handleDeleteSession(id: string) {
    try {
      await apiClient.deleteChatSession(id);
      if (activeSessionIdRef.current === id) {
        openSession(null);
      }
      refreshSessions();
    } catch (err) {
      setError((err as Error).message || "Could not delete this session.");
    }
  }

  async function sendMessage(text: string) {
    setError(null);
    setSending(true);
    setInput("");
    try {
      const sessionId = await ensureSessionId();
      setMessages((prev) => [
        ...prev,
        { id: `optimistic-${Date.now()}`, role: "user", content: text, citations: [], created_at: new Date().toISOString() },
      ]);
      const result = await apiClient.sendChatQuery(sessionId, text);
      setMessages((prev) => [
        ...prev,
        {
          id: `optimistic-a-${Date.now()}`, role: "assistant", content: result.answer,
          citations: result.citations, created_at: new Date().toISOString(),
        },
      ]);
      refreshSessions();
    } catch (err) {
      setError((err as Error).message || "The question could not be answered.");
    } finally {
      setSending(false);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (input.trim() && !sending) await sendMessage(input.trim());
  }

  async function handleAttach(file: File) {
    const sourceType = detectSourceType(file.name);
    if (!sourceType) {
      setError(`"${file.name}" isn't a supported file type (video, audio, image, or PDF).`);
      return;
    }
    setError(null);
    try {
      const sessionId = await ensureSessionId();
      const { job_id } = await submit(file, sourceType, sessionId);
      setAttachedJobIds((prev) => [...prev, job_id]);
      refreshSessions();
    } catch (err) {
      setError((err as Error).message || "Upload failed — is the backend reachable?");
    }
  }

  const activeSession = sessions.find((s) => s.id === activeSessionId) ?? null;

  return (
    <div className="flex h-[calc(100vh-160px)] min-h-[560px] gap-4">
      {/* Session history rail */}
      <div className="flex w-64 flex-shrink-0 flex-col border border-hairline bg-panel">
        <div className="border-b border-hairline p-3">
          <button
            onClick={() => openSession(null)}
            className="flex w-full items-center justify-center gap-2 rounded-md border border-signal/40 bg-signal/10 py-2.5 font-mono text-xs uppercase tracking-wide text-signal transition-colors hover:bg-signal/20"
          >
            <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" d="M12 5v14M5 12h14" /></svg>
            New chat
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {sessions.length === 0 ? (
            <p className="px-4 py-6 text-center font-sans text-[12px] text-dim">
              Your topics will appear here once you start a chat.
            </p>
          ) : (
            sessions.map((s) => (
              <div
                key={s.id}
                className={`group flex items-center justify-between border-b border-hairline/60 transition-colors ${
                  s.id === activeSessionId ? "bg-signal/10" : "hover:bg-panel2"
                }`}
              >
                <button
                  onClick={() => openSession(s.id)}
                  className="flex-1 min-w-0 px-3.5 py-3 text-left"
                >
                  <p className={`truncate font-sans text-[13px] ${s.id === activeSessionId ? "text-signal" : "text-ink"}`}>
                    {s.title}
                  </p>
                  <div className="mt-1 flex items-center gap-2 font-mono text-[10px] text-dim">
                    <span>{timeAgo(s.updated_at)}</span>
                    <span className="text-hairline">·</span>
                    <span>{s.source_count} source{s.source_count === 1 ? "" : "s"}</span>
                  </div>
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    if (confirm("Are you sure you want to delete this session?")) {
                      void handleDeleteSession(s.id);
                    }
                  }}
                  title="Delete Session"
                  className="opacity-0 group-hover:opacity-100 p-2.5 mr-1.5 rounded text-dim hover:text-amber hover:bg-amber/10 transition-all"
                >
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Thread */}
      <div className="flex flex-1 flex-col border border-hairline bg-panel">
        <div className="flex items-center justify-between border-b border-hairline px-5 py-3">
          <div>
            <p className="font-sans text-sm font-medium text-ink">{activeSession?.title ?? "New chat"}</p>
            {activeSession && (
              <p className="font-mono text-[11px] text-dim">
                {activeSession.source_count} source{activeSession.source_count === 1 ? "" : "s"} in this topic
              </p>
            )}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-5">
          {messages.length === 0 && !sending ? (
            <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
              <svg viewBox="0 0 24 24" className="h-8 w-8 text-signal/60" fill="none" stroke="currentColor" strokeWidth="1.4"><path strokeLinecap="round" strokeLinejoin="round" d="M12 3l1.8 4.6L18 9l-4.2 1.4L12 15l-1.8-4.6L6 9l4.2-1.4z" /></svg>
              <div>
                <p className="font-sans text-[15px] text-ink">Ask about a topic, or attach a file to start one.</p>
                <p className="mt-1 font-sans text-[13px] text-dim">Each chat stays scoped to what you upload into it.</p>
              </div>
              <div className="flex flex-wrap justify-center gap-2">
                {EXAMPLES.map((ex) => (
                  <button
                    key={ex}
                    onClick={() => sendMessage(ex)}
                    className="rounded-full border border-hairline px-2.5 py-1 font-sans text-[11px] text-dim transition-colors hover:border-signal/50 hover:text-signal"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              {messages.map((m) => (
                <div key={m.id} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div className={`max-w-[75%] ${m.role === "user" ? "" : "w-full"}`}>
                    <div
                      className={`rounded-md px-4 py-3 font-sans text-[14px] leading-relaxed ${
                        m.role === "user" ? "bg-signal/12 text-ink" : "border border-hairline bg-panel2 text-ink"
                      }`}
                    >
                      {m.content}
                    </div>
                    {m.role === "assistant" && m.citations.length > 0 && (
                      <div className="mt-2 grid grid-cols-1 gap-2 md:grid-cols-2">
                        {m.citations.map((c) => (
                          <EvidenceCard key={c.evidence_id} citation={c} onClick={() => setSelectedCitation(c)} />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {sending && (
                <div className="flex justify-start">
                  <div className="flex items-center gap-2 rounded-md border border-hairline bg-panel2 px-4 py-3">
                    <span className="h-1.5 w-1.5 animate-pulse2 rounded-full bg-signal" />
                    <span className="h-1.5 w-1.5 animate-pulse2 rounded-full bg-signal [animation-delay:150ms]" />
                    <span className="h-1.5 w-1.5 animate-pulse2 rounded-full bg-signal [animation-delay:300ms]" />
                  </div>
                </div>
              )}
              <div ref={threadEndRef} />
            </div>
          )}
        </div>

        {attachedJobs.length > 0 && (
          <div className="flex flex-col gap-1.5 border-t border-hairline bg-panel2 px-5 py-2.5">
            {attachedJobs.map((job) => (
              <div key={job.job_id} className="flex items-center gap-2 font-mono text-[11px]">
                <span
                  className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${
                    job.status === "done" ? "bg-signal" : job.status === "failed" ? "bg-amber" : "animate-pulse2 bg-amber"
                  }`}
                />
                <span className="truncate text-ink">{job.filename}</span>
                <span className="flex-shrink-0 text-dim">
                  {job.status === "done" ? "indexed" : job.status === "failed" ? `failed — ${job.error}` : job.stage ?? "upload"}
                </span>
              </div>
            ))}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex w-fit items-center gap-1.5 font-mono text-[11px] text-dim transition-colors hover:text-signal"
            >
              <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2.2"><path strokeLinecap="round" d="M12 5v14M5 12h14" /></svg>
              Add another file
            </button>
          </div>
        )}
        {error && (
          <p role="alert" className="border-t border-amber/40 bg-amber/5 px-5 py-2 font-mono text-xs text-amber">
            {error}
          </p>
        )}

        <form onSubmit={handleSubmit} className="flex items-center gap-2 border-t border-hairline p-3">
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void handleAttach(f);
              e.target.value = "";
            }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            aria-label="Attach a file"
            className="flex-shrink-0 rounded-md border border-hairline p-2.5 text-dim transition-colors hover:border-signal/50 hover:text-signal"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8"><path strokeLinecap="round" strokeLinejoin="round" d="M21.44 11.05l-9.19 9.19a5 5 0 01-7.07-7.07l9.19-9.19a3.5 3.5 0 014.95 4.95l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" /></svg>
          </button>
          <input
            className="flex-1 rounded-md border border-hairline bg-panel px-4 py-2.5 font-sans text-sm text-ink placeholder:text-dim focus:outline-none focus:border-signal/60"
            placeholder="Ask about this topic..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={sending}
          />
          <button
            type="submit"
            disabled={sending || !input.trim()}
            className="flex-shrink-0 rounded-md bg-signal px-5 py-2.5 font-mono text-xs uppercase tracking-wide text-void transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
          >
            Send
          </button>
        </form>
      </div>

      {/* Evidence preview — opens the exact video timestamp / PDF page a
          citation came from, on the right, without leaving the chat. */}
      {selectedCitation && (
        <div className="flex w-[420px] flex-shrink-0 flex-col border border-hairline bg-panel">
          <div className="flex items-center justify-between border-b border-hairline px-4 py-3">
            <p className="eyebrow">Evidence source</p>
            <button
              onClick={() => setSelectedCitation(null)}
              aria-label="Close evidence preview"
              className="font-mono text-xs text-dim hover:text-ink"
            >
              ✕
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            <EvidenceDetail
              u={{
                modality: selectedCitation.modality, raw_locator: selectedCitation.source,
                content_text: selectedCitation.content, timestamp_start: selectedCitation.timestamp_start,
                timestamp_end: selectedCitation.timestamp_end, page_number: selectedCitation.page_number,
                confidence: selectedCitation.confidence,
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
