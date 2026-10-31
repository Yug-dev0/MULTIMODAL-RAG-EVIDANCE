import { useRef, useState } from "react";
import { JobStatus, JobStage } from "../api/client";
import { ModalityGlyph } from "./ModalityGlyph";
import { SOURCE_TYPES, SourceType, detectSourceType } from "../lib/sourceType";

const ACCEPT_HINT: Record<string, string> = {
  video: "MP4, MOV, AVI, MKV, WEBM",
  audio: "MP3, WAV, M4A, FLAC, OGG",
  image: "PNG, JPG, WEBP, GIF, BMP",
  pdf: "PDF",
};

const PIPELINE_STAGES: { id: JobStage; label: string; sublabel: string }[] = [
  { id: "upload", label: "Upload", sublabel: "File received" },
  { id: "extract", label: "Extract", sublabel: "ASR, OCR, vision" },
  { id: "understand", label: "Understand", sublabel: "Entities resolved" },
  { id: "link", label: "Link", sublabel: "Relationships, events" },
  { id: "index", label: "Index", sublabel: "Vector + graph" },
];

/** How many files upload at once. The pipeline itself runs server-side, so this
 *  only bounds concurrent request bodies, not concurrent processing. */
const UPLOAD_CONCURRENCY = 3;


function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function glyphFor(type: string) {
  return type === "pdf" ? "document" : type === "image" ? "image" : "speech";
}

const STATUS_COPY: Record<string, string> = {
  pending: "Queued",
  running: "Processing…",
  done: "Ingested — ready to query",
  failed: "Ingestion failed",
};

interface QueuedFile {
  key: string;
  file: File;
  sourceType: SourceType;
}

function keyFor(f: File): string {
  return `${f.name}:${f.size}:${f.lastModified}`;
}

/** Runs `worker` over every item, at most `limit` in flight. Never rejects --
 *  the worker is responsible for recording its own failures. */
async function runWithConcurrency<T>(items: T[], limit: number, worker: (item: T, index: number) => Promise<void>) {
  let cursor = 0;
  const runners = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor++;
      await worker(items[index], index);
    }
  });
  await Promise.all(runners);
}

interface SubmitFailure {
  filename: string;
  message: string;
}

interface Props {
  jobs: JobStatus[];
  submit: (file: File, sourceType: string, sessionId?: string) => Promise<{ job_id: string; session_id: string }>;
  onStartChat?: (sessionId: string) => void;
}

export function UploadPanel({ jobs, submit, onStartChat }: Props) {
  const [sourceType, setSourceType] = useState<SourceType>("video");
  const [sourceTypeTouched, setSourceTypeTouched] = useState(false);
  const [queue, setQueue] = useState<QueuedFile[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [batchJobIds, setBatchJobIds] = useState<string[]>([]);
  const [focusedJobId, setFocusedJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [failures, setFailures] = useState<SubmitFailure[]>([]);
  const [uploadedSessionId, setUploadedSessionId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const batchJobs = batchJobIds
    .map((id) => jobs.find((j) => j.job_id === id))
    .filter((j): j is JobStatus => Boolean(j));
  const activeJob =
    batchJobs.find((j) => j.job_id === focusedJobId) ??
    batchJobs.find((j) => j.status === "pending" || j.status === "running") ??
    batchJobs[0] ??
    null;
  const currentStageIdx = activeJob ? PIPELINE_STAGES.findIndex((s) => s.id === activeJob.stage) : -1;
  const failedIdx = activeJob?.status === "failed" ? Math.max(currentStageIdx, 0) : -1;
  const doneCount = batchJobs.filter((j) => j.status === "done").length;

  /** Appends to the queue, skipping files already staged. Each file carries its
   *  own detected type so a mixed PDF + image batch ingests correctly. */
  function addFiles(incoming: FileList | File[] | null) {
    const list = Array.from(incoming ?? []);
    if (list.length === 0) return;

    setQueue((prev) => {
      const seen = new Set(prev.map((q) => q.key));
      const added: QueuedFile[] = [];
      for (const file of list) {
        const key = keyFor(file);
        if (seen.has(key)) continue;
        seen.add(key);
        added.push({ key, file, sourceType: detectSourceType(file.name) ?? sourceType });
      }
      return [...prev, ...added];
    });

    if (!sourceTypeTouched) {
      const detected = list.map((f) => detectSourceType(f.name)).find(Boolean);
      if (detected) setSourceType(detected);
    }
    setError(null);
  }

  function removeFile(key: string) {
    setQueue((prev) => prev.filter((q) => q.key !== key));
  }

  function setFileType(key: string, type: SourceType) {
    setQueue((prev) => prev.map((q) => (q.key === key ? { ...q, sourceType: type } : q)));
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    addFiles(e.dataTransfer.files);
  }

  async function uploadQueue(): Promise<string | null> {
    setError(null);
    setFailures([]);
    if (queue.length === 0) {
      setError("Choose at least one file first.");
      return null;
    }
    setSubmitting(true);

    const submitted = queue;
    const jobIds: (string | null)[] = new Array(submitted.length).fill(null);
    const errors: SubmitFailure[] = [];

    const record = (item: QueuedFile, err: unknown) => {
      errors.push({
        filename: item.file.name,
        message: (err as Error).message || "Upload failed — is the backend reachable?",
      });
    };

    // The whole batch has to land in ONE chat session, otherwise every file
    // gets its own and you can't ask a question spanning them. /ingest mints a
    // session when none is passed, so the first upload goes alone to establish
    // the id and the rest ride along on it.
    let sessionId: string | undefined;
    let cursor = 0;
    while (cursor < submitted.length && sessionId === undefined) {
      const item = submitted[cursor];
      try {
        const { job_id, session_id } = await submit(item.file, item.sourceType);
        jobIds[cursor] = job_id;
        sessionId = session_id;
      } catch (err) {
        record(item, err);
      }
      cursor++;
    }

    const remaining = submitted.slice(cursor).map((item, k) => ({ item, index: cursor + k }));

    await runWithConcurrency(remaining, UPLOAD_CONCURRENCY, async ({ item, index }) => {
      try {
        jobIds[index] = (await submit(item.file, item.sourceType, sessionId)).job_id;
      } catch (err) {
        record(item, err);
      }
    });

    const accepted = jobIds.filter((id): id is string => Boolean(id));
    setBatchJobIds(accepted);
    setFocusedJobId(accepted[0] ?? null);
    setFailures(errors);
    if (accepted.length === 0 && errors.length > 0) {
      setError(errors.length === 1 ? errors[0].message : `All ${errors.length} uploads failed.`);
    }
    // Keep only what didn't make it, so a partial failure can be retried as-is.
    const failedNames = new Set(errors.map((f) => f.filename));
    setQueue(submitted.filter((q) => failedNames.has(q.file.name)));
    if (fileInputRef.current) fileInputRef.current.value = "";
    setSubmitting(false);

    return sessionId ?? null;
  }

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    const sessionId = await uploadQueue();
    if (sessionId) {
      setUploadedSessionId(sessionId);
    }
  }

  async function handleIngestAndChat() {
    const sessionId = await uploadQueue();
    if (sessionId) {
      setUploadedSessionId(sessionId);
      if (onStartChat) {
        onStartChat(sessionId);
      }
    }
  }

  const totalBytes = queue.reduce((sum, q) => sum + q.file.size, 0);

  return (
    <div>
      <p className="eyebrow mb-1">Ingest</p>
      <h2 className="mb-1 font-sans text-2xl font-semibold text-ink">Upload &amp; ingest</h2>
      <p className="mb-5 text-sm text-dim">Add multimedia sources to build your knowledge corpus.</p>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div className="flex flex-col gap-4">
          <div className="flex gap-2">
            {SOURCE_TYPES.map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => { setSourceType(t); setSourceTypeTouched(true); }}
                className={`flex items-center gap-1.5 rounded-md border px-3.5 py-2 font-mono text-[12px] uppercase tracking-wide transition-colors ${
                  sourceType === t ? "border-signal/60 bg-signal/10 text-signal" : "border-hairline text-dim hover:border-dim"
                }`}
              >
                <ModalityGlyph modality={glyphFor(t)} className="h-4 w-4" />
                {t}
              </button>
            ))}
          </div>

          <form onSubmit={handleUpload} className="flex flex-col gap-4">
            <label
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              className={`bracket-frame flex cursor-pointer flex-col items-center gap-3 rounded-lg border-2 border-dashed px-6 py-16 text-center transition-colors ${
                dragOver ? "border-signal bg-signal/5" : "border-hairline bg-panel hover:border-signal/50"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                data-testid="file-input"
                className="sr-only"
                onChange={(e) => addFiles(e.target.files)}
              />
              <svg viewBox="0 0 24 24" className="h-10 w-10 text-dim" fill="none" stroke="currentColor" strokeWidth="1.4"><path strokeLinecap="round" strokeLinejoin="round" d="M12 16V6M7 11l5-5 5 5M4 20h16" /></svg>
              <span className="font-sans text-base text-ink">Drag &amp; drop files here</span>
              <span className="font-mono text-xs text-dim">or click to browse — multiple files, mixed types</span>
              <span className="font-mono text-[10px] text-dim/70">{ACCEPT_HINT[sourceType]}</span>
            </label>

            {queue.length > 0 && (
              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between font-mono text-[11px] text-dim">
                  <span data-testid="queue-summary">
                    {queue.length} file{queue.length === 1 ? "" : "s"} staged — {formatBytes(totalBytes)}
                  </span>
                  <button type="button" onClick={() => setQueue([])} className="text-dim hover:text-amber">
                    Clear all
                  </button>
                </div>
                {queue.map((q) => (
                  <div key={q.key} data-testid="queued-file" className="card flex items-center gap-3 p-3">
                    <ModalityGlyph modality={glyphFor(q.sourceType)} className="h-5 w-5 text-signal" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate font-sans text-sm text-ink">{q.file.name}</p>
                      <p className="font-mono text-[11px] text-dim">{formatBytes(q.file.size)}</p>
                    </div>
                    <select
                      value={q.sourceType}
                      aria-label={`Source type for ${q.file.name}`}
                      onChange={(e) => setFileType(q.key, e.target.value as SourceType)}
                      className="rounded-md border border-hairline bg-panel px-2 py-1 font-mono text-[11px] uppercase tracking-wide text-dim"
                    >
                      {SOURCE_TYPES.map((t) => (
                        <option key={t} value={t}>{t}</option>
                      ))}
                    </select>
                    <button
                      type="button"
                      onClick={() => removeFile(q.key)}
                      aria-label={`Remove ${q.file.name}`}
                      className="font-mono text-xs text-dim hover:text-amber"
                    >
                      Cancel
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="flex flex-wrap gap-3">
              <button
                type="submit"
                disabled={submitting || queue.length === 0}
                className="w-fit rounded-md border border-signal/50 bg-signal/10 px-6 py-2.5 font-mono text-xs uppercase tracking-wide text-signal transition-colors hover:bg-signal/20 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {submitting ? "Submitting" : queue.length > 1 ? `Ingest ${queue.length} files only` : "Ingest only"}
              </button>
              <button
                type="button"
                disabled={submitting || queue.length === 0}
                onClick={handleIngestAndChat}
                className="w-fit rounded-md border border-signal bg-signal/20 px-6 py-2.5 font-mono text-xs uppercase tracking-wide text-signal transition-colors hover:bg-signal/30 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Ingest &amp; Start Chat
              </button>
            </div>

            {error && (
              <p role="alert" className="border border-amber/40 bg-amber/5 px-4 py-3 font-mono text-xs text-amber">{error}</p>
            )}

            {failures.length > 0 && !error && (
              <div role="alert" className="border border-amber/40 bg-amber/5 px-4 py-3 font-mono text-xs text-amber">
                {failures.length} of {failures.length + batchJobIds.length} uploads failed:
                <ul className="mt-1 list-disc pl-4">
                  {failures.map((f) => (
                    <li key={f.filename}>{f.filename} — {f.message}</li>
                  ))}
                </ul>
              </div>
            )}
          </form>
        </div>

        <div className="card p-4">
          <div className="mb-4 flex items-baseline justify-between">
            <p className="eyebrow">Processing pipeline</p>
            {batchJobs.length > 1 && (
              <span data-testid="batch-progress" className="font-mono text-[10px] text-dim">
                {doneCount}/{batchJobs.length} done
              </span>
            )}
          </div>

          {batchJobs.length > 1 && (
            <div className="mb-4 flex flex-col gap-1">
              {batchJobs.map((j) => (
                <button
                  key={j.job_id}
                  type="button"
                  onClick={() => setFocusedJobId(j.job_id)}
                  className={`flex items-center gap-2 rounded border px-2 py-1.5 text-left font-mono text-[11px] transition-colors ${
                    j.job_id === activeJob?.job_id ? "border-signal/50 bg-signal/5 text-ink" : "border-hairline text-dim hover:border-dim"
                  }`}
                >
                  <span className={`h-1.5 w-1.5 flex-shrink-0 rounded-full ${
                    j.status === "failed" ? "bg-amber" : j.status === "done" ? "bg-signal" : "animate-pulse2 bg-signal"
                  }`} />
                  <span className="truncate">{j.filename ?? j.job_id.slice(0, 8)}</span>
                </button>
              ))}
            </div>
          )}

          {!activeJob ? (
            <p className="font-mono text-[11px] text-dim">Ingest a file to see live stage-by-stage progress here.</p>
          ) : (
            <div className="flex flex-col gap-4">
              {batchJobs.length > 1 && (
                <p className="truncate font-mono text-[11px] text-ink">{activeJob.filename ?? activeJob.job_id.slice(0, 8)}</p>
              )}
              {PIPELINE_STAGES.map((stage, i) => {
                const isFailed = failedIdx === i;
                const isDone = activeJob.status === "done" || (currentStageIdx > i && activeJob.status !== "failed");
                const isCurrent = i === currentStageIdx && activeJob.status !== "done";
                return (
                  <div key={stage.id} className="flex items-start gap-3">
                    <div className="flex flex-col items-center">
                      <span className={`flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full font-mono text-[10px] ${
                        isFailed ? "bg-amber text-void" : isDone ? "bg-signal text-void" : isCurrent ? "animate-pulse2 border-2 border-signal text-signal" : "border border-hairline text-dim"
                      }`}>
                        {isDone ? "✓" : isFailed ? "!" : i + 1}
                      </span>
                      {i < PIPELINE_STAGES.length - 1 && <span className={`mt-1 h-8 w-px ${isDone ? "bg-signal" : "bg-hairline"}`} />}
                    </div>
                    <div>
                      <p className={`font-sans text-sm ${isCurrent ? "text-signal" : isDone ? "text-ink" : "text-dim"}`}>{stage.label}</p>
                      <p className="font-mono text-[10px] text-dim">{stage.sublabel}</p>
                    </div>
                  </div>
                );
              })}
              {activeJob.status === "failed" && activeJob.error && (
                <p role="alert" className="mt-1 border border-amber/30 bg-amber/5 px-3 py-2 font-mono text-[11px] text-amber">{activeJob.error}</p>
              )}
              {activeJob.status === "done" && (
                <p data-testid="ingest-status" className="font-mono text-[11px] text-signal">Status: done — {STATUS_COPY.done}</p>
              )}
              {activeJob.status !== "done" && (
                <p data-testid="ingest-status" className="font-mono text-[11px] text-dim">Status: {activeJob.status} — {STATUS_COPY[activeJob.status] ?? activeJob.status}</p>
              )}
              {uploadedSessionId && onStartChat && (
                <button
                  type="button"
                  onClick={() => onStartChat(uploadedSessionId)}
                  className="mt-4 w-full flex items-center justify-center gap-2 rounded-md border border-signal bg-signal/15 py-2.5 font-mono text-xs uppercase tracking-wide text-signal transition-colors hover:bg-signal/25"
                >
                  <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                  Start Chatting
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {jobs.length > 0 && (
        <div className="mt-8">
          <p className="eyebrow mb-2">Ingestion history — keeps running across tabs</p>
          <div className="flex flex-col gap-1">
            {jobs.map((j) => (
              <div key={j.job_id} className="flex items-center justify-between font-mono text-[11px] text-dim border-b border-hairline/20 py-1" title={j.error ?? undefined}>
                <div className="flex items-center gap-2">
                  <span className={`h-1.5 w-1.5 rounded-full ${j.status === "failed" ? "bg-amber" : j.status === "done" ? "bg-signal" : "animate-pulse2 bg-signal"}`} />
                  <span className="text-ink">{j.filename ?? j.job_id.slice(0, 8)}</span>
                  <span>({STATUS_COPY[j.status] ?? j.status})</span>
                  {j.status === "failed" && j.error && <span className="truncate text-amber/80">— {j.error}</span>}
                </div>
                {j.session_id && onStartChat && (
                  <button
                    type="button"
                    onClick={() => onStartChat(j.session_id!)}
                    title="Chat about this file"
                    className="p-1 rounded text-signal hover:bg-signal/10 transition-colors"
                  >
                    <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="2.2">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                    </svg>
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
