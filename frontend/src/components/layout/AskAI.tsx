import { useState } from "react";
import { apiClient, QueryResponse } from "../../api/client";
import { EvidenceCard } from "../EvidenceCard";

export function AskAI({ onClose }: { onClose: () => void }) {
  const [queryText, setQueryText] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      setResult(await apiClient.query(queryText));
    } catch (err) {
      setError((err as Error).message || "The assistant could not reach the backend.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-void/70 p-6 pt-24" onClick={onClose}>
      <div
        className="bracket-frame w-full max-w-2xl border border-hairline bg-panel p-5 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-center justify-between">
          <p className="eyebrow flex items-center gap-1.5 text-signal">
            <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.8"><path strokeLinecap="round" strokeLinejoin="round" d="M12 3l1.8 4.6L18 9l-4.2 1.4L12 15l-1.8-4.6L6 9l4.2-1.4z" /></svg>
            Ask AI — grounded in your ingested corpus
          </p>
          <button onClick={onClose} className="font-mono text-xs text-dim hover:text-ink" aria-label="Close">✕</button>
        </div>

        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            autoFocus
            className="flex-1 rounded-md border border-hairline bg-panel2 px-3 py-2 font-sans text-sm text-ink placeholder:text-dim focus:outline-none focus:border-signal/60"
            placeholder="Ask anything about the ingested video, audio, images, or documents…"
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
          />
          <button
            type="submit"
            disabled={loading || !queryText}
            className="rounded-md border border-signal/50 bg-signal/10 px-4 py-2 font-mono text-xs uppercase tracking-wide text-signal disabled:opacity-40"
          >
            {loading ? "…" : "Ask"}
          </button>
        </form>

        {error && <p role="alert" className="mt-3 border border-amber/40 bg-amber/5 px-3 py-2 font-mono text-xs text-amber">{error}</p>}

        {result && (
          <div className="mt-4 max-h-[50vh] overflow-y-auto">
            <p className="font-sans text-sm leading-relaxed text-ink">{result.answer}</p>
            {result.citations.length > 0 && (
              <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-2">
                {result.citations.map((c) => <EvidenceCard key={c.evidence_id} citation={c} />)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
