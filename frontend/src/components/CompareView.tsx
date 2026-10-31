import { useState } from "react";
import { apiClient, CompareResponse } from "../api/client";

export function CompareView() {
  const [queryText, setQueryText] = useState("");
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      setResult(await apiClient.compare(queryText));
    } catch (err) {
      setError((err as Error).message || "The comparison could not be completed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-5">
      <div>
        <p className="eyebrow">Multimodal system vs. text-only baseline</p>
        <form onSubmit={handleSubmit} className="mt-2 flex gap-2">
          <input
            className="flex-1 rounded-md border border-hairline bg-panel px-4 py-2.5 font-sans text-sm text-ink placeholder:text-dim focus:outline-none focus:border-signal/60"
            placeholder="Compare a query across both systems..."
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
          />
          <button type="submit" disabled={loading || !queryText}
                  className="rounded-md border border-signal/50 bg-signal/10 px-5 py-2.5 font-mono text-xs uppercase tracking-wide text-signal transition-colors hover:bg-signal/20 disabled:cursor-not-allowed disabled:opacity-40">
            {loading ? "Comparing" : "Compare"}
          </button>
        </form>
      </div>

      {error && (
        <p role="alert" className="border border-amber/40 bg-amber/5 px-4 py-3 font-mono text-xs text-amber">
          {error}
        </p>
      )}

      {result && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="bracket-frame border border-signal/40 bg-panel p-5">
            <p className="eyebrow text-signal">Multimodal system</p>
            <p className="mt-3 font-sans text-sm leading-relaxed text-ink" data-testid="multimodal-answer">
              {result.multimodal.answer}
            </p>
            <p className="mt-4 font-mono text-[11px] text-dim">
              {result.multimodal.citations.length} traceable citation{result.multimodal.citations.length === 1 ? "" : "s"}
              {" "}across {new Set(result.multimodal.citations.map((c) => c.modality)).size} modalit
              {new Set(result.multimodal.citations.map((c) => c.modality)).size === 1 ? "y" : "ies"}
            </p>
          </div>
          <div className="border border-hairline bg-panel p-5">
            <p className="eyebrow">Text-only baseline</p>
            <p className="mt-3 font-sans text-sm leading-relaxed text-dim" data-testid="baseline-answer">
              {result.baseline.answer}
            </p>
            <p className="mt-4 font-mono text-[11px] text-dim">no traceable citations</p>
          </div>
        </div>
      )}
    </div>
  );
}
