import type { EvalSessionSummary } from "../../api/client";

export function EvalPanel({ summary }: { summary: EvalSessionSummary }) {
  const hasData = summary.comparisons_run > 0;

  return (
    <div className="border border-hairline bg-panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <p className="eyebrow">Multimodal vs. text-only baseline</p>
        {hasData && <span className="font-mono text-[10px] text-signal">{summary.comparisons_run} run{summary.comparisons_run === 1 ? "" : "s"}</span>}
      </div>
      {!hasData ? (
        <p className="font-mono text-[11px] text-dim">
          Run a query in the Compare tab to see live evidence-retrieval stats here — nothing to report yet.
        </p>
      ) : (
        <div className="flex flex-col gap-2 font-mono text-[11px]">
          <div className="flex items-center justify-between">
            <span className="text-dim">Avg. citations — multimodal</span>
            <span className="text-signal">{summary.avg_multimodal_citations}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-dim">Avg. citations — baseline</span>
            <span className="text-amber">{summary.avg_baseline_citations}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-dim">Queries with cross-modal evidence</span>
            <span className="text-ink">{summary.queries_with_cross_modal_evidence}/{summary.comparisons_run}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-dim">Baseline queries with zero provenance</span>
            <span className="text-ink">{summary.queries_baseline_had_no_provenance}/{summary.comparisons_run}</span>
          </div>
        </div>
      )}
    </div>
  );
}
