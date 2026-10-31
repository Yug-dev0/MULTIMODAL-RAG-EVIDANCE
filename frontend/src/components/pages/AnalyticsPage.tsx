import { useEffect, useState } from "react";
import { apiClient, StatsSummary, EvalSessionSummary } from "../../api/client";
import { ModalityBars } from "../dashboard/ModalityBars";
import { EvalPanel } from "../dashboard/EvalPanel";

export function AnalyticsPage() {
  const [stats, setStats] = useState<StatsSummary | null>(null);
  const [evalSummary, setEvalSummary] = useState<EvalSessionSummary | null>(null);

  useEffect(() => {
    apiClient.getStatsSummary().then(setStats).catch(() => {});
    apiClient.getEvalSessionSummary().then(setEvalSummary).catch(() => {});
  }, []);

  return (
    <div>
      <p className="eyebrow mb-1">Tools</p>
      <h2 className="mb-4 font-sans text-xl font-semibold text-ink">Analytics</h2>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <ModalityBars distribution={stats?.modality_distribution ?? {}} />
        {evalSummary && <EvalPanel summary={evalSummary} />}
        <div className="border border-hairline bg-panel p-4 md:col-span-2">
          <p className="eyebrow mb-3">Corpus totals</p>
          <div className="grid grid-cols-2 gap-3 font-mono text-xs md:grid-cols-5">
            {stats && Object.entries({
              Sources: stats.sources, Evidence: stats.evidence_units, Events: stats.knowledge_events,
              Relationships: stats.relationships, "Media hrs": (stats.total_media_seconds / 3600).toFixed(1),
            }).map(([k, v]) => (
              <div key={k}>
                <p className="text-dim">{k}</p>
                <p className="text-lg text-ink">{v}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
