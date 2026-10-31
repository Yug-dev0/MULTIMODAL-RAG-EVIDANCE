import { useCallback, useEffect, useState } from "react";
import { apiClient, StatsSummary, EventRow, RecentQuery, EvalSessionSummary } from "../../api/client";
import type { GraphNode, GraphLink } from "../GraphView";
import { SmartSearch } from "./SmartSearch";
import { StatCard } from "./StatCard";
import { RecentEventsPanel } from "./RecentEventsPanel";
import { KnowledgePulse } from "./KnowledgePulse";
import { SystemOverviewRing } from "./SystemOverviewRing";
import { ModalityBars } from "./ModalityBars";
import { RecentQueriesPanel } from "./RecentQueriesPanel";
import { EvalPanel } from "./EvalPanel";

const ICONS = {
  sources: <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8"><path strokeLinecap="round" d="M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5" /></svg>,
  media: <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8"><path strokeLinecap="round" d="M3 12h4l2 8 4-16 2 8h6" /></svg>,
  evidence: <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8"><path strokeLinecap="round" strokeLinejoin="round" d="M6 2h9l5 5v15H6zM9 13h6M9 17h6" /></svg>,
  events: <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="12" cy="12" r="9" /><path strokeLinecap="round" d="M12 7v5l3 3" /></svg>,
  relationships: <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="6" cy="6" r="2.5" /><circle cx="18" cy="8" r="2.5" /><circle cx="10" cy="18" r="2.5" /><path strokeLinecap="round" d="M8 7l7.5 1M9 16l6-6.5" /></svg>,
};

interface Props {
  onSearch: (q: string) => void;
  onOpenEvent: (id: string) => void;
  onOpenGraph: () => void;
  graphData: { nodes: GraphNode[]; links: GraphLink[] };
}

export function Dashboard({ onSearch, onOpenEvent, onOpenGraph, graphData }: Props) {
  const [stats, setStats] = useState<StatsSummary | null>(null);
  const [events, setEvents] = useState<EventRow[]>([]);
  const [recentQueries, setRecentQueries] = useState<RecentQuery[]>([]);
  const [evalSummary, setEvalSummary] = useState<EvalSessionSummary | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const [s, e, q, ev] = await Promise.all([
        apiClient.getStatsSummary(),
        apiClient.getEvents(6),
        apiClient.getRecentQueries(6),
        apiClient.getEvalSessionSummary(),
      ]);
      setStats(s);
      setEvents(e);
      setRecentQueries(q);
      setEvalSummary(ev);
    } catch {
      // backend unreachable -- widgets fall back to their empty states
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 8000);
    return () => clearInterval(id);
  }, [refresh]);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-sans text-[28px] font-semibold tracking-tight text-ink">
          Multimodal <span className="text-signal">Evidence</span> Console
        </h1>
        <p className="mt-1 text-[15px] text-dim">Ask questions across your videos, audio, images, and documents.</p>
      </div>

      <SmartSearch onSearch={onSearch} />

      <div className="grid grid-cols-2 gap-4 md:grid-cols-5">
        <StatCard icon={ICONS.sources} value={String(stats?.sources ?? (loading ? "…" : 0))} label="Sources" sublabel={stats ? `${stats.jobs_done} completed` : undefined} />
        <StatCard icon={ICONS.media} value={stats ? `${(stats.total_media_seconds / 3600).toFixed(1)}h` : loading ? "…" : "0h"} label="Media processed" />
        <StatCard icon={ICONS.evidence} value={String(stats?.evidence_units ?? (loading ? "…" : 0))} label="Evidence units" sublabel="Across all modalities" />
        <StatCard icon={ICONS.events} value={String(stats?.knowledge_events ?? (loading ? "…" : 0))} label="Knowledge events" sublabel="Reconstructed" accent="amber" />
        <StatCard icon={ICONS.relationships} value={String(stats?.relationships ?? (loading ? "…" : 0))} label="Relationships" sublabel="In graph" accent="amber" />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
          <RecentEventsPanel events={events} onOpen={onOpenEvent} />
          <KnowledgePulse nodes={graphData.nodes} links={graphData.links} onOpenGraph={onOpenGraph} />
        </div>

        <div className="flex flex-col gap-4">
          <SystemOverviewRing
            pct={stats?.processing_complete_pct ?? 0}
            processedSeconds={stats?.total_media_seconds ?? 0}
            totalSeconds={stats?.total_media_seconds ?? 0}
          />
          <ModalityBars distribution={stats?.modality_distribution ?? {}} />
          <RecentQueriesPanel queries={recentQueries} onSelect={onSearch} />
          {evalSummary && <EvalPanel summary={evalSummary} />}
        </div>
      </div>
    </div>
  );
}
