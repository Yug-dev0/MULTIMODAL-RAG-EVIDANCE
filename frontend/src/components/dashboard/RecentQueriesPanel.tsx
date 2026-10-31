import type { RecentQuery } from "../../api/client";

function timeLabel(ms: number): string {
  return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function RecentQueriesPanel({ queries, onSelect }: { queries: RecentQuery[]; onSelect: (q: string) => void }) {
  return (
    <div className="border border-hairline bg-panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <p className="eyebrow">Recent queries</p>
      </div>
      {queries.length === 0 ? (
        <p className="font-mono text-[11px] text-dim">No queries run this session yet.</p>
      ) : (
        <div className="flex flex-col divide-y divide-hairline/60">
          {queries.map((q, i) => (
            <button
              key={i}
              onClick={() => onSelect(q.query)}
              className="flex items-center justify-between gap-2 py-2 text-left transition-colors hover:text-signal"
            >
              <span className="truncate font-sans text-[12px] text-ink" title={q.query}>{q.query}</span>
              <span className="flex-shrink-0 font-mono text-[10px] text-dim">{timeLabel(q.timestamp * 1000)}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
