import type { TopEntity } from "../../api/client";

export function TopEntitiesPanel({ entities }: { entities: TopEntity[] }) {
  const max = Math.max(1, ...entities.map((e) => e.mention_count));
  return (
    <div className="border border-hairline bg-panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <p className="eyebrow">Top entities</p>
        <span className="font-mono text-[10px] text-dim">by mentions</span>
      </div>
      {entities.length === 0 ? (
        <p className="font-mono text-[11px] text-dim">No entities resolved yet.</p>
      ) : (
        <div className="flex flex-col gap-2.5">
          {entities.map((e, i) => (
            <div key={e.id} className="flex items-center gap-2">
              <span className="w-4 font-mono text-[11px] text-dim">{i + 1}</span>
              <span className="w-24 flex-shrink-0 truncate font-sans text-[12px] text-ink" title={e.canonical_name}>{e.canonical_name}</span>
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-hairline">
                <div className="h-full rounded-full bg-amber" style={{ width: `${(e.mention_count / max) * 100}%`, transition: "width 0.6s ease" }} />
              </div>
              <span className="w-6 text-right font-mono text-[11px] text-dim">{e.mention_count}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
