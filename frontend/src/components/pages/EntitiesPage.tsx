import { useEffect, useState } from "react";
import { apiClient, TopEntity } from "../../api/client";

export function EntitiesPage() {
  const [entities, setEntities] = useState<TopEntity[] | null>(null);

  useEffect(() => {
    apiClient.getTopEntities(100).then(setEntities).catch(() => setEntities([]));
  }, []);

  return (
    <div>
      <p className="eyebrow mb-1">Explore</p>
      <h2 className="mb-4 font-sans text-xl font-semibold text-ink">Entities</h2>
      {entities === null ? (
        <p className="font-mono text-xs text-dim">Loading…</p>
      ) : entities.length === 0 ? (
        <p className="font-mono text-xs text-dim">No entities resolved yet — ingest a source with real content first.</p>
      ) : (
        <div className="grid grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-4">
          {entities.map((e) => (
            <div key={e.id} className="border border-hairline bg-panel p-3">
              <p className="truncate font-sans text-sm text-ink" title={e.canonical_name}>{e.canonical_name}</p>
              <p className="font-mono text-[10px] uppercase tracking-wide text-dim">{e.entity_type}</p>
              <p className="mt-1 font-mono text-[11px] text-signal">{e.mention_count} mention{e.mention_count === 1 ? "" : "s"}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
