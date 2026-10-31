import { ModalityGlyph, modalityLabel } from "../ModalityGlyph";

export function ModalityBars({ distribution }: { distribution: Record<string, number> }) {
  const total = Object.values(distribution).reduce((a, b) => a + b, 0);
  const entries = Object.entries(distribution).sort((a, b) => b[1] - a[1]);

  return (
    <div className="border border-hairline bg-panel p-4">
      <p className="eyebrow mb-3">Modality distribution</p>
      {entries.length === 0 ? (
        <p className="font-mono text-[11px] text-dim">No evidence ingested yet.</p>
      ) : (
        <div className="flex flex-col gap-2.5">
          {entries.map(([modality, count]) => {
            const pct = total ? Math.round((count / total) * 100) : 0;
            return (
              <div key={modality}>
                <div className="mb-1 flex items-center justify-between text-xs text-dim">
                  <span className="flex items-center gap-1.5"><ModalityGlyph modality={modality} className="h-3 w-3" />{modalityLabel(modality)}</span>
                  <span className="font-mono">{pct}%</span>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-hairline">
                  <div className="h-full rounded-full bg-signal" style={{ width: `${pct}%`, transition: "width 0.6s ease" }} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
