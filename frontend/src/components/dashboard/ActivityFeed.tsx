import type { ActivityItem } from "../../api/client";
import { ModalityGlyph, modalityLabel } from "../ModalityGlyph";

function timeAgo(ms: number): string {
  const s = Math.round((Date.now() - ms) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}

const STATUS_LABEL: Record<string, string> = {
  done: "Completed", running: "Processing…", pending: "Queued", failed: "Failed",
};

export function ActivityFeed({ items }: { items: ActivityItem[] }) {
  return (
    <div className="border border-hairline bg-panel p-4">
      <p className="eyebrow mb-3">Recent activity</p>
      {items.length === 0 ? (
        <p className="font-mono text-[11px] text-dim">Nothing ingested yet — head to Upload to add a source.</p>
      ) : (
        <div className="flex flex-col">
          {items.map((item, i) => (
            <div key={item.job_id} className={`flex items-start gap-3 py-3 ${i !== items.length - 1 ? "border-b border-hairline/60" : ""}`}>
              <span className={`mt-1 h-2 w-2 flex-shrink-0 rounded-full ${item.status === "failed" ? "bg-amber" : item.status === "done" ? "bg-signal" : "animate-pulse2 bg-signal"}`} />
              <div className="min-w-0 flex-1">
                <p className="truncate font-sans text-[13px] text-ink">
                  {item.status === "done" ? "Processed" : STATUS_LABEL[item.status] ?? item.status}: {item.filename ?? item.job_id.slice(0, 8)}
                </p>
                {item.status === "done" && item.modalities.length > 0 && (
                  <p className="mt-0.5 text-[11px] text-dim">
                    {item.evidence_count} evidence units{item.event_count ? `, ${item.event_count} events` : ""}
                  </p>
                )}
                {item.status === "failed" && item.error && (
                  <p className="mt-0.5 truncate font-mono text-[11px] text-amber">{item.error}</p>
                )}
                <div className="mt-1.5 flex flex-wrap gap-1.5">
                  {item.modalities.map((m) => (
                    <span key={m} className="flex items-center gap-1 rounded bg-panel2 px-1.5 py-0.5 font-mono text-[10px] text-dim">
                      <ModalityGlyph modality={m} className="h-2.5 w-2.5" />{modalityLabel(m)}
                    </span>
                  ))}
                </div>
              </div>
              <div className="flex flex-shrink-0 flex-col items-end gap-1">
                <span className="font-mono text-[10px] text-dim">{timeAgo(item.submitted_at * 1000)}</span>
                <span className={`font-mono text-[10px] ${item.status === "failed" ? "text-amber" : item.status === "done" ? "text-signal" : "text-dim"}`}>
                  {STATUS_LABEL[item.status] ?? item.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
