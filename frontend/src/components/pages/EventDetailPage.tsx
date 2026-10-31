import { useEffect, useState } from "react";
import { apiClient, EventDetail } from "../../api/client";
import { mediaUrl } from "../../api/media";
import { ModalityGlyph, SignalBar, modalityLabel } from "../ModalityGlyph";

function fmtTime(s: number | null): string {
  if (s === null) return "";
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function downloadJson(detail: EventDetail) {
  const blob = new Blob([JSON.stringify(detail, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `event-${detail.id.slice(0, 8)}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

export function EventDetailPage({ eventId, onBack, onOpenGraph }: { eventId: string; onBack: () => void; onOpenGraph: () => void }) {
  const [detail, setDetail] = useState<EventDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDetail(null);
    setError(null);
    apiClient.getEventDetail(eventId).then(setDetail).catch((e) => setError(e.message));
  }, [eventId]);

  if (error) return <p className="font-mono text-xs text-amber">{error}</p>;
  if (!detail) return <p className="font-mono text-xs text-dim">Loading…</p>;

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <button onClick={onBack} className="flex items-center gap-1 font-mono text-[11px] text-dim hover:text-ink">
          ← Back to results
        </button>
        <div className="flex gap-2">
          <button onClick={() => downloadJson(detail)} className="rounded-md border border-hairline px-3 py-1.5 font-mono text-[11px] text-dim hover:text-ink">Export</button>
          <button onClick={onOpenGraph} className="rounded-md border border-signal/50 bg-signal/10 px-3 py-1.5 font-mono text-[11px] text-signal">Open in graph</button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_300px]">
        <div className="flex flex-col gap-5">
          <div className="card p-5">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="font-sans text-xl font-semibold text-ink">{detail.title}</h2>
              <span className="rounded bg-signal/15 px-2 py-0.5 font-mono text-[10px] uppercase text-signal">{detail.event_type}</span>
              <span className="font-mono text-[11px] text-dim">
                {detail.timestamp_start !== null ? `${fmtTime(detail.timestamp_start)} – ${fmtTime(detail.timestamp_end)}` : ""}
              </span>
              <span className="ml-auto font-mono text-[11px] text-signal">{Math.round(detail.confidence * 100)}% confidence</span>
            </div>
            <p className="mt-3 text-sm leading-relaxed text-dim">{detail.summary}</p>
          </div>

          <div>
            <p className="eyebrow mb-2">Evidence sources ({detail.evidence.length})</p>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {detail.evidence.map((ev) => {
                const img = mediaUrl(ev.raw_locator);
                return (
                  <div key={ev.id} className="card overflow-hidden">
                    <div className="flex items-center justify-between border-b border-hairline px-3 py-2">
                      <span className="flex items-center gap-1.5 font-mono text-[10px] uppercase text-signal">
                        <ModalityGlyph modality={ev.modality} className="h-3 w-3" />{modalityLabel(ev.modality)}
                      </span>
                      <span className="font-mono text-[10px] text-dim">{ev.timestamp_start !== null ? `${fmtTime(ev.timestamp_start)}` : ev.page_number ? `p.${ev.page_number}` : ""}</span>
                    </div>
                    {img ? (
                      <a href={img} target="_blank" rel="noopener noreferrer">
                        <img src={img} alt={ev.content_type} className="h-32 w-full object-cover" />
                      </a>
                    ) : (
                      <p className="line-clamp-4 px-3 py-3 text-[12px] text-ink">{ev.content_text}</p>
                    )}
                    <div className="flex items-center justify-between px-3 py-2">
                      <SignalBar value={ev.confidence} />
                      {img && <span className="font-mono text-[10px] text-dim">View full image ↗</span>}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div>
            <p className="eyebrow mb-2">Relationships ({detail.relationships.length})</p>
            {detail.relationships.length === 0 ? (
              <p className="font-mono text-[11px] text-dim">No direct relationships recorded for this event's evidence.</p>
            ) : (
              <div className="card divide-y divide-hairline">
                {detail.relationships.map((r) => (
                  <div key={r.id} className="flex items-center gap-2 px-4 py-2.5 font-mono text-[12px]">
                    <span className="truncate text-ink" title={r.source_label}>{r.source_label}</span>
                    <span className="flex-shrink-0 rounded bg-panel2 px-1.5 py-0.5 text-[10px] uppercase text-signal">{r.relationship_type}</span>
                    <span className="truncate text-ink" title={r.target_label}>{r.target_label}</span>
                    <span className="ml-auto flex-shrink-0 text-dim">{r.confidence.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="card p-4">
            <p className="eyebrow mb-3">Event details</p>
            <div className="flex flex-col gap-2 font-mono text-[11px]">
              {[
                ["Event ID", detail.id.slice(0, 8)],
                ["Type", detail.event_type],
                ["Start time", detail.timestamp_start !== null ? fmtTime(detail.timestamp_start) : "—"],
                ["End time", detail.timestamp_end !== null ? fmtTime(detail.timestamp_end) : "—"],
                ["Confidence", `${Math.round(detail.confidence * 100)}%`],
                ["Evidence count", String(detail.evidence.length)],
                ["Entity count", String(detail.entities.length)],
              ].map(([k, v]) => (
                <div key={k} className="flex items-center justify-between">
                  <span className="text-dim">{k}</span>
                  <span className="text-ink">{v}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card p-4">
            <p className="eyebrow mb-3">Entities</p>
            {detail.entities.length === 0 ? (
              <p className="font-mono text-[11px] text-dim">None resolved.</p>
            ) : (
              <div className="flex flex-col gap-2">
                {detail.entities.map((e) => (
                  <div key={e.id} className="flex items-center justify-between">
                    <span className="text-[13px] text-ink">{e.canonical_name}</span>
                    <span className="font-mono text-[10px] uppercase text-dim">{e.entity_type}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
