import { useEffect, useState } from "react";
import { apiClient, EventRow } from "../../api/client";

export function EventsPage({ onOpen }: { onOpen: (id: string) => void }) {
  const [events, setEvents] = useState<EventRow[] | null>(null);

  useEffect(() => {
    apiClient.getEvents(100).then(setEvents).catch(() => setEvents([]));
  }, []);

  return (
    <div>
      <p className="eyebrow mb-1">Explore</p>
      <h2 className="mb-4 font-sans text-2xl font-semibold text-ink">Knowledge events</h2>
      {events === null ? (
        <p className="font-mono text-xs text-dim">Loading…</p>
      ) : events.length === 0 ? (
        <p className="font-mono text-xs text-dim">No knowledge events assembled yet.</p>
      ) : (
        <div className="flex flex-col gap-2">
          {events.map((ev) => (
            <button
              key={ev.id}
              onClick={() => onOpen(ev.id)}
              className="card flex flex-col items-start gap-1 p-4 text-left transition-colors hover:border-signal/40"
            >
              <div className="flex w-full items-center justify-between">
                <span className="rounded bg-signal/15 px-2 py-0.5 font-mono text-[10px] uppercase text-signal">{ev.event_type}</span>
                <span className="font-mono text-[10px] text-dim">{Math.round(ev.confidence * 100)}% confidence</span>
              </div>
              <p className="mt-2 font-sans text-sm text-ink">{ev.title}</p>
              <p className="mt-1 text-[12px] text-dim">{ev.summary}</p>
              <div className="mt-1 flex items-center gap-3 font-mono text-[10px] text-dim">
                {ev.timestamp_start !== null && <span>t {ev.timestamp_start.toFixed(1)}–{ev.timestamp_end?.toFixed(1)}s</span>}
                <span>{ev.evidence_count ?? 0} evidence</span>
                <span>{ev.relationship_count ?? 0} relationships</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
