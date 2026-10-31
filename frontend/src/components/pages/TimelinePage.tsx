import { useEffect, useState } from "react";
import { apiClient, EventRow } from "../../api/client";

export function TimelinePage({ onOpen }: { onOpen: (id: string) => void }) {
  const [events, setEvents] = useState<EventRow[] | null>(null);

  useEffect(() => {
    apiClient.getEvents(200).then((rows) => {
      const sorted = [...rows].sort((a, b) => {
        if (a.timestamp_start === null && b.timestamp_start === null) return (a.created_at ?? "").localeCompare(b.created_at ?? "");
        if (a.timestamp_start === null) return 1;
        if (b.timestamp_start === null) return -1;
        return a.timestamp_start - b.timestamp_start;
      });
      setEvents(sorted);
    }).catch(() => setEvents([]));
  }, []);

  return (
    <div>
      <p className="eyebrow mb-1">Tools</p>
      <h2 className="mb-4 font-sans text-xl font-semibold text-ink">Timeline</h2>
      {events === null ? (
        <p className="font-mono text-xs text-dim">Loading…</p>
      ) : events.length === 0 ? (
        <p className="font-mono text-xs text-dim">No events to place on a timeline yet.</p>
      ) : (
        <div className="relative border-l border-hairline pl-6">
          {events.map((ev) => (
            <button key={ev.id} onClick={() => onOpen(ev.id)} className="relative block w-full pb-6 text-left">
              <span className="absolute -left-[29px] top-1 h-2.5 w-2.5 rounded-full border-2 border-void bg-signal" />
              <p className="font-mono text-[10px] text-dim">
                {ev.timestamp_start !== null ? `t = ${ev.timestamp_start.toFixed(1)}s` : ev.created_at ? new Date(ev.created_at).toLocaleString() : "unordered"}
              </p>
              <p className="font-sans text-sm text-ink hover:text-signal">{ev.title}</p>
              <p className="text-[12px] text-dim">{ev.summary}</p>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
