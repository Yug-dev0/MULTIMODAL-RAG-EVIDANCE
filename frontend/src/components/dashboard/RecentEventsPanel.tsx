import type { EventRow } from "../../api/client";

function timeRange(ev: EventRow): string {
  if (ev.timestamp_start === null) return ev.created_at ? new Date(ev.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "";
  const fmt = (s: number) => {
    const m = Math.floor(s / 60), sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };
  return ev.timestamp_end !== null ? `${fmt(ev.timestamp_start)} – ${fmt(ev.timestamp_end)}` : fmt(ev.timestamp_start);
}

const TYPE_TAGS: Record<string, string> = {
  discussion: "Discussion", decision: "Decision", proposal: "Proposal", explanation: "Explanation",
  diagram_explanation: "Architecture", definition: "Definition", question: "Question", answer: "Answer",
  comparison: "Comparison", action_item: "Action item", observation: "Observation", reference: "Reference",
};

export function RecentEventsPanel({ events, onOpen }: { events: EventRow[]; onOpen: (id: string) => void }) {
  return (
    <div className="card flex h-full flex-col p-4">
      <div className="mb-3 flex items-center justify-between">
        <p className="eyebrow">Recent knowledge events</p>
        <span className="font-mono text-[10px] text-dim">{events.length}</span>
      </div>
      {events.length === 0 ? (
        <p className="font-mono text-[11px] text-dim">No knowledge events reconstructed yet — ingest a source to begin.</p>
      ) : (
        <div className="flex flex-col gap-2.5">
          {events.map((ev) => (
            <button
              key={ev.id}
              onClick={() => onOpen(ev.id)}
              className="flex flex-col items-start gap-1.5 rounded-md border border-hairline bg-panel2/50 p-3 text-left transition-colors hover:border-signal/40"
            >
              <div className="flex w-full items-center justify-between">
                <p className="truncate font-sans text-[14px] font-medium text-ink">{ev.title}</p>
                <span className="flex-shrink-0 font-mono text-[11px] text-dim">{timeRange(ev)}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="rounded bg-signal/15 px-1.5 py-0.5 font-mono text-[10px] uppercase text-signal">
                  {TYPE_TAGS[ev.event_type] ?? ev.event_type}
                </span>
                <span className="font-mono text-[10px] text-dim">{Math.round(ev.confidence * 100)}% confidence</span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
