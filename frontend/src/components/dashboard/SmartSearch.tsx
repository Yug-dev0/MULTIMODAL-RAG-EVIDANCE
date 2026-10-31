import { useState } from "react";

const EXAMPLES = ["Show me the caching architecture", "Who discussed Redis?", "Where was the diagram shown?"];

export function SmartSearch({ onSearch }: { onSearch: (q: string) => void }) {
  const [value, setValue] = useState("");

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (value.trim()) onSearch(value.trim());
  }

  return (
    <div className="bracket-frame border border-hairline bg-panel p-5">
      <p className="eyebrow mb-2 flex items-center gap-1.5 text-signal">
        <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2"><path strokeLinecap="round" strokeLinejoin="round" d="M12 3l1.8 4.6L18 9l-4.2 1.4L12 15l-1.8-4.6L6 9l4.2-1.4z" /></svg>
        Smart search
      </p>
      <p className="mb-3 font-sans text-sm text-dim">Ask questions across your videos, audio, images, and documents.</p>
      <form onSubmit={submit} className="flex gap-2">
        <div className="flex flex-1 items-center gap-2 rounded-md border border-hairline bg-panel px-3">
          <svg viewBox="0 0 24 24" className="h-4 w-4 text-dim" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="11" cy="11" r="7" /><path strokeLinecap="round" d="M21 21l-4.3-4.3" /></svg>
          <input
            className="flex-1 bg-transparent py-2.5 font-sans text-sm text-ink placeholder:text-dim focus:outline-none"
            placeholder="What architecture was proposed to reduce database load and who explained it?"
            value={value}
            onChange={(e) => setValue(e.target.value)}
          />
        </div>
        <button type="submit" disabled={!value.trim()} className="rounded-md bg-signal px-5 py-2.5 font-mono text-xs uppercase tracking-wide text-void disabled:opacity-40">
          Search
        </button>
      </form>
      <div className="mt-3 flex flex-wrap gap-2">
        <span className="font-mono text-[10px] uppercase tracking-wide text-dim">Examples:</span>
        {EXAMPLES.map((ex) => (
          <button key={ex} onClick={() => onSearch(ex)} className="rounded-full border border-hairline px-2.5 py-1 font-sans text-[11px] text-dim transition-colors hover:border-signal/50 hover:text-signal">
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}
