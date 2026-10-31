export function SystemOverviewRing({ pct, processedSeconds, totalSeconds }: { pct: number; processedSeconds: number; totalSeconds: number }) {
  const r = 34;
  const c = 2 * Math.PI * r;
  const offset = c - (pct / 100) * c;
  const fmtHrs = (s: number) => (s / 3600).toFixed(1);

  return (
    <div className="border border-hairline bg-panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <p className="eyebrow">System overview</p>
        <span className="flex items-center gap-1 font-mono text-[10px] text-signal"><span className="h-1.5 w-1.5 rounded-full bg-signal animate-pulse2" /> live</span>
      </div>
      <div className="flex items-center gap-4">
        <svg viewBox="0 0 80 80" className="h-20 w-20 -rotate-90">
          <circle cx="40" cy="40" r={r} fill="none" stroke="rgb(var(--c-hairline))" strokeWidth="7" />
          <circle cx="40" cy="40" r={r} fill="none" stroke="rgb(var(--c-signal))" strokeWidth="7" strokeLinecap="round"
                  strokeDasharray={c} strokeDashoffset={offset} style={{ transition: "stroke-dashoffset 0.6s ease" }} />
        </svg>
        <div>
          <p className="font-sans text-2xl font-semibold text-ink">{pct}%</p>
          <p className="text-xs text-dim">Ingestion jobs complete</p>
          <p className="mt-1 font-mono text-[11px] text-dim">
            {fmtHrs(processedSeconds)} / {fmtHrs(totalSeconds || processedSeconds)} hrs media processed
          </p>
        </div>
      </div>
    </div>
  );
}
