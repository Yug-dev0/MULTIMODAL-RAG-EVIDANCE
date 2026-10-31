interface Props {
  icon: React.ReactNode;
  value: string;
  label: string;
  sublabel?: string;
  accent?: "signal" | "amber";
}

export function StatCard({ icon, value, label, sublabel, accent = "signal" }: Props) {
  return (
    <div
      className="card flex flex-col gap-3 border-l-[3px] p-5"
      style={{ borderLeftColor: accent === "signal" ? "rgb(var(--c-signal))" : "rgb(var(--c-amber))" }}
    >
      <div className={`flex h-9 w-9 items-center justify-center rounded-md ${accent === "signal" ? "bg-signal/15 text-signal" : "bg-amber/15 text-amber"}`}>
        {icon}
      </div>
      <p className="font-sans text-[32px] font-semibold leading-none text-ink">{value}</p>
      <div>
        <p className="text-sm text-dim">{label}</p>
        {sublabel && <p className="mt-0.5 font-mono text-[11px] text-dim/70">{sublabel}</p>}
      </div>
    </div>
  );
}
