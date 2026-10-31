const LABELS: Record<string, string> = {
  speech: "Speech",
  ocr: "On-screen text",
  image: "Visual",
  document: "Document",
};

export function modalityLabel(modality: string): string {
  return LABELS[modality] ?? modality;
}

/** Small inline glyph per modality: waveform / scanlines / aperture / folio-corner. */
export function ModalityGlyph({ modality, className = "h-4 w-4" }: { modality: string; className?: string }) {
  const stroke = "currentColor";
  switch (modality) {
    case "speech":
      return (
        <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden="true">
          <path d="M2 12h2l1.5-5 2 10 2-14 2 16 2-10 1.5 3H22" stroke={stroke} strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      );
    case "ocr":
      return (
        <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden="true">
          <path d="M3 6h18M3 11h12M3 16h18M3 21h8" stroke={stroke} strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      );
    case "image":
      return (
        <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden="true">
          <circle cx="12" cy="12" r="9" stroke={stroke} strokeWidth="1.6" />
          <circle cx="12" cy="12" r="3.2" stroke={stroke} strokeWidth="1.6" />
          <path d="M12 3v3M12 18v3M21 12h-3M6 12H3" stroke={stroke} strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      );
    case "document":
      return (
        <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden="true">
          <path d="M6 2h9l5 5v15H6z" stroke={stroke} strokeWidth="1.6" strokeLinejoin="round" />
          <path d="M15 2v5h5" stroke={stroke} strokeWidth="1.6" strokeLinejoin="round" />
          <path d="M9 13h6M9 17h6" stroke={stroke} strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      );
    default:
      return (
        <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden="true">
          <circle cx="12" cy="12" r="9" stroke={stroke} strokeWidth="1.6" />
        </svg>
      );
  }
}

/** Confidence rendered as a VU-meter-style signal bar, not a percentage badge. */
export function SignalBar({ value, className = "" }: { value: number; className?: string }) {
  const pct = Math.max(0, Math.min(1, value));
  const bars = 10;
  const filled = Math.round(pct * bars);
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <div className="flex gap-[2px]" role="img" aria-label={`confidence ${Math.round(pct * 100)}%`}>
        {Array.from({ length: bars }).map((_, i) => (
          <span
            key={i}
            className={`h-3 w-[3px] rounded-sm ${i < filled ? (pct > 0.7 ? "bg-signal" : "bg-amber") : "bg-hairline"}`}
          />
        ))}
      </div>
      <span className="font-mono text-[11px] text-dim">{Math.round(pct * 100)}%</span>
    </div>
  );
}
