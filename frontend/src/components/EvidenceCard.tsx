import { BASE_URL, Citation } from "../api/client";
import { ModalityGlyph, SignalBar, modalityLabel } from "./ModalityGlyph";
import { LocationBadge } from "./EvidencePreview";

export function EvidenceCard({ citation, onClick }: { citation: Citation; onClick?: () => void }) {
  const Wrapper = onClick ? "button" : "div";

  return (
    <Wrapper
      onClick={onClick}
      className={`group w-full rounded-md border border-hairline bg-panel p-4 text-left transition-colors hover:border-signal/40 ${
        onClick ? "cursor-pointer" : ""
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-signal">
          <ModalityGlyph modality={citation.modality} className="h-4 w-4" />
          <span className="eyebrow text-signal/90">{modalityLabel(citation.modality)}</span>
        </div>
        <LocationBadge
          u={{
            modality: citation.modality, raw_locator: citation.source, content_text: citation.content,
            timestamp_start: citation.timestamp_start, timestamp_end: citation.timestamp_end,
            page_number: citation.page_number, confidence: citation.confidence,
          }}
        />
      </div>

      {citation.thumbnail && (
        <img
          src={`${BASE_URL}${citation.thumbnail}`}
          alt={`${modalityLabel(citation.modality)} evidence`}
          className="mt-3 max-h-40 w-full rounded border border-hairline object-cover"
          loading="lazy"
          onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
        />
      )}

      <p className="mt-3 line-clamp-4 text-sm text-ink">{citation.content}</p>

      <div className="mt-3 flex items-center justify-between gap-3">
        <SignalBar value={citation.confidence} />
        {onClick && (
          <span className="font-mono text-[10px] uppercase tracking-wide text-dim opacity-0 transition-opacity group-hover:opacity-100 group-hover:text-signal">
            View source →
          </span>
        )}
      </div>
    </Wrapper>
  );
}
