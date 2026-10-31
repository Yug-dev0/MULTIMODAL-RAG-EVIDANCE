import { mediaUrl, mediaFileUrl } from "../api/media";
import { ModalityGlyph, SignalBar, modalityLabel } from "./ModalityGlyph";
import { VideoPlayer } from "./VideoPlayer";
import { PDFViewer } from "./PDFViewer";

/** Common shape both EvidenceRow (the Evidence page) and Citation (chat
 *  answers) can be read as, so the same accurate, click-to-preview detail
 *  view works in both places instead of two separate implementations. */
export interface PreviewableEvidence {
  modality: string;
  raw_locator: string;
  content_text: string;
  timestamp_start: number | null;
  timestamp_end: number | null;
  page_number: number | null;
  confidence: number;
  source_type?: string;
}

export function formatTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const mm = h > 0 ? String(m).padStart(2, "0") : String(m);
  return h > 0 ? `${h}:${mm}:${String(sec).padStart(2, "0")}` : `${mm}:${String(sec).padStart(2, "0")}`;
}

export function LocationBadge({ u, className = "" }: { u: PreviewableEvidence; className?: string }) {
  if (u.timestamp_start !== null) {
    const label =
      u.timestamp_end !== null && u.timestamp_end !== u.timestamp_start
        ? `${formatTime(u.timestamp_start)}–${formatTime(u.timestamp_end)}`
        : formatTime(u.timestamp_start);
    return (
      <span className={`flex-shrink-0 rounded border border-signal/30 bg-signal/10 px-2 py-0.5 font-mono text-[11px] text-signal ${className}`}>
        {label}
      </span>
    );
  }
  if (u.page_number !== null) {
    return (
      <span className={`flex-shrink-0 rounded border border-signal/30 bg-signal/10 px-2 py-0.5 font-mono text-[11px] text-signal ${className}`}>
        Page {u.page_number}
      </span>
    );
  }
  return null;
}

/** The exact-timestamp / exact-page evidence preview: a real player seeked
 *  to where the evidence actually occurred, not just a text summary. */
export function EvidenceDetail({ u }: { u: PreviewableEvidence }) {
  const fileUrl = mediaFileUrl(u.raw_locator);
  const imageUrl = mediaUrl(u.raw_locator);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-wide text-signal">
          <ModalityGlyph modality={u.modality} className="h-3.5 w-3.5" />
          {modalityLabel(u.modality)}
        </span>
        <LocationBadge u={u} />
      </div>

      {u.modality === "speech" && fileUrl && (
        <VideoPlayer src={fileUrl} seekToSeconds={u.timestamp_start ?? undefined} />
      )}
      {u.modality === "document" && fileUrl && (
        <PDFViewer src={fileUrl} pageNumber={u.page_number ?? undefined} />
      )}
      {(u.modality === "ocr" || u.modality === "image") && imageUrl && (
        <div className="bracket-frame border border-hairline bg-panel p-1">
          <img src={imageUrl} alt={modalityLabel(u.modality)} className="w-full" />
        </div>
      )}

      <div>
        <p className="eyebrow mb-1.5">
          {u.modality === "document" ? "Extracted text on this page" : "What this evidence shows"}
        </p>
        <p className="whitespace-pre-wrap rounded-md border border-hairline bg-panel2 p-3 font-sans text-[13px] leading-relaxed text-ink">
          {u.content_text}
        </p>
      </div>

      <div className="flex items-center justify-between border-t border-hairline pt-3">
        <SignalBar value={u.confidence} />
        {u.source_type && <span className="font-mono text-[10px] text-dim">{u.source_type}</span>}
      </div>
    </div>
  );
}
