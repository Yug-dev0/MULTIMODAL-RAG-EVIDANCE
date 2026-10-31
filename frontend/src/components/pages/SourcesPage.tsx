import { useEffect, useState } from "react";
import { apiClient, SourceRow } from "../../api/client";
import { ModalityGlyph } from "../ModalityGlyph";
import { VideoPlayer } from "../VideoPlayer";
import { PDFViewer } from "../PDFViewer";
import { mediaFileUrl, mediaUrl } from "../../api/media";

export function SourcesPage() {
  const [sources, setSources] = useState<SourceRow[] | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  useEffect(() => {
    apiClient.getSources().then(setSources).catch(() => setSources([]));
  }, []);

  async function handleDelete(sourceId: string) {
    if (!confirm("Delete this source and all evidence, events, and graph data derived from it? This can't be undone.")) return;
    setDeletingId(sourceId);
    try {
      await apiClient.deleteSource(sourceId);
      setSources((prev) => prev?.filter((s) => s.source_id !== sourceId) ?? null);
      if (selectedId === sourceId) setSelectedId(null);
    } catch {
      alert("Couldn't delete this source. Try again.");
    } finally {
      setDeletingId(null);
    }
  }

  const selected = sources?.find((s) => s.source_id === selectedId) ?? null;

  return (
    <div>
      <p className="eyebrow mb-1">Ingest</p>
      <h2 className="mb-4 font-sans text-xl font-semibold text-ink">Sources</h2>
      {sources === null ? (
        <p className="font-mono text-xs text-dim">Loading…</p>
      ) : sources.length === 0 ? (
        <p className="font-mono text-xs text-dim">No sources ingested yet.</p>
      ) : (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_440px]">
          <div className="overflow-x-auto border border-hairline">
            <table className="w-full text-left text-sm">
              <thead className="bg-panel2 font-mono text-[10px] uppercase tracking-wide text-dim">
                <tr>
                  <th className="px-3 py-2">Type</th>
                  <th className="px-3 py-2">Source ID</th>
                  <th className="px-3 py-2">Evidence</th>
                  <th className="px-3 py-2">Duration</th>
                  <th className="px-3 py-2">First ingested</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {sources.map((s) => (
                  <tr
                    key={s.source_id}
                    onClick={() => setSelectedId(s.source_id)}
                    className={`cursor-pointer border-t border-hairline/60 transition-colors hover:bg-panel2/60 ${
                      selectedId === s.source_id ? "bg-signal/5" : ""
                    }`}
                  >
                    <td className="px-3 py-2">
                      <span className="flex items-center gap-1.5 text-ink">
                        <ModalityGlyph modality={s.source_type === "pdf" ? "document" : s.source_type} className="h-3.5 w-3.5" />
                        {s.source_type}
                      </span>
                    </td>
                    <td className="px-3 py-2 font-mono text-xs text-dim">{s.source_id.slice(0, 8)}</td>
                    <td className="px-3 py-2 text-ink">{s.evidence_count}</td>
                    <td className="px-3 py-2 text-dim">{s.duration_seconds ? `${s.duration_seconds.toFixed(1)}s` : "—"}</td>
                    <td className="px-3 py-2 font-mono text-xs text-dim">{s.first_seen ? new Date(s.first_seen).toLocaleString() : "—"}</td>
                    <td className="px-3 py-2 text-right">
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDelete(s.source_id); }}
                        disabled={deletingId === s.source_id}
                        className="rounded border border-hairline px-2 py-1 font-mono text-[10px] uppercase tracking-wide text-dim transition-colors hover:border-red-400/50 hover:text-red-400 disabled:opacity-40"
                      >
                        {deletingId === s.source_id ? "Deleting…" : "Delete"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="lg:sticky lg:top-4 lg:self-start">
            <div className="bracket-frame border border-hairline bg-panel p-4">
              {selected ? <SourcePreview source={selected} /> : (
                <p className="py-10 text-center font-mono text-xs text-dim">
                  Select a source to open it — PDFs open on page one, video and audio play from the start.
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function SourcePreview({ source }: { source: SourceRow }) {
  const fileUrl = mediaFileUrl(source.raw_locator);
  const imageUrl = mediaUrl(source.raw_locator);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-wide text-signal">
        <ModalityGlyph modality={source.source_type === "pdf" ? "document" : source.source_type} className="h-3.5 w-3.5" />
        {source.source_type}
      </div>

      {!fileUrl && !imageUrl && (
        <p className="font-mono text-xs text-dim">This source's file isn't available to preview.</p>
      )}
      {(source.source_type === "video" || source.source_type === "audio") && fileUrl && (
        <VideoPlayer src={fileUrl} />
      )}
      {source.source_type === "pdf" && fileUrl && <PDFViewer src={fileUrl} />}
      {source.source_type === "image" && imageUrl && (
        <div className="bracket-frame border border-hairline bg-panel p-1">
          <img src={imageUrl} alt="Source" className="w-full" />
        </div>
      )}
    </div>
  );
}
