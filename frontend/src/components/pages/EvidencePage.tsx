import { useEffect, useMemo, useState } from "react";
import { apiClient, EvidenceRow, QueryEvidenceGroup } from "../../api/client";
import { ModalityGlyph, SignalBar, modalityLabel } from "../ModalityGlyph";
import { EvidenceDetail, LocationBadge } from "../EvidencePreview";

const BROWSE_LIMIT = 200;
// A search that returns the whole corpus re-sorted is still "showing me
// everything" -- the tail is noise no matter how well the head is ranked.
const SEARCH_LIMIT = 25;

function EvidenceCardButton({ u, selected, onSelect }: {
  u: EvidenceRow; selected: boolean; onSelect: () => void;
}) {
  return (
    <button
      data-testid="evidence-card"
      onClick={onSelect}
      className={`border bg-panel p-3 text-left transition-colors ${
        selected ? "border-signal/60 bg-signal/5" : "border-hairline hover:border-signal/30"
      }`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wide text-signal">
          <ModalityGlyph modality={u.modality} className="h-3 w-3" />
          {modalityLabel(u.modality)}
        </span>
        <LocationBadge u={u} />
      </div>
      <p className="mt-2 line-clamp-2 text-[13px] text-ink">{u.content_text}</p>
      <div className="mt-2 flex items-center justify-between gap-3">
        {u.relevance !== null ? (
          <span data-testid="relevance-score" className="font-mono text-[11px] text-signal">
            {Math.round(u.relevance * 100)}% match
          </span>
        ) : (
          <span />
        )}
        <span className="flex items-center gap-1.5" title="How confident extraction was that it read this correctly — not how well it matches your question">
          <span className="font-mono text-[10px] uppercase tracking-wide text-dim">Extraction</span>
          <SignalBar value={u.confidence} />
        </span>
      </div>
    </button>
  );
}

export function EvidencePage() {
  const [evidence, setEvidence] = useState<EvidenceRow[] | null>(null);
  const [queryGroups, setQueryGroups] = useState<QueryEvidenceGroup[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  // The query the CURRENT results answer -- not the input's live value, so the
  // "results for" heading can never describe a search that hasn't run yet.
  const [activeQuery, setActiveQuery] = useState("");
  const [searching, setSearching] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setSearching(true);
    apiClient
      .getEvidence(activeQuery ? SEARCH_LIMIT : BROWSE_LIMIT, activeQuery)
      .then((rows) => { if (!cancelled) setEvidence(rows); })
      .catch(() => { if (!cancelled) setEvidence([]); })
      .finally(() => { if (!cancelled) setSearching(false); });
    return () => { cancelled = true; };
  }, [activeQuery]);

  useEffect(() => {
    let cancelled = false;
    apiClient
      .getEvidenceByQuery()
      .then((groups) => { if (!cancelled) setQueryGroups(groups); })
      // Browsing the corpus still works without the question grouping, so a
      // failure here degrades to the plain listing instead of an error page.
      .catch(() => { if (!cancelled) setQueryGroups([]); });
    return () => { cancelled = true; };
  }, []);

  function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    setSelectedId(null);
    setActiveQuery(draft.trim());
  }

  function clearSearch() {
    setDraft("");
    setSelectedId(null);
    setActiveQuery("");
  }

  // Evidence no answer has ever cited. Kept visible so the page is still a
  // complete corpus browser, but under a heading that doesn't imply it
  // answered a question.
  const uncited = useMemo(() => {
    if (!evidence) return [];
    const citedIds = new Set(queryGroups.flatMap((g) => g.evidence.map((e) => e.id)));
    return evidence.filter((u) => !citedIds.has(u.id));
  }, [evidence, queryGroups]);

  const groups = useMemo(() => {
    if (!evidence) return [];
    const bySession = new Map<string, { title: string; items: EvidenceRow[] }>();
    for (const u of evidence) {
      const key = u.session_id ?? "__ungrouped__";
      const title = u.session_title ?? "Ungrouped";
      if (!bySession.has(key)) bySession.set(key, { title, items: [] });
      bySession.get(key)!.items.push(u);
    }
    const entries = Array.from(bySession.entries());
    entries.sort(([a], [b]) => (a === "__ungrouped__" ? 1 : b === "__ungrouped__" ? -1 : 0));
    return entries;
  }, [evidence]);

  const selected = evidence?.find((u) => u.id === selectedId) ?? null;

  return (
    <div>
      <p className="eyebrow mb-1">Explore</p>
      <h2 className="mb-3 font-sans text-xl font-semibold text-ink">Evidence</h2>

      <form onSubmit={handleSearch} className="mb-4 flex gap-2">
        <input
          data-testid="evidence-search"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask what you want evidence for — e.g. what are the specializations in AI/ML?"
          className="min-w-0 flex-1 rounded-md border border-hairline bg-panel px-3 py-2 font-mono text-xs text-ink placeholder:text-dim focus:border-signal/50 focus:outline-none"
        />
        <button
          type="submit"
          disabled={searching}
          className="rounded-md border border-signal/50 bg-signal/10 px-5 py-2 font-mono text-xs uppercase tracking-wide text-signal transition-colors hover:bg-signal/20 disabled:opacity-40"
        >
          {searching ? "Searching" : "Search"}
        </button>
        {activeQuery && (
          <button
            type="button"
            onClick={clearSearch}
            className="rounded-md border border-hairline px-4 py-2 font-mono text-xs uppercase tracking-wide text-dim transition-colors hover:text-ink"
          >
            Clear
          </button>
        )}
      </form>

      {evidence === null ? (
        <p className="font-mono text-xs text-dim">Loading…</p>
      ) : evidence.length === 0 ? (
        <p data-testid="evidence-empty" className="font-mono text-xs text-dim">
          {activeQuery ? `No evidence matches “${activeQuery}”.` : "No evidence ingested yet."}
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_440px]">
          <div className="flex flex-col gap-5">
            {activeQuery ? (
              // Ranked results are deliberately NOT grouped by chat session.
              // Session titles are generated from the first question asked, so
              // grouping under them made an unranked dump look like an answer.
              <div>
                <p data-testid="results-heading" className="eyebrow mb-2 flex items-center gap-2">
                  Best matches for “{activeQuery}”
                  <span className="rounded-full bg-panel2 px-1.5 py-0.5 font-mono text-[10px] text-dim">
                    top {evidence.length}
                  </span>
                </p>
                <div className="flex flex-col gap-2">
                  {evidence.map((u) => (
                    <EvidenceCardButton
                      key={u.id} u={u} selected={selectedId === u.id}
                      onSelect={() => setSelectedId(u.id)}
                    />
                  ))}
                </div>
              </div>
            ) : queryGroups.length > 0 ? (
              // The default view: each question you asked, carrying the
              // evidence its answer actually cited. Grouping by chat session
              // instead put one question's evidence in the same pile as
              // everything else ever ingested into that session.
              <>
                {queryGroups.map((group, i) => (
                  <div key={`${group.session_id}-${group.asked_at}-${i}`} data-testid={`query-group-${i}`}>
                    <p className="eyebrow mb-2 flex items-center gap-2">
                      <span className="flex-shrink-0 whitespace-nowrap text-dim">You asked:</span>
                      <span className="normal-case text-ink">“{group.query}”</span>
                      <span className="rounded-full bg-panel2 px-1.5 py-0.5 font-mono text-[10px] text-dim">
                        {group.evidence.length}
                      </span>
                    </p>
                    {group.evidence.length === 0 ? (
                      // An answer with no citations is worth showing as such
                      // rather than hiding the question entirely.
                      <p className="font-mono text-xs text-dim">
                        This answer cited no evidence.
                      </p>
                    ) : (
                      <div className="flex flex-col gap-2">
                        {group.evidence.map((u) => (
                          <EvidenceCardButton
                            key={u.id} u={u} selected={selectedId === u.id}
                            onSelect={() => setSelectedId(u.id)}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                ))}
                {uncited.length > 0 && (
                  <div data-testid="uncited-group">
                    <p className="eyebrow mb-2 flex items-center gap-2">
                      Not cited in any answer yet
                      <span className="rounded-full bg-panel2 px-1.5 py-0.5 font-mono text-[10px] text-dim">
                        {uncited.length}
                      </span>
                    </p>
                    <div className="flex flex-col gap-2">
                      {uncited.map((u) => (
                        <EvidenceCardButton
                          key={u.id} u={u} selected={selectedId === u.id}
                          onSelect={() => setSelectedId(u.id)}
                        />
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              groups.map(([key, group]) => (
                <div key={key}>
                  {/* Fallback for a corpus nothing has been asked about yet.
                      "All evidence from" is load-bearing: the title comes from
                      the session's first question, so a bare heading read as
                      "these rows answer that question" when they are simply
                      everything ingested in that session. */}
                  <p className="eyebrow mb-2 flex items-center gap-2">
                    {key === "__ungrouped__" ? group.title : `All evidence from: ${group.title}`}
                    <span className="rounded-full bg-panel2 px-1.5 py-0.5 font-mono text-[10px] text-dim">
                      {group.items.length}
                    </span>
                  </p>
                  <div className="flex flex-col gap-2">
                    {group.items.map((u) => (
                      <EvidenceCardButton
                        key={u.id} u={u} selected={selectedId === u.id}
                        onSelect={() => setSelectedId(u.id)}
                      />
                    ))}
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="lg:sticky lg:top-4 lg:self-start">
            <div className="bracket-frame border border-hairline bg-panel p-4">
              {selected ? (
                <EvidenceDetail u={selected} />
              ) : (
                <p className="py-10 text-center font-mono text-xs text-dim">
                  Select a piece of evidence to preview it here — video and audio open at their exact
                  timestamp, PDFs open on their exact page.
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
