import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph from "force-graph";
import { THEME_CHANGE_EVENT } from "../hooks/useTheme";
import { drawNodeGlyph } from "./graph/graphIcons";
import { dashPatternFor, relationshipStylesInUse } from "./graph/graphStyle";

function readCssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function rgb(varName: string, alpha = 1): string {
  const parts = readCssVar(varName).split(/\s+/).map(Number);
  if (parts.length !== 3 || parts.some(Number.isNaN)) return `rgba(110,235,195,${alpha})`;
  return `rgba(${parts[0]},${parts[1]},${parts[2]},${alpha})`;
}

export interface GraphNode {
  id: string; label: string; type: "entity" | "evidence" | "event";
  entity_type?: string | null; aliases?: string[] | null; description?: string | null;
  event_type?: string | null; summary?: string | null;
  modality?: string | null; content_text?: string | null; source_id?: string | null;
  timestamp_start?: number | null; timestamp_end?: number | null; confidence?: number | null;
}
export interface GraphLink { source: string; target: string; relationship_type: string; confidence?: number | null; }

const NODE_COLOR: Record<string, string> = { entity: "#F5B052", event: "#6EEBC3", evidence: "#7FA8DE" };
const TYPE_LABEL: Record<string, string> = { entity: "Entity", event: "Knowledge Event", evidence: "Evidence Unit" };
const TYPE_FILTERS = ["all", "entity", "event", "evidence"] as const;
type TypeFilter = (typeof TYPE_FILTERS)[number];
type Tab = "overview" | "connections" | "events" | "evidence";

interface Props {
  nodes: GraphNode[];
  links: GraphLink[];
  active: boolean;
  onOpenEvent?: (id: string) => void;
}

export function GraphView({ nodes, links, active, onOpenEvent }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<ForceGraph | null>(null);
  const seenIds = useRef<Set<string>>(new Set());
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [tab, setTab] = useState<Tab>("overview");
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<TypeFilter>("all");
  const [lastUpdated, setLastUpdated] = useState(Date.now());

  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);
  const filteredNodes = useMemo(() => (typeFilter === "all" ? nodes : nodes.filter((n) => n.type === typeFilter)), [nodes, typeFilter]);
  const filteredIds = useMemo(() => new Set(filteredNodes.map((n) => n.id)), [filteredNodes]);
  const filteredLinks = useMemo(() => links.filter((l) => filteredIds.has(String(l.source)) && filteredIds.has(String(l.target))), [links, filteredIds]);
  const highlightIds = useMemo(() => {
    if (!search.trim()) return null;
    const q = search.toLowerCase();
    return new Set(nodes.filter((n) => n.label.toLowerCase().includes(q)).map((n) => n.id));
  }, [nodes, search]);

  // ---- Everything below is derived client-side from the already-loaded
  // graph snapshot -- no extra backend calls needed for the detail panel.
  const allConnections = useMemo(() => {
    if (!selected) return [];
    return links
      .map((l) => {
        const sId = typeof l.source === "string" ? l.source : (l.source as any).id;
        const tId = typeof l.target === "string" ? l.target : (l.target as any).id;
        if (sId !== selected.id && tId !== selected.id) return null;
        const otherId = sId === selected.id ? tId : sId;
        const other = byId.get(otherId);
        if (!other) return null;
        return { node: other, relationship: l.relationship_type, confidence: l.confidence ?? null, outgoing: sId === selected.id };
      })
      .filter((c): c is NonNullable<typeof c> => c !== null);
  }, [selected, links, byId]);

  const connectionsByType = useMemo(() => {
    const groups = new Map<string, typeof allConnections>();
    for (const c of allConnections) {
      if (!groups.has(c.relationship)) groups.set(c.relationship, []);
      groups.get(c.relationship)!.push(c);
    }
    return [...groups.entries()];
  }, [allConnections]);

  // For an Entity: evidence units mentioning it (MENTIONS edges), and the
  // distinct knowledge events those evidence units belong to (BELONGS_TO).
  const entityEvidence = useMemo(() => {
    if (!selected || selected.type !== "entity") return [];
    return allConnections.filter((c) => c.relationship.toLowerCase() === "mentions" && c.node.type === "evidence").map((c) => c.node);
  }, [selected, allConnections]);

  const entityEvents = useMemo(() => {
    if (!selected || selected.type !== "entity") return [];
    const evidenceIds = new Set(entityEvidence.map((e) => e.id));
    const eventIds = new Set<string>();
    for (const l of links) {
      const sId = typeof l.source === "string" ? l.source : (l.source as any).id;
      const tId = typeof l.target === "string" ? l.target : (l.target as any).id;
      if (l.relationship_type.toLowerCase() === "belongs_to" && evidenceIds.has(sId)) eventIds.add(tId);
    }
    return [...eventIds].map((id) => byId.get(id)).filter((n): n is GraphNode => !!n);
  }, [selected, entityEvidence, links, byId]);

  // For an Event: its member evidence units (BELONGS_TO edges pointing to it).
  const eventEvidence = useMemo(() => {
    if (!selected || selected.type !== "event") return [];
    return links
      .filter((l) => {
        const sId = typeof l.source === "string" ? l.source : (l.source as any).id;
        const tId = typeof l.target === "string" ? l.target : (l.target as any).id;
        return l.relationship_type.toLowerCase() === "belongs_to" && tId === selected.id && byId.get(sId);
      })
      .map((l) => byId.get(typeof l.source === "string" ? l.source : (l.source as any).id)!);
  }, [selected, links, byId]);

  const usedRelationshipTypes = useMemo(() => relationshipStylesInUse(links.map((l) => l.relationship_type)), [links]);

  // ---- Canvas graph lifecycle: created once, updated in place. ----
  useEffect(() => {
    if (!containerRef.current) return;
    const fg = new ForceGraph(containerRef.current)
      .backgroundColor(rgb("--c-void"))
      .nodeRelSize(6)
      .nodeCanvasObject((node: any, ctx, globalScale) => {
        const color = seenIds.current.has(node.id) ? NODE_COLOR[node.type] ?? "#6EEBC3" : "#FFFFFF";
        const opacity = highlightRef.current ? (highlightRef.current.has(node.id) ? 1 : 0.15) : 1;
        ctx.globalAlpha = opacity;
        drawNodeGlyph(ctx, node, node.x, node.y, 9, color);
        const fontSize = Math.max(3.4, 11 / globalScale);
        ctx.font = `${fontSize}px sans-serif`;
        ctx.textAlign = "center";
        ctx.fillStyle = rgb("--c-ink", opacity);
        const label = node.label.length > 22 ? `${node.label.slice(0, 20)}…` : node.label;
        ctx.fillText(label, node.x, node.y + 16);
        ctx.globalAlpha = 1;
      })
      .nodePointerAreaPaint((node: any, color, ctx) => {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(node.x, node.y, 10, 0, 2 * Math.PI);
        ctx.fill();
      })
      .linkCanvasObjectMode(() => "replace")
      .linkCanvasObject((link: any, ctx, globalScale) => {
        const s = link.source, t = link.target;
        if (!s || !t || typeof s === "string" || typeof t === "string") return;
        ctx.save();
        ctx.setLineDash(dashPatternFor(link.relationship_type));
        ctx.strokeStyle = rgb("--c-signal", 0.45);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(s.x, s.y);
        ctx.lineTo(t.x, t.y);
        ctx.stroke();
        ctx.setLineDash([]);

        // arrowhead
        const angle = Math.atan2(t.y - s.y, t.x - s.x);
        const ah = 4;
        const ax = t.x - 10 * Math.cos(angle), ay = t.y - 10 * Math.sin(angle);
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(ax - ah * Math.cos(angle - Math.PI / 6), ay - ah * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(ax - ah * Math.cos(angle + Math.PI / 6), ay - ah * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fillStyle = rgb("--c-signal", 0.6);
        ctx.fill();

        if (globalScale > 1.1) {
          const midX = (s.x + t.x) / 2, midY = (s.y + t.y) / 2;
          const fontSize = Math.max(2.6, 8 / globalScale);
          ctx.font = `${fontSize}px monospace`;
          const label = link.relationship_type;
          const w = ctx.measureText(label).width;
          ctx.fillStyle = rgb("--c-panel", 0.85);
          ctx.fillRect(midX - w / 2 - 2, midY - fontSize / 2 - 1, w + 4, fontSize + 2);
          ctx.fillStyle = rgb("--c-ink");
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(label, midX, midY);
        }
        ctx.restore();
      })
      .onNodeClick((n: any) => { setSelected(n); setTab("overview"); })
      .onNodeHover((n) => { if (containerRef.current) containerRef.current.style.cursor = n ? "pointer" : "default"; });

    fgRef.current = fg;

    const ro = new ResizeObserver(() => {
      if (!containerRef.current) return;
      fg.width(containerRef.current.clientWidth).height(containerRef.current.clientHeight);
    });
    ro.observe(containerRef.current);

    const onThemeChange = () => fg.backgroundColor(rgb("--c-void"));
    window.addEventListener(THEME_CHANGE_EVENT, onThemeChange);

    return () => {
      window.removeEventListener(THEME_CHANGE_EVENT, onThemeChange);
      ro.disconnect();
      fg.pauseAnimation();
      fgRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const highlightRef = useRef<Set<string> | null>(null);
  useEffect(() => {
    highlightRef.current = highlightIds;
  }, [highlightIds]);

  useEffect(() => {
    if (!fgRef.current) return;
    const isFirstLoad = seenIds.current.size === 0;
    fgRef.current.graphData({ nodes: filteredNodes as any, links: filteredLinks as any });
    setLastUpdated(Date.now());
    if (!isFirstLoad) {
      setTimeout(() => nodes.forEach((n) => seenIds.current.add(n.id)), 1800);
    } else {
      nodes.forEach((n) => seenIds.current.add(n.id));
    }
  }, [filteredNodes, filteredLinks, nodes]);

  useEffect(() => {
    if (!fgRef.current) return;
    if (active) fgRef.current.resumeAnimation();
    else fgRef.current.pauseAnimation();
  }, [active]);

  function fitToView() {
    fgRef.current?.zoomToFit(500, 60);
  }
  function zoomBy(factor: number) {
    if (!fgRef.current) return;
    fgRef.current.zoom(fgRef.current.zoom() * factor, 300);
  }
  function toggleFullscreen() {
    if (!wrapRef.current) return;
    if (document.fullscreenElement) document.exitFullscreen();
    else wrapRef.current.requestFullscreen();
  }

  function fmtTime(s: number | null | undefined): string {
    if (s === null || s === undefined) return "—";
    const m = Math.floor(s / 60), sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, "0")}`;
  }

  return (
    <div ref={wrapRef} className="bg-void">
      <div className="mb-3">
        <p className="eyebrow mb-1">Explore</p>
        <h2 className="font-sans text-2xl font-semibold text-ink">Knowledge graph</h2>
        <p className="mt-1 text-sm text-dim">Explore entities, events, and relationships across your corpus.</p>
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="flex flex-1 items-center gap-2 rounded-md border border-hairline bg-panel px-3 py-2 md:max-w-xs">
          <svg viewBox="0 0 24 24" className="h-4 w-4 text-dim" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="11" cy="11" r="7" /><path strokeLinecap="round" d="M21 21l-4.3-4.3" /></svg>
          <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search nodes…" className="w-full bg-transparent text-sm text-ink placeholder:text-dim focus:outline-none" />
        </div>
        <div className="flex gap-1 rounded-md border border-hairline bg-panel p-1">
          {TYPE_FILTERS.map((t) => (
            <button key={t} onClick={() => setTypeFilter(t)}
                    className={`rounded px-2.5 py-1 font-mono text-[11px] uppercase transition-colors ${typeFilter === t ? "bg-signal/15 text-signal" : "text-dim hover:text-ink"}`}>
              {t}
            </button>
          ))}
        </div>
        <button onClick={fitToView} className="rounded-md border border-hairline px-3 py-1.5 font-mono text-[11px] text-dim hover:text-ink">Fit to view</button>
        <button onClick={() => zoomBy(1.4)} className="rounded-md border border-hairline p-1.5 text-dim hover:text-ink" aria-label="Zoom in">
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="11" cy="11" r="7" /><path strokeLinecap="round" d="M21 21l-4.3-4.3M11 8v6M8 11h6" /></svg>
        </button>
        <button onClick={() => zoomBy(1 / 1.4)} className="rounded-md border border-hairline p-1.5 text-dim hover:text-ink" aria-label="Zoom out">
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="11" cy="11" r="7" /><path strokeLinecap="round" d="M21 21l-4.3-4.3M8 11h6" /></svg>
        </button>
        <button onClick={toggleFullscreen} className="rounded-md border border-hairline p-1.5 text-dim hover:text-ink" aria-label="Fullscreen">
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8"><path strokeLinecap="round" strokeLinejoin="round" d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5" /></svg>
        </button>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        {/* force-graph takes full ownership of containerRef's DOM children (it
            clears and repopulates them itself), so the "no data" hint lives in
            a sibling rather than inside that ref — otherwise React's render of
            it gets wiped out by the library on mount. */}
        <div className="bracket-frame relative h-[72vh] min-h-[520px] w-full border border-hairline bg-void" data-testid="graph-view">
          <div className="absolute inset-0" ref={containerRef} />
          {nodes.length === 0 && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <p className="font-mono text-xs text-dim">No graph data yet — ingest a source to populate the corpus.</p>
            </div>
          )}
        </div>

        <div className="card flex h-[72vh] min-h-[520px] flex-col overflow-hidden">
          {!selected ? (
            <div className="flex flex-1 items-center justify-center p-4 text-center">
              <p className="font-mono text-[11px] text-dim">Click a node to inspect its details and connections.</p>
            </div>
          ) : (
            <>
              <div className="border-b border-hairline p-4">
                <div className="mb-2 flex items-center justify-between">
                  <p className="eyebrow">Node details</p>
                  <button onClick={() => setSelected(null)} className="font-mono text-xs text-dim hover:text-ink">✕</button>
                </div>
                <div className="flex items-center gap-3">
                  <span className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full" style={{ background: `${NODE_COLOR[selected.type]}26`, border: `1.5px solid ${NODE_COLOR[selected.type]}` }}>
                    <span className="h-2.5 w-2.5 rounded-full" style={{ background: NODE_COLOR[selected.type] }} />
                  </span>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="truncate font-sans text-base font-semibold text-ink">{selected.label}</p>
                      <span className="flex-shrink-0 rounded px-1.5 py-0.5 font-mono text-[9px] uppercase" style={{ background: `${NODE_COLOR[selected.type]}26`, color: NODE_COLOR[selected.type] }}>
                        {TYPE_LABEL[selected.type]}
                      </span>
                    </div>
                    <p className="font-mono text-[10px] text-dim">{selected.id.slice(0, 8)}</p>
                  </div>
                </div>

                <div className="mt-3 flex flex-col gap-1 font-mono text-[11px]">
                  {selected.type === "entity" && (
                    <>
                      <div className="flex justify-between"><span className="text-dim">Type</span><span className="text-ink">{selected.entity_type ?? "—"}</span></div>
                      <div className="flex justify-between"><span className="text-dim">Mentions</span><span className="text-ink">{entityEvidence.length} across {entityEvents.length} event{entityEvents.length === 1 ? "" : "s"}</span></div>
                      {selected.aliases && selected.aliases.length > 0 && (
                        <div className="flex justify-between gap-2"><span className="flex-shrink-0 text-dim">Aliases</span><span className="truncate text-ink">{selected.aliases.join(", ")}</span></div>
                      )}
                    </>
                  )}
                  {selected.type === "event" && (
                    <>
                      <div className="flex justify-between"><span className="text-dim">Type</span><span className="text-ink">{selected.event_type ?? "—"}</span></div>
                      <div className="flex justify-between"><span className="text-dim">Time</span><span className="text-ink">{fmtTime(selected.timestamp_start)} – {fmtTime(selected.timestamp_end)}</span></div>
                    </>
                  )}
                  {selected.type === "evidence" && (
                    <>
                      <div className="flex justify-between"><span className="text-dim">Modality</span><span className="text-ink">{selected.modality ?? "—"}</span></div>
                      <div className="flex justify-between"><span className="text-dim">Time</span><span className="text-ink">{fmtTime(selected.timestamp_start)}</span></div>
                    </>
                  )}
                  {selected.confidence != null && (
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-dim">Confidence</span>
                      <span className="flex items-center gap-2"><span className="h-1.5 w-16 overflow-hidden rounded-full bg-hairline"><span className="block h-full bg-signal" style={{ width: `${Math.round(selected.confidence * 100)}%` }} /></span>{Math.round(selected.confidence * 100)}%</span>
                    </div>
                  )}
                </div>

                {selected.type === "event" && onOpenEvent && (
                  <button onClick={() => onOpenEvent(selected.id)} className="mt-3 w-fit rounded-md border border-signal/50 bg-signal/10 px-3 py-1.5 font-mono text-[11px] text-signal">
                    Open event details →
                  </button>
                )}
              </div>

              <div className="flex border-b border-hairline font-mono text-[10px] uppercase">
                {([
                  ["overview", "Overview"],
                  ["connections", `Connections (${allConnections.length})`],
                  ...(selected.type === "entity" ? [["events", `Events (${entityEvents.length})`] as const] : []),
                  ["evidence", `Evidence (${selected.type === "event" ? eventEvidence.length : entityEvidence.length})`],
                ] as [Tab, string][]).map(([id, tlabel]) => (
                  <button key={id} onClick={() => setTab(id)} className={`px-3 py-2 tracking-wide transition-colors ${tab === id ? "border-b-2 border-signal text-signal" : "text-dim hover:text-ink"}`}>
                    {tlabel}
                  </button>
                ))}
              </div>

              <div className="flex-1 overflow-y-auto p-4">
                {tab === "overview" && (
                  <p className="text-[12px] leading-relaxed text-dim">
                    {selected.description || selected.summary || selected.content_text || "No further description available."}
                  </p>
                )}

                {tab === "connections" && (
                  connectionsByType.length === 0 ? <p className="font-mono text-[11px] text-dim">No connections.</p> : (
                    <div className="flex flex-col gap-4">
                      {connectionsByType.map(([rel, items]) => (
                        <div key={rel}>
                          <p className="eyebrow mb-1.5">{rel.replace(/_/g, " ")}</p>
                          <div className="flex flex-col gap-1.5">
                            {items.map((c, i) => (
                              <button key={i} onClick={() => { setSelected(c.node); setTab("overview"); }} className="flex items-center gap-2 rounded-md border border-hairline bg-panel2/60 px-2.5 py-2 text-left transition-colors hover:border-signal/40">
                                <span className="h-2 w-2 flex-shrink-0 rounded-full" style={{ background: NODE_COLOR[c.node.type] }} />
                                <span className="min-w-0 flex-1 truncate text-[12px] text-ink">{c.node.label}</span>
                                {c.confidence != null && <span className="flex-shrink-0 font-mono text-[10px] text-dim">{c.confidence.toFixed(2)}</span>}
                              </button>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )
                )}

                {tab === "events" && (
                  entityEvents.length === 0 ? <p className="font-mono text-[11px] text-dim">Not part of any knowledge event.</p> : (
                    <div className="flex flex-col gap-1.5">
                      {entityEvents.map((e) => (
                        <button key={e.id} onClick={() => onOpenEvent?.(e.id)} className="rounded-md border border-hairline bg-panel2/60 px-2.5 py-2 text-left text-[12px] text-ink hover:border-signal/40">{e.label}</button>
                      ))}
                    </div>
                  )
                )}

                {tab === "evidence" && (
                  <div className="flex flex-col gap-1.5">
                    {(selected.type === "event" ? eventEvidence : entityEvidence).length === 0 ? (
                      <p className="font-mono text-[11px] text-dim">No evidence units.</p>
                    ) : (
                      (selected.type === "event" ? eventEvidence : entityEvidence).map((u) => (
                        <button key={u.id} onClick={() => { setSelected(u); setTab("overview"); }} className="flex items-center justify-between gap-2 rounded-md border border-hairline bg-panel2/60 px-2.5 py-2 text-left">
                          <span className="min-w-0 flex-1 truncate text-[12px] text-ink">{u.label}</span>
                          <span className="flex-shrink-0 font-mono text-[10px] uppercase text-dim">{u.modality}</span>
                        </button>
                      ))
                    )}
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>

      <p className="mt-2 font-mono text-[10px] text-dim/70">drag to pan · scroll to zoom · click a node to inspect · last updated {new Date(lastUpdated).toLocaleTimeString()}</p>
    </div>
  );
}

export function GraphLegend({ nodes, links }: { nodes: GraphNode[]; links: GraphLink[] }) {
  const rels = useMemo(() => relationshipStylesInUse(links.map((l) => l.relationship_type)), [links]);
  return (
    <div className="flex flex-col gap-4 text-[12px]">
      <div>
        <p className="eyebrow mb-2">Legend</p>
        <div className="flex flex-col gap-1.5">
          {[["entity", "Entity"], ["event", "Knowledge Event"], ["evidence", "Evidence Unit"]].map(([t, label]) => (
            <span key={t} className="flex items-center gap-2 text-dim">
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: NODE_COLOR[t] }} />{label}
            </span>
          ))}
        </div>
      </div>
      {rels.length > 0 && (
        <div>
          <p className="eyebrow mb-2">Relationships</p>
          <div className="flex flex-col gap-1.5">
            {rels.map((r) => (
              <span key={r.type} className="flex items-center gap-2 text-dim">
                <svg width="20" height="2"><line x1="0" y1="1" x2="20" y2="1" stroke="rgb(var(--c-signal))" strokeWidth="1.5" strokeDasharray={r.dash.join(",")} /></svg>
                {r.type.replace(/_/g, " ")}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function GraphStatus({ nodes, links }: { nodes: GraphNode[]; links: GraphLink[] }) {
  return (
    <div className="rounded-md border border-hairline bg-panel2/50 p-3">
      <div className="mb-1 flex items-center gap-1.5">
        <span className="h-1.5 w-1.5 rounded-full bg-signal animate-pulse2" />
        <p className="eyebrow">Graph status · live</p>
      </div>
      <div className="flex justify-between font-mono text-[11px] text-dim"><span>Nodes</span><span className="text-ink">{nodes.length}</span></div>
      <div className="flex justify-between font-mono text-[11px] text-dim"><span>Edges</span><span className="text-ink">{links.length}</span></div>
    </div>
  );
}
