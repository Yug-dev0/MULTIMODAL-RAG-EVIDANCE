import { useRef, useState } from "react";
import type { View } from "../../App";

interface NavItem { id: View; label: string; icon: string; }
interface NavSection { label?: string; items: NavItem[]; }

const SECTIONS: NavSection[] = [
  {
    items: [
      { id: "home", label: "Home", icon: "home" },
      { id: "dashboard", label: "Dashboard", icon: "grid" },
      { id: "chat", label: "Chat", icon: "chat" },
    ],
  },
  {
    label: "Ingest",
    items: [
      { id: "upload", label: "Upload", icon: "upload" },
      { id: "sources", label: "Sources", icon: "layers" },
      { id: "processing", label: "Processing", icon: "activity" },
    ],
  },
  {
    label: "Explore",
    items: [
      { id: "graph", label: "Knowledge Graph", icon: "graph" },
      { id: "entities", label: "Entities", icon: "tag" },
      { id: "events", label: "Events", icon: "calendar" },
      { id: "evidence", label: "Evidence", icon: "file" },
    ],
  },
  {
    label: "Tools",
    items: [
      { id: "compare", label: "Compare (Text RAG)", icon: "split" },
      { id: "analytics", label: "Analytics", icon: "bar" },
      { id: "timeline", label: "Timeline", icon: "clock" },
    ],
  },
];

function Icon({ name, className = "h-4 w-4" }: { name: string; className?: string }) {
  const p = { stroke: "currentColor", strokeWidth: 1.6, fill: "none", strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  switch (name) {
    case "home": return <svg viewBox="0 0 24 24" className={className}><path {...p} d="M3 11l9-7 9 7M5 10v10h14V10" /></svg>;
    case "grid": return <svg viewBox="0 0 24 24" className={className}><path {...p} d="M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z" /></svg>;
    case "chat": return <svg viewBox="0 0 24 24" className={className}><path {...p} d="M4 5h16v11H8l-4 4V5z" /><path {...p} d="M8 9h8M8 12h5" /></svg>;
    case "upload": return <svg viewBox="0 0 24 24" className={className}><path {...p} d="M12 16V4M7 9l5-5 5 5M4 18v2h16v-2" /></svg>;
    case "layers": return <svg viewBox="0 0 24 24" className={className}><path {...p} d="M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5" /></svg>;
    case "activity": return <svg viewBox="0 0 24 24" className={className}><path {...p} d="M3 12h4l2 8 4-16 2 8h6" /></svg>;
    case "graph": return <svg viewBox="0 0 24 24" className={className}><circle {...p} cx="6" cy="6" r="2.5" /><circle {...p} cx="18" cy="8" r="2.5" /><circle {...p} cx="10" cy="18" r="2.5" /><path {...p} d="M8 7l7.5 1M9 16l6-6.5" /></svg>;
    case "tag": return <svg viewBox="0 0 24 24" className={className}><path {...p} d="M12 3h6a2 2 0 0 1 2 2v6l-9 9-8-8 9-9z" /><circle cx="16" cy="8" r="1" fill="currentColor" /></svg>;
    case "calendar": return <svg viewBox="0 0 24 24" className={className}><rect {...p} x="3" y="5" width="18" height="16" rx="1.5" /><path {...p} d="M3 10h18M8 3v4M16 3v4" /></svg>;
    case "file": return <svg viewBox="0 0 24 24" className={className}><path {...p} d="M6 2h9l5 5v15H6z" /><path {...p} d="M15 2v5h5M9 13h6M9 17h6" /></svg>;
    case "split": return <svg viewBox="0 0 24 24" className={className}><path {...p} d="M12 3v18M6 7l6-4 6 4M6 21l6-4 6 4" /></svg>;
    case "bar": return <svg viewBox="0 0 24 24" className={className}><path {...p} d="M4 20V10M12 20V4M20 20v-7" /></svg>;
    case "clock": return <svg viewBox="0 0 24 24" className={className}><circle {...p} cx="12" cy="12" r="9" /><path {...p} d="M12 7v5l3.5 3.5" /></svg>;
    case "chevron-left": return <svg viewBox="0 0 24 24" className={className}><path {...p} d="M15 5l-7 7 7 7" /></svg>;
    case "chevron-right": return <svg viewBox="0 0 24 24" className={className}><path {...p} d="M9 5l7 7-7 7" /></svg>;
    default: return null;
  }
}

interface Props {
  view: View;
  setView: (v: View) => void;
  runningJobs: number;
  systemUptimePct: number;
  footer?: React.ReactNode;
}

export function Sidebar({ view, setView, runningJobs, systemUptimePct, footer }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const [hovering, setHovering] = useState(false);
  const expanded = !collapsed || hovering;
  // Guards against the pointer still resting over the rail right as it
  // resizes on toggle -- without this, a stray hover recomputed mid-CSS
  // transition can immediately re-expand the sidebar the user just collapsed.
  const suppressHoverUntil = useRef(0);

  function toggleCollapsed(next: boolean) {
    setHovering(false);
    setCollapsed(next);
    suppressHoverUntil.current = Date.now() + 400;
  }

  return (
    <>
      <div className={`h-screen flex-shrink-0 transition-[width] duration-200 ${collapsed ? "w-16" : "w-60"}`} />
      <aside
        onMouseEnter={() => collapsed && Date.now() > suppressHoverUntil.current && setHovering(true)}
        onMouseLeave={() => setHovering(false)}
        className={`fixed left-0 top-0 z-30 flex h-screen flex-col overflow-hidden border-r border-hairline bg-panel px-3 py-4 transition-[width] duration-200 ${
          expanded ? "w-60" : "w-16"
        } ${hovering && collapsed ? "shadow-2xl" : ""}`}
      >
        <div className={`mb-6 flex items-center gap-2 px-2 ${expanded ? "" : "justify-center"}`}>
          <svg viewBox="0 0 24 24" className="h-6 w-6 flex-shrink-0 text-signal"><circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" fill="none" /><circle cx="12" cy="12" r="3" fill="currentColor" /></svg>
          {expanded && <span className="whitespace-nowrap font-mono text-sm font-semibold tracking-wide text-ink">MME Console</span>}
          {expanded && (
            <button
              onClick={() => toggleCollapsed(true)}
              aria-label="Collapse sidebar"
              className="ml-auto flex-shrink-0 rounded-md p-1 text-dim transition-colors hover:bg-panel2 hover:text-ink"
            >
              <Icon name="chevron-left" className="h-4 w-4" />
            </button>
          )}
        </div>

        {!expanded && (
          <button
            onClick={() => toggleCollapsed(false)}
            aria-label="Expand sidebar"
            className="mb-6 flex justify-center rounded-md p-1 text-dim transition-colors hover:bg-panel2 hover:text-ink"
          >
            <Icon name="chevron-right" className="h-4 w-4" />
          </button>
        )}

        <nav className="flex-1 overflow-y-auto overflow-x-hidden">
          {SECTIONS.map((section, i) => (
            <div key={i} className="mb-4">
              {section.label && expanded && <p className="eyebrow mb-1 whitespace-nowrap px-2">{section.label}</p>}
              <div className="flex flex-col gap-0.5">
                {section.items.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => setView(item.id)}
                    title={expanded ? undefined : item.label}
                    className={`relative flex items-center gap-2.5 rounded-md px-2.5 py-2 text-left font-sans text-[13px] transition-colors ${
                      expanded ? "" : "justify-center"
                    } ${view === item.id ? "bg-signal/12 text-signal" : "text-dim hover:bg-panel2 hover:text-ink"}`}
                  >
                    <span className="flex-shrink-0"><Icon name={item.icon} /></span>
                    {expanded && <span className="whitespace-nowrap">{item.label}</span>}
                    {item.id === "processing" && runningJobs > 0 && (
                      <span
                        className={`rounded-full bg-amber/20 font-mono text-amber ${
                          expanded ? "ml-auto px-1.5 py-0.5 text-[10px]" : "absolute right-1 top-1 h-2 w-2 p-0"
                        }`}
                      >
                        {expanded ? runningJobs : ""}
                      </span>
                    )}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </nav>

        <a
          href="/showcase.html"
          target="_blank"
          rel="noopener noreferrer"
          title={expanded ? undefined : "3D Showcase"}
          className={`mb-3 flex items-center gap-2.5 rounded-md px-2.5 py-2 font-sans text-[13px] text-dim transition-colors hover:bg-panel2 hover:text-ink ${
            expanded ? "" : "justify-center"
          }`}
        >
          <svg viewBox="0 0 24 24" className="h-4 w-4 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth="1.6"><path strokeLinecap="round" strokeLinejoin="round" d="M14 3h7v7M21 3l-9 9M5 5h6v2H7v10h10v-4h2v6H5z" /></svg>
          {expanded && <span className="whitespace-nowrap">3D Showcase ↗</span>}
        </a>

        {expanded &&
          (footer ?? (
            <div className="rounded-md border border-hairline bg-panel2 p-3">
              <div className="mb-1 flex items-center gap-1.5">
                <span className="h-1.5 w-1.5 rounded-full bg-signal animate-pulse2" />
                <p className="eyebrow">System status</p>
              </div>
              <p className="font-sans text-[12px] text-ink">All systems operational</p>
              <p className="mt-1 font-mono text-[11px] text-dim">Uptime {systemUptimePct.toFixed(2)}% <span className="text-dim/60">(this session)</span></p>
              <div className="mt-2.5 pt-2 border-t border-hairline flex items-center justify-between font-mono text-[10px]">
                <span className="text-dim">Author & Maker</span>
                <span className="font-semibold text-signal">Yug</span>
              </div>
            </div>
          ))}
      </aside>
    </>
  );
}
