import { useEffect, useRef, useState } from "react";
import type { JobStatus } from "../../api/client";
import { AskAI } from "./AskAI";
import { useAuth } from "../../context/AuthContext";

function SunIcon() {
  return <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="12" cy="12" r="4" /><path strokeLinecap="round" d="M12 2v2M12 20v2M4 12H2M22 12h-2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" /></svg>;
}
function MoonIcon() {
  return <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6"><path strokeLinecap="round" strokeLinejoin="round" d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" /></svg>;
}
function BellIcon() {
  return <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6"><path strokeLinecap="round" strokeLinejoin="round" d="M6 10a6 6 0 1 1 12 0c0 4 1.5 5.5 1.5 5.5H4.5S6 14 6 10zM9.5 18.5a2.5 2.5 0 0 0 5 0" /></svg>;
}

interface Props {
  online: boolean | null;
  lastSyncMs: number | null;
  theme: "dark" | "light";
  onToggleTheme: () => void;
  jobs: JobStatus[];
}

function timeAgo(ms: number | null): string {
  if (ms === null) return "never";
  const s = Math.round((Date.now() - ms) / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}

export function Topbar({ online, lastSyncMs, theme, onToggleTheme, jobs }: Props) {
  const { user, logout } = useAuth();
  const [, forceTick] = useState(0);
  const [notifOpen, setNotifOpen] = useState(false);
  const [askOpen, setAskOpen] = useState(false);
  const seenNotif = useRef<Set<string>>(new Set());
  const notifRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const id = setInterval(() => forceTick((n) => n + 1), 15000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (notifRef.current && !notifRef.current.contains(e.target as Node)) setNotifOpen(false);
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  // Real notifications: derived directly from actual job state, not fabricated.
  const notifications = jobs
    .filter((j) => j.status === "done" || j.status === "failed")
    .slice(0, 8);
  const unseen = notifications.filter((j) => !seenNotif.current.has(j.job_id));

  function openNotif() {
    setNotifOpen((o) => !o);
    notifications.forEach((j) => seenNotif.current.add(j.job_id));
  }

  return (
    <div className="flex items-center justify-between gap-4 border-b border-hairline px-6 py-3">
      <div className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.14em] text-dim">
        <span className={`h-1.5 w-1.5 rounded-full ${online ? "bg-signal animate-pulse2" : online === false ? "bg-amber" : "bg-dim"}`} />
        <span>{online === null ? "connecting…" : online ? "system online" : "backend unreachable"}</span>
        <span className="mx-2 text-hairline">·</span>
        <span className="normal-case tracking-normal">Last sync: {timeAgo(lastSyncMs)}</span>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={() => setAskOpen(true)}
          className="flex items-center gap-1.5 rounded-md border border-signal/40 bg-gradient-to-r from-signal/15 to-panel2/40 px-3 py-1.5 font-mono text-[11px] uppercase tracking-wide text-signal transition-colors hover:from-signal/25"
        >
          <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth="1.8"><path strokeLinecap="round" strokeLinejoin="round" d="M12 3l1.8 4.6L18 9l-4.2 1.4L12 15l-1.8-4.6L6 9l4.2-1.4z" /><path strokeLinecap="round" d="M18 16l.9 2.1L21 19l-2.1.9L18 22l-.9-2.1L15 19l2.1-.9z" /></svg>
          Ask AI
        </button>

        <button onClick={onToggleTheme} className="rounded-md border border-hairline p-2 text-dim transition-colors hover:text-ink" aria-label="Toggle theme">
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </button>

        <div className="relative" ref={notifRef}>
          <button onClick={openNotif} className="relative rounded-md border border-hairline p-2 text-dim transition-colors hover:text-ink" aria-label="Notifications">
            <BellIcon />
            {unseen.length > 0 && (
              <span className="absolute -right-1 -top-1 flex h-4 w-4 items-center justify-center rounded-full bg-amber font-mono text-[9px] text-void">
                {unseen.length}
              </span>
            )}
          </button>
          {notifOpen && (
            <div className="absolute right-0 z-20 mt-2 w-72 border border-hairline bg-panel shadow-xl">
              <p className="eyebrow border-b border-hairline px-3 py-2">Notifications</p>
              {notifications.length === 0 ? (
                <p className="px-3 py-4 font-mono text-[11px] text-dim">No ingestion activity yet.</p>
              ) : (
                <div className="max-h-64 overflow-y-auto">
                  {notifications.map((j) => (
                    <div key={j.job_id} className="border-b border-hairline/60 px-3 py-2 last:border-0">
                      <p className="font-mono text-[11px] text-ink">{j.filename ?? j.job_id.slice(0, 8)}</p>
                      <p className={`font-mono text-[10px] ${j.status === "failed" ? "text-amber" : "text-signal"}`}>
                        {j.status === "failed" ? `Failed — ${j.error ?? "unknown error"}` : "Ingestion complete"}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {user ? (
          <div className="flex items-center gap-2 rounded-md border border-hairline bg-panel2/60 px-2.5 py-1">
            <span className="h-1.5 w-1.5 rounded-full bg-signal" />
            <span className="font-mono text-[10px] text-ink max-w-[130px] truncate" title={user.email || ""}>
              {user.email}
            </span>
            <span className="text-hairline">|</span>
            <button
              onClick={logout}
              className="font-mono text-[10px] uppercase text-dim hover:text-signal transition-colors"
            >
              Sign Out
            </button>
          </div>
        ) : (
          <div className="flex h-8 w-8 items-center justify-center rounded-full border border-hairline bg-panel2 text-dim">
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="12" cy="8" r="3.5" /><path strokeLinecap="round" d="M4.5 20a7.5 7.5 0 0 1 15 0" /></svg>
          </div>
        )}
      </div>

      {askOpen && <AskAI onClose={() => setAskOpen(false)} />}
    </div>
  );
}
