import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { ChatPage, ChatOpenSignal } from "./components/pages/ChatPage";
import { CompareView } from "./components/CompareView";
import type { GraphNode, GraphLink } from "./components/GraphView";
import { UploadPanel } from "./components/UploadPanel";
import { Sidebar } from "./components/layout/Sidebar";
import { Topbar } from "./components/layout/Topbar";
import { Dashboard } from "./components/dashboard/Dashboard";
import { HomePage } from "./components/dashboard/HomePage";
import { SourcesPage } from "./components/pages/SourcesPage";
import { ProcessingPage } from "./components/pages/ProcessingPage";
import { EntitiesPage } from "./components/pages/EntitiesPage";
import { EventsPage } from "./components/pages/EventsPage";
import { EventDetailPage } from "./components/pages/EventDetailPage";
import { EvidencePage } from "./components/pages/EvidencePage";
import { AnalyticsPage } from "./components/pages/AnalyticsPage";
import { TimelinePage } from "./components/pages/TimelinePage";
import { useIngestionJobs } from "./hooks/useIngestionJobs";
import { useTheme } from "./hooks/useTheme";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { Login } from "./components/Login";

// The canvas graph renderer is deferred to a separate chunk so it never
// costs anything on the initial page load (Query/Home, the common case).
const GraphView = lazy(() => import("./components/GraphView").then((m) => ({ default: m.GraphView })));
const GraphLegend = lazy(() => import("./components/GraphView").then((m) => ({ default: m.GraphLegend })));
const GraphStatus = lazy(() => import("./components/GraphView").then((m) => ({ default: m.GraphStatus })));

export type View =
  | "home" | "dashboard" | "chat" | "compare" | "graph" | "upload" | "event-detail"
  | "sources" | "processing" | "entities" | "events" | "evidence" | "analytics" | "timeline";

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const HEALTH_POLL_MS = 15000;

function MainApp() {
  const { user, loading, token } = useAuth();
  const [view, setView] = useState<View>("home");
  const [graphMounted, setGraphMounted] = useState(false);
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; links: GraphLink[] }>({ nodes: [], links: [] });
  const [online, setOnline] = useState<boolean | null>(null);
  const [lastSyncMs, setLastSyncMs] = useState<number | null>(null);
  const [chatOpenSignal, setChatOpenSignal] = useState<ChatOpenSignal | null>(null);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const healthChecks = useRef({ ok: 0, total: 0 });
  const [uptimePct, setUptimePct] = useState(100);
  const { theme, toggle: toggleTheme } = useTheme();

  const refreshGraph = useCallback(() => {
    if (!token) return;
    fetch(`${BASE_URL}/graph`, {
      headers: { "Authorization": `Bearer ${token}` }
    })
      .then((r) => {
        if (!r.ok) return null;
        return r.json();
      })
      .then((data) => {
        if (data && Array.isArray(data.nodes) && Array.isArray(data.links)) {
          setGraphData(data);
        }
      })
      .catch(() => {});
  }, [token]);

  const { jobs, submit, runningCount } = useIngestionJobs(refreshGraph);

  const checkHealth = useCallback(async () => {
    healthChecks.current.total += 1;
    try {
      const res = await fetch(`${BASE_URL}/health`);
      if (res.ok) healthChecks.current.ok += 1;
      setOnline(res.ok);
    } catch {
      setOnline(false);
    }
    setLastSyncMs(Date.now());
    setUptimePct((healthChecks.current.ok / healthChecks.current.total) * 100);
  }, []);

  useEffect(() => {
    checkHealth();
    const id = setInterval(checkHealth, HEALTH_POLL_MS);
    return () => clearInterval(id);
  }, [checkHealth]);

  useEffect(() => {
    if (user && token) {
      refreshGraph();
    }
  }, [user, token, refreshGraph]);

  const goToQuery = useCallback((q: string) => {
    setChatOpenSignal({ query: q, id: Date.now() });
    setView("chat");
  }, []);

  const openGraph = useCallback(() => {
    setView("graph");
    setGraphMounted(true);
  }, []);

  const openEvent = useCallback((id: string) => {
    setSelectedEventId(id);
    setView("event-detail");
  }, []);

  const safeNodes = Array.isArray(graphData?.nodes) ? graphData.nodes : [];
  const safeLinks = Array.isArray(graphData?.links) ? graphData.links : [];

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-void">
        <div className="text-center flex flex-col gap-2">
          <p className="font-mono text-xs text-dim animate-pulse2">INITIALIZING SYSTEM CONTEXT…</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-void flex items-center justify-center p-6">
        <Login />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar
        view={view}
        setView={(v) => (v === "graph" ? openGraph() : setView(v))}
        runningJobs={runningCount}
        systemUptimePct={uptimePct}
        footer={
          view === "graph" && graphMounted ? (
            <Suspense fallback={null}>
              <div className="flex flex-col gap-4">
                <GraphLegend nodes={graphData.nodes} links={graphData.links} />
                <GraphStatus nodes={graphData.nodes} links={graphData.links} />
              </div>
            </Suspense>
          ) : undefined
        }
      />

      <div className="flex min-h-screen flex-1 flex-col">
        <Topbar online={online} lastSyncMs={lastSyncMs} theme={theme} onToggleTheme={toggleTheme} jobs={jobs} />

        <div className="flex-1 overflow-y-auto px-8 py-7">
          <div className="mx-auto max-w-[1400px]">
            {view === "home" && (
              <HomePage onSearch={goToQuery} onOpenEvent={openEvent} onOpenGraph={openGraph} graphData={graphData} />
            )}
            {view === "dashboard" && (
              <Dashboard onSearch={goToQuery} onOpenEvent={openEvent} onOpenGraph={openGraph} graphData={graphData} />
            )}

            <div className={view === "chat" ? "" : "hidden"}><ChatPage jobs={jobs} submit={submit} openSignal={chatOpenSignal} /></div>
            <div className={view === "compare" ? "" : "hidden"}><CompareView /></div>
            <div className={view === "graph" ? "" : "hidden"}>
              {graphMounted && (
                <Suspense fallback={<p className="font-mono text-xs text-dim">Loading graph…</p>}>
                  <GraphView nodes={safeNodes} links={safeLinks} active={view === "graph"} onOpenEvent={openEvent} />
                </Suspense>
              )}
            </div>
            <div className={view === "upload" ? "" : "hidden"}><UploadPanel jobs={jobs} submit={submit} onStartChat={(sessionId) => { setChatOpenSignal({ query: "", id: Date.now(), sessionId }); setView("chat"); }} /></div>

            {view === "sources" && <SourcesPage />}
            {view === "processing" && <ProcessingPage />}
            {view === "entities" && <EntitiesPage />}
            {view === "events" && <EventsPage onOpen={openEvent} />}
            {view === "event-detail" && selectedEventId && (
              <EventDetailPage eventId={selectedEventId} onBack={() => setView("events")} onOpenGraph={openGraph} />
            )}
            {view === "evidence" && <EvidencePage />}
            {view === "analytics" && <AnalyticsPage />}
            {view === "timeline" && <TimelinePage onOpen={openEvent} />}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <MainApp />
    </AuthProvider>
  );
}
