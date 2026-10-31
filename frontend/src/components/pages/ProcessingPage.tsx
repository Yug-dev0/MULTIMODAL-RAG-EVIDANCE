import { useEffect, useState } from "react";
import { apiClient, JobStatus } from "../../api/client";

const STATUS_COLOR: Record<string, string> = {
  done: "text-signal", failed: "text-amber", running: "text-signal", pending: "text-dim",
};

export function ProcessingPage() {
  const [jobs, setJobs] = useState<JobStatus[] | null>(null);

  useEffect(() => {
    function load() {
      apiClient.getJobs().then(setJobs).catch(() => setJobs([]));
    }
    load();
    const id = setInterval(load, 3000);
    return () => clearInterval(id);
  }, []);

  return (
    <div>
      <p className="eyebrow mb-1">Ingest</p>
      <h2 className="mb-4 font-sans text-xl font-semibold text-ink">Processing</h2>
      {jobs === null ? (
        <p className="font-mono text-xs text-dim">Loading…</p>
      ) : jobs.length === 0 ? (
        <p className="font-mono text-xs text-dim">No ingestion jobs submitted yet.</p>
      ) : (
        <div className="flex flex-col gap-2">
          {jobs.map((j) => (
            <div key={j.job_id} className="flex items-center gap-3 border border-hairline bg-panel px-4 py-3">
              <span className={`h-2 w-2 rounded-full ${j.status === "running" || j.status === "pending" ? "animate-pulse2 bg-signal" : j.status === "failed" ? "bg-amber" : "bg-signal"}`} />
              <div className="min-w-0 flex-1">
                <p className="truncate font-sans text-sm text-ink">{j.filename ?? j.job_id}</p>
                {j.error && <p className="truncate font-mono text-[11px] text-amber">{j.error}</p>}
              </div>
              <span className="font-mono text-[11px] text-dim">{j.source_type}</span>
              <span className={`font-mono text-[11px] uppercase ${STATUS_COLOR[j.status] ?? "text-dim"}`}>{j.status}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
