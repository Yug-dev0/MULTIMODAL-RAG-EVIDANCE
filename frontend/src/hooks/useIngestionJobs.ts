import { useCallback, useEffect, useRef, useState } from "react";
import { apiClient, JobStatus } from "../api/client";

/**
 * Lives at the App root so polling survives tab switches -- a job submitted
 * from the Upload panel keeps updating even if the user navigates to Query
 * or Graph before it finishes.
 */
export function useIngestionJobs(onJobDone?: () => void) {
  const [jobs, setJobs] = useState<JobStatus[]>([]);
  const intervals = useRef<Record<string, ReturnType<typeof setInterval>>>({});
  const onJobDoneRef = useRef(onJobDone);
  onJobDoneRef.current = onJobDone;

  const upsertJob = useCallback((job: JobStatus) => {
    setJobs((prev) => {
      const idx = prev.findIndex((j) => j.job_id === job.job_id);
      if (idx === -1) return [job, ...prev];
      const next = [...prev];
      // Merge so a partial update (e.g. the lost-contact fallback) doesn't wipe
      // the filename we recorded at submit time.
      next[idx] = { ...prev[idx], ...job };
      return next;
    });
  }, []);

  const poll = useCallback(
    (jobId: string) => {
      if (intervals.current[jobId]) return;
      intervals.current[jobId] = setInterval(async () => {
        try {
          const s = await apiClient.getIngestStatus(jobId);
          upsertJob(s);
          if (s.status === "done" || s.status === "failed") {
            clearInterval(intervals.current[jobId]);
            delete intervals.current[jobId];
            if (s.status === "done") onJobDoneRef.current?.();
          }
        } catch {
          clearInterval(intervals.current[jobId]);
          delete intervals.current[jobId];
          upsertJob({ job_id: jobId, status: "failed", source_id: null, error: "Lost contact with the backend." });
        }
      }, 2000);
    },
    [upsertJob],
  );

  const submit = useCallback(
    async (file: File, sourceType: string, sessionId?: string) => {
      const { job_id, session_id } = await apiClient.ingest(file, sourceType, sessionId);
      // Carry the name/type through immediately -- a multi-file batch needs to
      // label each job before the first poll lands 2s later.
      upsertJob({
        job_id,
        status: "pending",
        source_id: null,
        error: null,
        filename: file.name,
        source_type: sourceType,
      });
      poll(job_id);
      return { job_id, session_id };
    },
    [poll, upsertJob],
  );

  useEffect(() => {
    const map = intervals.current;
    return () => {
      Object.values(map).forEach(clearInterval);
    };
  }, []);

  const runningCount = jobs.filter((j) => j.status === "pending" || j.status === "running").length;

  return { jobs, submit, runningCount };
}
