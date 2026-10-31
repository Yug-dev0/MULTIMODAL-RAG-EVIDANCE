import { renderHook, act, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useIngestionJobs } from "./useIngestionJobs";
import { apiClient } from "../api/client";

vi.mock("../api/client", () => ({
  apiClient: { ingest: vi.fn(), getIngestStatus: vi.fn() },
}));

describe("useIngestionJobs", () => {
  beforeEach(() => vi.useFakeTimers({ shouldAdvanceTime: true }));
  afterEach(() => vi.useRealTimers());

  it("keeps polling after being submitted, independent of any component mount/unmount", async () => {
    (apiClient.ingest as any).mockResolvedValue({ job_id: "job1" });
    (apiClient.getIngestStatus as any)
      .mockResolvedValueOnce({ job_id: "job1", status: "running", source_id: "s1", error: null })
      .mockResolvedValueOnce({ job_id: "job1", status: "done", source_id: "s1", error: null });

    const onJobDone = vi.fn();
    const { result } = renderHook(() => useIngestionJobs(onJobDone));

    await act(async () => {
      await result.current.submit(new File(["x"], "a.mp4"), "video");
    });
    expect(result.current.jobs[0].status).toBe("pending");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(result.current.jobs[0].status).toBe("running");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    await waitFor(() => expect(result.current.jobs[0].status).toBe("done"));
    expect(onJobDone).toHaveBeenCalledTimes(1);
  });
});
