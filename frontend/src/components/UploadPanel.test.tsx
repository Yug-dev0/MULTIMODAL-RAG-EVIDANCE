import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { UploadPanel } from "./UploadPanel";
import { JobStatus } from "../api/client";

describe("UploadPanel", () => {
  it("submits a file via the injected submit callback and shows the resulting status", async () => {
    const submit = vi.fn().mockResolvedValue({ job_id: "job1", session_id: "sess1" });
    const jobs: JobStatus[] = [{ job_id: "job1", status: "done", source_id: "s1", error: null }];

    render(<UploadPanel jobs={jobs} submit={submit} />);
    const file = new File(["data"], "video.mp4", { type: "video/mp4" });
    fireEvent.change(screen.getByTestId("file-input"), { target: { files: [file] } });
    fireEvent.submit(screen.getByTestId("file-input").closest("form")!);

    await waitFor(() => expect(submit).toHaveBeenCalledWith(file, "video"));
    await waitFor(() => expect(screen.getByTestId("ingest-status")).toHaveTextContent("done"));
  });

  it("shows an error if submit rejects", async () => {
    const submit = vi.fn().mockRejectedValue(new Error("backend unreachable"));
    render(<UploadPanel jobs={[]} submit={submit} />);
    const file = new File(["data"], "video.mp4", { type: "video/mp4" });
    fireEvent.change(screen.getByTestId("file-input"), { target: { files: [file] } });
    fireEvent.submit(screen.getByTestId("file-input").closest("form")!);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("backend unreachable"));
  });

  it("auto-detects the source type from the file extension so a mismatched submit can't happen by default", async () => {
    const submit = vi.fn().mockResolvedValue({ job_id: "job1", session_id: "sess1" });
    render(<UploadPanel jobs={[]} submit={submit} />);
    const file = new File(["%PDF-1.4"], "resume.pdf", { type: "application/pdf" });
    fireEvent.change(screen.getByTestId("file-input"), { target: { files: [file] } });
    fireEvent.submit(screen.getByTestId("file-input").closest("form")!);
    await waitFor(() => expect(submit).toHaveBeenCalledWith(file, "pdf"));
  });

  it("shows the backend's error message when a job fails, not just a bare status", async () => {
    const submit = vi.fn().mockResolvedValue({ job_id: "job1", session_id: "sess1" });
    const jobs: JobStatus[] = [{
      job_id: "job1", status: "failed", source_id: "s1",
      error: "You selected 'video' but the file has a '.pdf' extension.",
    }];
    render(<UploadPanel jobs={jobs} submit={submit} />);
    const file = new File(["data"], "resume.pdf", { type: "application/pdf" });
    fireEvent.change(screen.getByTestId("file-input"), { target: { files: [file] } });
    fireEvent.submit(screen.getByTestId("file-input").closest("form")!);
    await waitFor(() => expect(screen.getByText(/you selected 'video'/i)).toBeInTheDocument());
  });

  it("stages every selected file and submits one job per file", async () => {
    const submit = vi.fn().mockImplementation((f: File) =>
      Promise.resolve({ job_id: `job-${f.name}`, session_id: "sess1" }));
    render(<UploadPanel jobs={[]} submit={submit} />);
    const pdf = new File(["%PDF-1.4"], "resume.pdf", { type: "application/pdf" });
    const png = new File(["\x89PNG"], "chart.png", { type: "image/png" });

    fireEvent.change(screen.getByTestId("file-input"), { target: { files: [pdf, png] } });
    expect(screen.getAllByTestId("queued-file")).toHaveLength(2);

    fireEvent.submit(screen.getByTestId("file-input").closest("form")!);
    await waitFor(() => expect(submit).toHaveBeenCalledTimes(2));
    // Each file keeps its own detected type -- a mixed batch must not collapse
    // onto whichever type happened to be selected.
    expect(submit).toHaveBeenCalledWith(pdf, "pdf");
    expect(submit).toHaveBeenCalledWith(png, "image", "sess1");
  });

  it("puts the whole batch in one chat session instead of one session per file", async () => {
    const submit = vi.fn().mockImplementation((f: File) =>
      Promise.resolve({ job_id: `job-${f.name}`, session_id: `sess-${f.name}` }));
    render(<UploadPanel jobs={[]} submit={submit} />);
    const files = ["a.png", "b.png", "c.png"].map((n) => new File(["x"], n, { type: "image/png" }));
    fireEvent.change(screen.getByTestId("file-input"), { target: { files } });
    fireEvent.submit(screen.getByTestId("file-input").closest("form")!);

    await waitFor(() => expect(submit).toHaveBeenCalledTimes(3));
    // First upload mints the session; every later one must reuse that id,
    // otherwise the files land in separate sessions and can't be queried together.
    expect(submit.mock.calls[0][2]).toBeUndefined();
    expect(submit.mock.calls[1][2]).toBe("sess-a.png");
    expect(submit.mock.calls[2][2]).toBe("sess-a.png");
  });

  it("accumulates across separate picks and de-duplicates the same file", () => {
    render(<UploadPanel jobs={[]} submit={vi.fn()} />);
    const a = new File(["a"], "a.png", { type: "image/png" });
    const b = new File(["b"], "b.png", { type: "image/png" });
    const input = screen.getByTestId("file-input");

    fireEvent.change(input, { target: { files: [a] } });
    fireEvent.change(input, { target: { files: [b] } });
    expect(screen.getAllByTestId("queued-file")).toHaveLength(2);

    fireEvent.change(input, { target: { files: [a] } });
    expect(screen.getAllByTestId("queued-file")).toHaveLength(2);
  });

  it("removes a single staged file without dropping the rest", () => {
    render(<UploadPanel jobs={[]} submit={vi.fn()} />);
    const a = new File(["a"], "a.png", { type: "image/png" });
    const b = new File(["b"], "b.png", { type: "image/png" });
    fireEvent.change(screen.getByTestId("file-input"), { target: { files: [a, b] } });

    fireEvent.click(screen.getByLabelText("Remove a.png"));
    expect(screen.getAllByTestId("queued-file")).toHaveLength(1);
    expect(screen.getByText("b.png")).toBeInTheDocument();
  });

  it("keeps failed files staged for retry and reports which ones failed", async () => {
    const submit = vi.fn().mockImplementation((f: File) =>
      f.name === "bad.pdf"
        ? Promise.reject(new Error("boom"))
        : Promise.resolve({ job_id: "job-ok", session_id: "sess1" }),
    );
    render(<UploadPanel jobs={[]} submit={submit} />);
    const ok = new File(["ok"], "good.pdf", { type: "application/pdf" });
    const bad = new File(["bad"], "bad.pdf", { type: "application/pdf" });
    fireEvent.change(screen.getByTestId("file-input"), { target: { files: [ok, bad] } });
    fireEvent.submit(screen.getByTestId("file-input").closest("form")!);

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("bad.pdf — boom"));
    const staged = screen.getAllByTestId("queued-file");
    expect(staged).toHaveLength(1);
    expect(staged[0]).toHaveTextContent("bad.pdf");
  });

  it("shows per-job progress for a batch and lets you focus one", async () => {
    const submit = vi.fn().mockImplementation((f: File) =>
      Promise.resolve({ job_id: `job-${f.name}`, session_id: "sess1" }));
    const jobs: JobStatus[] = [
      { job_id: "job-a.png", status: "done", source_id: "s1", error: null, filename: "a.png" },
      { job_id: "job-b.png", status: "running", source_id: null, error: null, filename: "b.png", stage: "extract" },
    ];
    render(<UploadPanel jobs={jobs} submit={submit} />);
    const a = new File(["a"], "a.png", { type: "image/png" });
    const b = new File(["b"], "b.png", { type: "image/png" });
    fireEvent.change(screen.getByTestId("file-input"), { target: { files: [a, b] } });
    fireEvent.submit(screen.getByTestId("file-input").closest("form")!);

    await waitFor(() => expect(screen.getByTestId("batch-progress")).toHaveTextContent("1/2 done"));
    // Focus defaults to the first job in the batch.
    expect(screen.getByTestId("ingest-status")).toHaveTextContent("done");
    fireEvent.click(screen.getByRole("button", { name: /b\.png/ }));
    expect(screen.getByTestId("ingest-status")).toHaveTextContent("running");
  });
});
