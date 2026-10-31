import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { ChatPage } from "./ChatPage";
import { apiClient } from "../../api/client";

vi.mock("../../api/client", () => ({
  apiClient: {
    listChatSessions: vi.fn(),
    createChatSession: vi.fn(),
    getChatMessages: vi.fn(),
    sendChatQuery: vi.fn(),
  },
}));

describe("ChatPage", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows the empty state with no sessions yet", async () => {
    (apiClient.listChatSessions as any).mockResolvedValue([]);
    render(<ChatPage jobs={[]} submit={vi.fn()} />);
    await waitFor(() => expect(screen.getByText(/your topics will appear here/i)).toBeInTheDocument());
    expect(screen.getByText(/attach a file to start one/i)).toBeInTheDocument();
  });

  it("sending the first message creates a new session and shows the answer with citations", async () => {
    (apiClient.listChatSessions as any).mockResolvedValue([]);
    (apiClient.createChatSession as any).mockResolvedValue({
      id: "s1", title: "New chat", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z",
      message_count: 0, source_count: 0,
    });
    (apiClient.sendChatQuery as any).mockResolvedValue({
      answer: "Redis was proposed to reduce database load.",
      citations: [{ evidence_id: "e1", modality: "speech", source: "v.mp4", timestamp_start: 1, timestamp_end: 2, page_number: null, thumbnail: null, confidence: 0.9 }],
      events_used: ["ev1"],
      session_title: "Redis Caching",
    });

    render(<ChatPage jobs={[]} submit={vi.fn()} />);
    await waitFor(() => expect(apiClient.listChatSessions).toHaveBeenCalled());

    fireEvent.change(screen.getByPlaceholderText(/ask about this topic/i), { target: { value: "What reduces load?" } });
    fireEvent.click(screen.getByText("Send"));

    await waitFor(() => expect(apiClient.createChatSession).toHaveBeenCalled());
    await waitFor(() => expect(apiClient.sendChatQuery).toHaveBeenCalledWith("s1", "What reduces load?"));
    await waitFor(() => expect(screen.getByText(/redis was proposed to reduce database load/i)).toBeInTheDocument());
  });

  it("clicking a citation opens its exact source (video seeked to the right timestamp) on the right, and closes cleanly", async () => {
    (apiClient.listChatSessions as any).mockResolvedValue([]);
    (apiClient.createChatSession as any).mockResolvedValue({
      id: "s1", title: "New chat", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z",
      message_count: 0, source_count: 0,
    });
    (apiClient.sendChatQuery as any).mockResolvedValue({
      answer: "Redis was proposed to reduce database load.",
      citations: [{
        evidence_id: "e1", modality: "speech", source: "./data/media/job1_meeting.mp4#t=430.0,440.0",
        content: "We propose Redis to reduce database load", timestamp_start: 430.0, timestamp_end: 440.0,
        page_number: null, thumbnail: null, confidence: 0.9,
      }],
      events_used: ["ev1"],
      session_title: "Redis Caching",
    });

    render(<ChatPage jobs={[]} submit={vi.fn()} />);
    await waitFor(() => expect(apiClient.listChatSessions).toHaveBeenCalled());

    fireEvent.change(screen.getByPlaceholderText(/ask about this topic/i), { target: { value: "What reduces load?" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() => expect(screen.getByText("7:10–7:20")).toBeInTheDocument());

    // No preview open yet.
    expect(screen.queryByText("Evidence source")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /we propose redis to reduce database load/i }));

    expect(screen.getByText("Evidence source")).toBeInTheDocument();
    const video = screen.getByTestId("video-player") as HTMLVideoElement;
    expect(video.src).toContain("/media/job1_meeting.mp4");
    fireEvent.loadedMetadata(video);
    expect(video.currentTime).toBe(430);

    fireEvent.click(screen.getByLabelText("Close evidence preview"));
    expect(screen.queryByText("Evidence source")).not.toBeInTheDocument();
  });

  it("attaching a file then immediately sending a message reuses the same session, not two", async () => {
    // Regression test: an attach and a send fired back-to-back (before the
    // attach's session creation has resolved) used to each independently
    // create a session -- the upload landed in session A while the
    // question was asked against a brand-new, sourceless session B.
    (apiClient.listChatSessions as any).mockResolvedValue([]);
    (apiClient.createChatSession as any).mockResolvedValue({
      id: "s1", title: "New chat", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z",
      message_count: 0, source_count: 0,
    });
    const submit = vi.fn().mockResolvedValue({ job_id: "j1", session_id: "s1" });
    (apiClient.sendChatQuery as any).mockResolvedValue({
      answer: "answer", citations: [], events_used: [], session_title: "t",
    });

    render(<ChatPage jobs={[]} submit={submit} />);
    await waitFor(() => expect(apiClient.listChatSessions).toHaveBeenCalled());

    const file = new File(["data"], "video.mp4", { type: "video/mp4" });
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file] } });

    fireEvent.change(screen.getByPlaceholderText(/ask about this topic/i), { target: { value: "What is this?" } });
    fireEvent.click(screen.getByText("Send"));

    await waitFor(() => expect(submit).toHaveBeenCalledWith(file, "video", "s1"));
    await waitFor(() => expect(apiClient.sendChatQuery).toHaveBeenCalledWith("s1", "What is this?"));
    expect(apiClient.createChatSession).toHaveBeenCalledTimes(1);
  });

  it("attaching a second file keeps the first attachment visible and adds the second alongside it", async () => {
    (apiClient.listChatSessions as any).mockResolvedValue([]);
    (apiClient.createChatSession as any).mockResolvedValue({
      id: "s1", title: "New chat", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z",
      message_count: 0, source_count: 0,
    });
    const submit = vi.fn()
      .mockResolvedValueOnce({ job_id: "j1", session_id: "s1" })
      .mockResolvedValueOnce({ job_id: "j2", session_id: "s1" });

    const jobs = [
      { job_id: "j1", status: "running" as const, source_id: null, error: null, filename: "meeting.mp4", stage: "extract" as const },
      { job_id: "j2", status: "pending" as const, source_id: null, error: null, filename: "notes.pdf" },
    ];

    const { rerender } = render(<ChatPage jobs={[]} submit={submit} />);
    await waitFor(() => expect(apiClient.listChatSessions).toHaveBeenCalled());

    const file1 = new File(["data"], "meeting.mp4", { type: "video/mp4" });
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [file1] } });
    await waitFor(() => expect(submit).toHaveBeenCalledTimes(1));
    rerender(<ChatPage jobs={[jobs[0]]} submit={submit} />);
    await waitFor(() => expect(screen.getByText("meeting.mp4")).toBeInTheDocument());

    // The "Add another file" affordance appears once something is attached,
    // and a second attach must not replace the first -- both stay visible.
    fireEvent.click(screen.getByText("Add another file"));
    const file2 = new File(["data"], "notes.pdf", { type: "application/pdf" });
    fireEvent.change(fileInput, { target: { files: [file2] } });
    await waitFor(() => expect(submit).toHaveBeenCalledTimes(2));
    rerender(<ChatPage jobs={jobs} submit={submit} />);

    expect(screen.getByText("meeting.mp4")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("notes.pdf")).toBeInTheDocument());
  });

  it("switching to a past session loads its message history", async () => {
    (apiClient.listChatSessions as any).mockResolvedValue([
      { id: "s1", title: "Redis Caching", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z", message_count: 2, source_count: 1 },
    ]);
    (apiClient.getChatMessages as any).mockResolvedValue([
      { id: "m1", role: "user", content: "What reduces load?", citations: [], created_at: "2026-01-01T00:00:00Z" },
      { id: "m2", role: "assistant", content: "Redis was proposed.", citations: [], created_at: "2026-01-01T00:00:01Z" },
    ]);

    render(<ChatPage jobs={[]} submit={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("Redis Caching")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Redis Caching"));

    await waitFor(() => expect(apiClient.getChatMessages).toHaveBeenCalledWith("s1"));
    await waitFor(() => expect(screen.getByText("Redis was proposed.")).toBeInTheDocument());
  });
});
