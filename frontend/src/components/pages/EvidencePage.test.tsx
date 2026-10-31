import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { EvidencePage } from "./EvidencePage";
import { apiClient } from "../../api/client";

vi.mock("../../api/client", () => ({
  apiClient: { getEvidence: vi.fn(), getEvidenceByQuery: vi.fn() },
}));

beforeEach(() => {
  (apiClient.getEvidenceByQuery as any).mockResolvedValue([]);
});

const VIDEO_ROW = {
  id: "v1", content_text: "We should cache with Redis here.", modality: "speech",
  source_type: "video", raw_locator: "./data/media/job1_meeting.mp4#t=430.0,440.0",
  timestamp_start: 430.0, timestamp_end: 440.0, page_number: null, confidence: 0.9,
  relevance: null,
  extracted_at: "2026-01-01T00:00:00Z", session_id: "s1", session_title: "Redis Architecture",
};

const PDF_ROW = {
  id: "p1", content_text: "Page discusses the caching layer diagram.", modality: "document",
  source_type: "pdf", raw_locator: "./data/media/job2_notes.pdf#page=3",
  timestamp_start: null, timestamp_end: null, page_number: 3, confidence: 0.98,
  relevance: null,
  extracted_at: "2026-01-01T00:00:00Z", session_id: null, session_title: null,
};

describe("EvidencePage", () => {
  it("shows the empty state when nothing has been ingested", async () => {
    (apiClient.getEvidence as any).mockResolvedValue([]);
    render(<EvidencePage />);
    await waitFor(() => expect(screen.getByText(/no evidence ingested yet/i)).toBeInTheDocument());
  });

  it("shows a timestamp badge for video evidence and a page badge for PDF evidence", async () => {
    (apiClient.getEvidence as any).mockResolvedValue([VIDEO_ROW, PDF_ROW]);
    render(<EvidencePage />);
    await waitFor(() => expect(screen.getByText("7:10–7:20")).toBeInTheDocument());
    expect(screen.getByText("Page 3")).toBeInTheDocument();
  });

  it("groups evidence by chat session, with ungrouped evidence in its own section", async () => {
    (apiClient.getEvidence as any).mockResolvedValue([VIDEO_ROW, PDF_ROW]);
    render(<EvidencePage />);
    // "All evidence from" matters: the session title is generated from the
    // first question asked, so a bare title read as "these answer that".
    await waitFor(() => expect(screen.getByText(/All evidence from: Redis Architecture/)).toBeInTheDocument());
    expect(screen.getByText("Ungrouped")).toBeInTheDocument();
  });

  it("groups evidence under the question it answered, not scattered across the corpus", async () => {
    (apiClient.getEvidence as any).mockResolvedValue([VIDEO_ROW, PDF_ROW]);
    (apiClient.getEvidenceByQuery as any).mockResolvedValue([
      {
        query: "what are the specializations in AI/ML?",
        asked_at: "2026-01-02T00:00:00Z", session_id: "s1", session_title: "Redis Architecture",
        evidence: [PDF_ROW],
      },
    ]);

    render(<EvidencePage />);

    await waitFor(() =>
      expect(screen.getByText(/what are the specializations in AI\/ML\?/)).toBeInTheDocument(),
    );
    // The evidence that answered it sits under the question...
    const group = screen.getByTestId("query-group-0");
    expect(group).toHaveTextContent("Page discusses the caching layer diagram.");
    // ...and evidence that answered nothing does not ride along under it.
    expect(group).not.toHaveTextContent("We should cache with Redis here.");
  });

  it("keeps uncited evidence browsable in a section that doesn't claim it answered anything", async () => {
    (apiClient.getEvidence as any).mockResolvedValue([VIDEO_ROW, PDF_ROW]);
    (apiClient.getEvidenceByQuery as any).mockResolvedValue([
      {
        query: "what are the specializations in AI/ML?",
        asked_at: "2026-01-02T00:00:00Z", session_id: "s1", session_title: "Redis Architecture",
        evidence: [PDF_ROW],
      },
    ]);

    render(<EvidencePage />);

    await waitFor(() => expect(screen.getByTestId("uncited-group")).toBeInTheDocument());
    const uncited = screen.getByTestId("uncited-group");
    expect(uncited).toHaveTextContent("We should cache with Redis here.");
    expect(uncited).not.toHaveTextContent("Page discusses the caching layer diagram.");
  });

  it("falls back to listing everything when no questions have been asked yet", async () => {
    (apiClient.getEvidence as any).mockResolvedValue([VIDEO_ROW, PDF_ROW]);
    (apiClient.getEvidenceByQuery as any).mockResolvedValue([]);
    render(<EvidencePage />);
    await waitFor(() => expect(screen.getByText(/All evidence from: Redis Architecture/)).toBeInTheDocument());
    expect(screen.queryByTestId("query-group-0")).not.toBeInTheDocument();
  });

  it("sends the typed question to the backend instead of listing everything", async () => {
    (apiClient.getEvidence as any).mockResolvedValue([VIDEO_ROW, PDF_ROW]);
    render(<EvidencePage />);
    await waitFor(() => expect(apiClient.getEvidence).toHaveBeenCalledWith(200, ""));

    fireEvent.change(screen.getByTestId("evidence-search"), {
      target: { value: "what are the specializations in AI/ML?" },
    });
    fireEvent.submit(screen.getByTestId("evidence-search").closest("form")!);

    // Searching asks for a small top-N, not the whole corpus re-sorted.
    await waitFor(() =>
      expect(apiClient.getEvidence).toHaveBeenCalledWith(25, "what are the specializations in AI/ML?"),
    );
  });

  it("shows ranked matches for the question rather than session groups", async () => {
    const hit = { ...PDF_ROW, id: "hit", content_text: "Specializes in NLP and computer vision.", relevance: 0.91 };
    const weak = { ...VIDEO_ROW, id: "weak", content_text: "Contact: someone@example.com", relevance: 0.42 };
    (apiClient.getEvidence as any).mockResolvedValue([hit, weak]);

    render(<EvidencePage />);
    fireEvent.change(screen.getByTestId("evidence-search"), { target: { value: "specializations" } });
    fireEvent.submit(screen.getByTestId("evidence-search").closest("form")!);

    await waitFor(() => expect(screen.getByTestId("results-heading")).toHaveTextContent("specializations"));
    // Relevance is shown, and the misleading session grouping is gone.
    expect(screen.queryByText(/All evidence from/)).not.toBeInTheDocument();
    const scores = screen.getAllByTestId("relevance-score").map((n) => n.textContent);
    expect(scores).toEqual(["91% match", "42% match"]);
  });

  it("labels extraction confidence so it can't be misread as relevance", async () => {
    (apiClient.getEvidence as any).mockResolvedValue([PDF_ROW]);
    render(<EvidencePage />);
    // PDF_ROW has confidence 0.98 but no relevance -- previously that rendered
    // as a bare "98%", which looked like a 98% match to the question.
    await waitFor(() => expect(screen.getAllByText("Extraction").length).toBeGreaterThan(0));
    expect(screen.queryByTestId("relevance-score")).not.toBeInTheDocument();
  });

  it("clearing the search goes back to browsing everything", async () => {
    (apiClient.getEvidence as any).mockResolvedValue([VIDEO_ROW, PDF_ROW]);
    render(<EvidencePage />);
    fireEvent.change(screen.getByTestId("evidence-search"), { target: { value: "redis" } });
    fireEvent.submit(screen.getByTestId("evidence-search").closest("form")!);
    await waitFor(() => expect(screen.getByTestId("results-heading")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: /clear/i }));
    await waitFor(() => expect(screen.queryByTestId("results-heading")).not.toBeInTheDocument());
    expect(screen.getByText(/All evidence from: Redis Architecture/)).toBeInTheDocument();
  });

  it("says nothing matched instead of silently showing unrelated evidence", async () => {
    (apiClient.getEvidence as any).mockResolvedValue([]);
    render(<EvidencePage />);
    fireEvent.change(screen.getByTestId("evidence-search"), { target: { value: "quantum tunnelling" } });
    fireEvent.submit(screen.getByTestId("evidence-search").closest("form")!);
    await waitFor(() =>
      expect(screen.getByTestId("evidence-empty")).toHaveTextContent("No evidence matches “quantum tunnelling”"),
    );
  });

  it("clicking a video evidence row opens the player seeked to its exact timestamp", async () => {
    (apiClient.getEvidence as any).mockResolvedValue([VIDEO_ROW]);
    render(<EvidencePage />);
    await waitFor(() => expect(screen.getByText(/we should cache with redis here/i)).toBeInTheDocument());

    fireEvent.click(screen.getByText(/we should cache with redis here/i));

    const video = await screen.findByTestId("video-player") as HTMLVideoElement;
    expect(video.src).toContain("/media/job1_meeting.mp4");
    fireEvent.loadedMetadata(video);
    expect(video.currentTime).toBe(430);
  });

  it("clicking a PDF evidence row opens it on its exact page with the extracted text alongside", async () => {
    (apiClient.getEvidence as any).mockResolvedValue([PDF_ROW]);
    render(<EvidencePage />);
    await waitFor(() => expect(screen.getByText(/page discusses the caching layer diagram/i)).toBeInTheDocument());

    fireEvent.click(screen.getByText(/page discusses the caching layer diagram/i));

    const iframe = await screen.findByTestId("pdf-viewer") as HTMLIFrameElement;
    expect(iframe.src).toContain("/media/job2_notes.pdf#page=3");
    expect(screen.getByText("Extracted text on this page")).toBeInTheDocument();
  });
});
