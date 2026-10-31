import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { EvidenceCard } from "./EvidenceCard";
import type { Citation } from "../api/client";

const CITATION: Citation = {
  evidence_id: "e1", modality: "speech", source: "./data/media/job_meeting.mp4#t=430.0,440.0",
  content: "We propose Redis to reduce database load", timestamp_start: 430.0, timestamp_end: 440.0,
  page_number: null, thumbnail: null, confidence: 0.9,
};

describe("EvidenceCard", () => {
  it("shows a timestamp badge derived from the citation, not a raw file path", () => {
    render(<EvidenceCard citation={CITATION} />);
    expect(screen.getByText("7:10–7:20")).toBeInTheDocument();
  });

  it("shows a page badge for document citations", () => {
    const pdfCitation: Citation = { ...CITATION, modality: "document", timestamp_start: null, timestamp_end: null, page_number: 3 };
    render(<EvidenceCard citation={pdfCitation} />);
    expect(screen.getByText("Page 3")).toBeInTheDocument();
  });

  it("renders as a plain (non-interactive) card when no onClick is given", () => {
    render(<EvidenceCard citation={CITATION} />);
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("is clickable and invokes onClick when a handler is provided", () => {
    const onClick = vi.fn();
    render(<EvidenceCard citation={CITATION} onClick={onClick} />);
    fireEvent.click(screen.getByRole("button"));
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
