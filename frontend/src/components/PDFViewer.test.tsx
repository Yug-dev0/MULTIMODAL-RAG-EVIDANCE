import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { PDFViewer } from "./PDFViewer";

describe("PDFViewer", () => {
  it("includes the page number as a URL fragment", () => {
    render(<PDFViewer src="doc.pdf" pageNumber={3} />);
    const iframe = screen.getByTestId("pdf-viewer") as HTMLIFrameElement;
    expect(iframe.src).toContain("doc.pdf#page=3");
  });
});
