import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { CompareView } from "./CompareView";
import { apiClient } from "../api/client";

vi.mock("../api/client", () => ({ apiClient: { compare: vi.fn() } }));

describe("CompareView", () => {
  it("shows both systems' answers after comparing", async () => {
    (apiClient.compare as any).mockResolvedValue({
      multimodal: { answer: "Redis, explained over the architecture diagram.", citations: [{}], events_used: ["e1"] },
      baseline: { answer: "Redis was mentioned." },
    });
    render(<CompareView />);
    fireEvent.change(screen.getByPlaceholderText(/compare a query/i), { target: { value: "What reduces load?" } });
    fireEvent.click(screen.getByText("Compare"));
    await waitFor(() => expect(screen.getByTestId("multimodal-answer")).toHaveTextContent("architecture diagram"));
    expect(screen.getByTestId("baseline-answer")).toHaveTextContent("Redis was mentioned.");
  });
});
