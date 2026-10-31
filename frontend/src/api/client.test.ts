import { describe, it, expect, vi, beforeEach } from "vitest";
import { apiClient } from "./client";

describe("apiClient", () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  it("query posts to /query and returns parsed JSON", async () => {
    (globalThis.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({ answer: "Redis", citations: [], events_used: [] }),
    });
    const result = await apiClient.query("What reduces load?");
    expect(result.answer).toBe("Redis");
    expect(globalThis.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/query"),
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("getIngestStatus throws on a non-ok response", async () => {
    (globalThis.fetch as any).mockResolvedValue({ ok: false, status: 404 });
    await expect(apiClient.getIngestStatus("bad-id")).rejects.toThrow();
  });
});
