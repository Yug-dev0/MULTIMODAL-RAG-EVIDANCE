import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { VideoPlayer } from "./VideoPlayer";

describe("VideoPlayer", () => {
  it("renders with the given source", () => {
    render(<VideoPlayer src="video.mp4" />);
    const video = screen.getByTestId("video-player") as HTMLVideoElement;
    expect(video.src).toContain("video.mp4");
  });

  it("seeks to the given timestamp once the browser has loaded the video's metadata", () => {
    // Regression test: setting currentTime before loadedmetadata fires
    // (readyState 0, the state right after src is set) is silently ignored
    // by real browsers -- jsdom doesn't model this, so a naive "set it on
    // mount" implementation looked correct in tests while doing nothing in
    // an actual browser.
    render(<VideoPlayer src="video.mp4" seekToSeconds={5} />);
    const video = screen.getByTestId("video-player") as HTMLVideoElement;
    expect(video.currentTime).toBe(0);

    fireEvent.loadedMetadata(video);
    expect(video.currentTime).toBe(5);
  });
});
