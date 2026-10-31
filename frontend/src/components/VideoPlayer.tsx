import { useEffect, useRef } from "react";

export function VideoPlayer({ src, seekToSeconds }: { src: string; seekToSeconds?: number }) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || seekToSeconds === undefined) return;
    // Setting currentTime before the browser has loaded the video's
    // metadata (readyState 0, the common case right after src changes) is
    // silently dropped -- it must be applied once loadedmetadata fires, not
    // synchronously on mount.
    const applySeek = () => {
      video.currentTime = seekToSeconds;
    };
    if (video.readyState >= 1) {
      applySeek();
      return;
    }
    video.addEventListener("loadedmetadata", applySeek);
    return () => video.removeEventListener("loadedmetadata", applySeek);
  }, [seekToSeconds, src]);

  return (
    <div className="bracket-frame border border-hairline bg-panel p-1">
      <video ref={videoRef} src={src} controls className="w-full" data-testid="video-player" />
    </div>
  );
}
