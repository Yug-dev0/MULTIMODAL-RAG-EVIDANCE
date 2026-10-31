export const SOURCE_TYPES = ["video", "audio", "image", "pdf"] as const;
export type SourceType = (typeof SOURCE_TYPES)[number];

const EXTENSION_TO_TYPE: Record<string, SourceType> = {
  mp4: "video", mov: "video", mkv: "video", avi: "video", webm: "video", m4v: "video",
  mp3: "audio", wav: "audio", m4a: "audio", flac: "audio", ogg: "audio", aac: "audio",
  png: "image", jpg: "image", jpeg: "image", webp: "image", gif: "image", bmp: "image",
  pdf: "pdf",
};

export function detectSourceType(filename: string): SourceType | null {
  const ext = filename.split(".").pop()?.toLowerCase();
  return ext ? EXTENSION_TO_TYPE[ext] ?? null : null;
}
