const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/**
 * raw_locator is a backend filesystem path (e.g. "./data/media/<uuid>/frame_0001.jpg"
 * or "video.mp4#t=1,2"). Only paths actually under the media directory are
 * servable images; anything else (a fragment-suffixed source reference) has
 * no browser-viewable file and returns null.
 */
export function mediaUrl(rawLocator: string): string | null {
  const marker = "data/media/";
  const idx = rawLocator.indexOf(marker);
  if (idx === -1) return null;
  const relative = rawLocator.slice(idx + marker.length);
  if (relative.includes("#")) return null;
  return `${BASE_URL}/media/${relative}`;
}

export function isImagePath(rawLocator: string): boolean {
  return /\.(png|jpe?g|webp|gif|bmp)$/i.test(rawLocator);
}

/**
 * Whole-file evidence (speech/pdf-text units) has a raw_locator like
 * "./data/media/<job>_<name>.mp4#t=430.0,440.0" -- the fragment carries the
 * same timestamp/page already present as structured fields on the row, so
 * this only needs to resolve the file itself into a URL a <video>/<iframe>
 * can load; the caller passes seekToSeconds/pageNumber separately.
 */
export function mediaFileUrl(rawLocator: string): string | null {
  const marker = "data/media/";
  const base = rawLocator.split("#")[0];
  const idx = base.indexOf(marker);
  if (idx === -1) return null;
  return `${BASE_URL}/media/${base.slice(idx + marker.length)}`;
}
