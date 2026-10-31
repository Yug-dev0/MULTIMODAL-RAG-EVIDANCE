// Compact 24x24-viewBox icon paths, rasterized on the graph canvas via Path2D.
// Picked per real entity_type / event_type / modality value -- never per a
// fixed node id, so an icon always reflects the actual field it's keyed on.
const ICON_PATHS: Record<string, string> = {
  person: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4 21a8 8 0 0 1 16 0",
  database: "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6",
  chip: "M8 3v3M16 3v3M8 18v3M16 18v3M3 8h3M3 16h3M18 8h3M18 16h3M7 7h10v10H7z",
  lightbulb: "M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9c.6.5 1 1.2 1 2.1h5c0-.9.4-1.6 1-2.1A6 6 0 0 0 12 3z",
  gavel: "m14 5 5 5M2 22l7-7M9.5 8.5 15.5 14.5M12.5 5.5l6 6M4 17l3 3",
  video: "M3 6h12v12H3zM15 9l6-3v12l-6-3",
  mic: "M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3zM6 11a6 6 0 0 0 12 0M12 17v4M9 21h6",
  grid: "M4 5h16M4 11h16M4 17h10",
  file: "M6 2h9l5 5v15H6zM15 2v5h5",
  circle: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z",
};

/** Real entity_type value -> icon key. Falls back to a generic circle for any type not mapped. */
function entityIcon(entityType: string | null | undefined): string {
  const t = (entityType ?? "").toLowerCase();
  if (t.includes("person") || t.includes("speaker")) return "person";
  if (t.includes("database") || t.includes("cache") || t.includes("data")) return "database";
  if (t.includes("api") || t.includes("service") || t.includes("component") || t.includes("gateway")) return "chip";
  return "circle";
}

/** Real event_type value -> icon key (discussion/proposal read as "idea", decision reads as "ruling"). */
function eventIcon(eventType: string | null | undefined): string {
  const t = (eventType ?? "").toLowerCase();
  if (t.includes("decision") || t.includes("action_item")) return "gavel";
  return "lightbulb";
}

/** Real modality value -> icon key. */
function evidenceIcon(modality: string | null | undefined): string {
  const m = (modality ?? "").toLowerCase();
  if (m === "speech") return "mic";
  if (m === "image") return "video";
  if (m === "ocr") return "grid";
  if (m === "document") return "file";
  return "circle";
}

export function iconKeyFor(node: { type: string; entity_type?: string | null; event_type?: string | null; modality?: string | null }): string {
  if (node.type === "entity") return entityIcon(node.entity_type);
  if (node.type === "event") return eventIcon(node.event_type);
  return evidenceIcon(node.modality);
}

const pathCache = new Map<string, Path2D>();
function getPath(key: string): Path2D {
  let p = pathCache.get(key);
  if (!p) {
    p = new Path2D(ICON_PATHS[key] ?? ICON_PATHS.circle);
    pathCache.set(key, p);
  }
  return p;
}

export function drawNodeGlyph(ctx: CanvasRenderingContext2D, node: { type: string; entity_type?: string | null; event_type?: string | null; modality?: string | null }, x: number, y: number, radius: number, color: string) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
  ctx.fillStyle = `${color}33`;
  ctx.fill();
  ctx.lineWidth = radius * 0.12;
  ctx.strokeStyle = color;
  ctx.stroke();

  const key = iconKeyFor(node);
  const scale = (radius * 1.15) / 24;
  ctx.translate(x - 12 * scale, y - 12 * scale);
  ctx.scale(scale, scale);
  ctx.lineWidth = 2.2;
  ctx.strokeStyle = color;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.stroke(getPath(key));
  ctx.restore();
}
