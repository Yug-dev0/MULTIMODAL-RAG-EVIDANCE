// Deterministic line-dash pattern per relationship_type. The set of patterns
// is fixed, but WHICH relationship_type gets which pattern is computed from
// whatever types are actually present in the live graph -- never a hardcoded
// list of example relationship names.
const DASH_PATTERNS: number[][] = [
  [], // solid
  [6, 3],
  [1.5, 3],
  [8, 3, 1.5, 3],
  [2, 2],
  [10, 2],
];

// A handful of structural/semantic types get a fixed, meaningful pattern so
// they read consistently across sessions; anything else is assigned by a
// stable hash so the same relationship_type always renders the same way.
const FIXED: Record<string, number[]> = {
  explains: [],
  contradicts: [2, 2],
  corresponds_to: [6, 3],
  mentions: [1.5, 3],
  belongs_to: [1.5, 3],
};

function hashString(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h;
}

export function dashPatternFor(relationshipType: string): number[] {
  const key = relationshipType.toLowerCase();
  if (FIXED[key]) return FIXED[key];
  return DASH_PATTERNS[hashString(key) % DASH_PATTERNS.length];
}

export function relationshipStylesInUse(relationshipTypes: string[]): { type: string; dash: number[] }[] {
  const unique = [...new Set(relationshipTypes)].sort();
  return unique.map((type) => ({ type, dash: dashPatternFor(type) }));
}
