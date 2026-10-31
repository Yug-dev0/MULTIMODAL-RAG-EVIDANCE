import type { GraphNode, GraphLink } from "../GraphView";

const NODE_COLOR: Record<string, string> = { entity: "#F5B052", event: "#6EEBC3", evidence: "#7FA8DE" };

interface Props {
  nodes: GraphNode[];
  links: GraphLink[];
  onOpenGraph: () => void;
}

export function KnowledgePulse({ nodes, links, onOpenGraph }: Props) {
  if (nodes.length === 0) {
    return (
      <div className="card flex h-full min-h-[260px] flex-col p-4">
        <p className="eyebrow mb-3">Knowledge pulse</p>
        <div className="flex flex-1 items-center justify-center">
          <p className="max-w-[220px] text-center font-mono text-[11px] text-dim">
            Ingest a source to see how evidence, entities, and events connect.
          </p>
        </div>
      </div>
    );
  }

  const byId = new Map(nodes.map((n) => [n.id, n]));
  const degree = new Map<string, number>();
  for (const l of links) {
    degree.set(l.source, (degree.get(l.source) ?? 0) + 1);
    degree.set(l.target, (degree.get(l.target) ?? 0) + 1);
  }
  const hub = [...nodes].sort((a, b) => (degree.get(b.id) ?? 0) - (degree.get(a.id) ?? 0))[0];
  const neighborLinks = links
    .filter((l) => l.source === hub.id || l.target === hub.id)
    .slice(0, 6);
  const neighbors = neighborLinks.map((l) => ({
    node: byId.get(l.source === hub.id ? l.target : l.source),
    relationship: l.relationship_type,
    outgoing: l.source === hub.id,
  })).filter((n) => n.node);

  const cx = 130, cy = 130, r = 92;
  const angleStep = (2 * Math.PI) / Math.max(neighbors.length, 1);

  return (
    <div className="card flex h-full flex-col p-4">
      <div className="mb-1 flex items-center justify-between">
        <p className="eyebrow">Knowledge pulse</p>
        <button onClick={onOpenGraph} className="font-mono text-[10px] text-signal hover:underline">Open full graph →</button>
      </div>
      <svg viewBox="0 0 260 260" className="w-full flex-1">
        {neighbors.map((n, i) => {
          const angle = i * angleStep - Math.PI / 2;
          const x = cx + r * Math.cos(angle);
          const y = cy + r * Math.sin(angle);
          return (
            <g key={n.node!.id}>
              <line x1={cx} y1={cy} x2={x} y2={y} stroke="rgb(var(--c-hairline))" strokeWidth="1.5" />
              <text x={(cx + x) / 2} y={(cy + y) / 2 - 4} textAnchor="middle" className="fill-dim" fontSize="6.5" fontFamily="monospace">
                {n.relationship}
              </text>
              <circle cx={x} cy={y} r="14" fill={NODE_COLOR[n.node!.type]} opacity="0.9" />
              <text x={x} y={y + 26} textAnchor="middle" className="fill-current text-ink" fontSize="8.5" fontWeight="500">
                {n.node!.label.length > 16 ? `${n.node!.label.slice(0, 14)}…` : n.node!.label}
              </text>
            </g>
          );
        })}
        <circle cx={cx} cy={cy} r="20" fill={NODE_COLOR[hub.type]} stroke="rgb(var(--c-void))" strokeWidth="3" />
        <text x={cx} y={cy + 36} textAnchor="middle" className="fill-current text-ink" fontSize="9.5" fontWeight="600">
          {hub.label.length > 18 ? `${hub.label.slice(0, 16)}…` : hub.label}
        </text>
      </svg>
    </div>
  );
}
