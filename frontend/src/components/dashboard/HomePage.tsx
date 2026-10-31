import { useState } from "react";
import type { GraphNode, GraphLink } from "../GraphView";
import { Dashboard } from "./Dashboard";
import { HomeIntro } from "./HomeIntro";

interface Props {
  onSearch: (q: string) => void;
  onOpenEvent: (id: string) => void;
  onOpenGraph: () => void;
  graphData: { nodes: GraphNode[]; links: GraphLink[] };
}

export function HomePage({ onSearch, onOpenEvent, onOpenGraph, graphData }: Props) {
  // Plays on every visit to Home, not just the first -- it's part of the
  // page's identity, not a one-time onboarding moment.
  const [introDone, setIntroDone] = useState(false);

  return (
    <div className="relative min-h-[70vh]">
      <Dashboard onSearch={onSearch} onOpenEvent={onOpenEvent} onOpenGraph={onOpenGraph} graphData={graphData} />
      {!introDone && <HomeIntro onDone={() => setIntroDone(true)} />}
    </div>
  );
}
