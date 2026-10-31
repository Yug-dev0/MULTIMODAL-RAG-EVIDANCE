import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { GraphView, GraphLegend, GraphStatus } from "./GraphView";

vi.mock("force-graph", () => {
  function makeChainable(): any {
    let proxy: any;
    const handler = { get: () => (..._args: any[]) => proxy };
    proxy = new Proxy({}, handler);
    return proxy;
  }
  return {
    default: class {
      constructor() {
        return makeChainable();
      }
    },
  };
});

describe("GraphView", () => {
  it("renders a container for the graph without crashing", () => {
    render(<GraphView nodes={[{ id: "1", label: "Redis", type: "entity" }]} links={[]} active />);
    expect(screen.getByTestId("graph-view")).toBeInTheDocument();
  });

  it("shows an empty-state hint when there are no nodes", () => {
    render(<GraphView nodes={[]} links={[]} active />);
    expect(screen.getByText(/no graph data yet/i)).toBeInTheDocument();
  });
});

describe("GraphLegend", () => {
  it("lists node type colors and only relationship types actually present in the data", () => {
    render(
      <GraphLegend
        nodes={[{ id: "1", label: "Redis", type: "entity" }]}
        links={[{ source: "1", target: "2", relationship_type: "explains" }]}
      />,
    );
    expect(screen.getByText("Entity")).toBeInTheDocument();
    expect(screen.getByText("explains")).toBeInTheDocument();
  });
});

describe("GraphStatus", () => {
  it("shows real node and edge counts", () => {
    render(<GraphStatus nodes={[{ id: "1", label: "Redis", type: "entity" }]} links={[]} />);
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("0")).toBeInTheDocument();
  });
});
