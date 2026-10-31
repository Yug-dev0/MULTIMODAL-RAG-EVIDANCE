import "@testing-library/jest-dom";

// jsdom has no ResizeObserver; GraphView uses it to keep the WebGL canvas
// sized to its container.
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as any;
}

// jsdom has no scrollIntoView; ChatPage uses it to keep the thread pinned
// to the latest message.
if (typeof Element !== "undefined" && !Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {};
}
