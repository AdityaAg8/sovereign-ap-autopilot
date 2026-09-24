/** Dedicated worker that loads pdf.js and paints invoice pages into the viewer's canvas via
 * OffscreenCanvas, so pdf.js's ~0.5 MB display library is compiled and evaluated off the main
 * thread (it was the invoice page's one 200 ms long task). pdf.js's own parsing worker runs as
 * a nested worker; the protocol is typed in pdf-render-protocol.ts and driven by pdf-canvas.tsx. */

import { getDocument, GlobalWorkerOptions } from "@/components/invoice/pdfjs";
import type { PDFDocumentLoadingTask, PDFDocumentProxy, RenderTask } from "pdfjs-dist";
import type { RenderRequest, RenderResponse } from "@/components/invoice/pdf-render-protocol";

const scope = self as unknown as {
  postMessage: (message: RenderResponse) => void;
  onmessage: ((event: MessageEvent<RenderRequest>) => void) | null;
};

/** Canvases pdf.js needs for patterns, masks and groups: OffscreenCanvas instead of the default
 * factory's `document.createElement("canvas")`, which does not exist in a worker. */
class OffscreenCanvasFactory {
  create(width: number, height: number) {
    if (width <= 0 || height <= 0) throw new Error("Invalid canvas size");
    const canvas = new OffscreenCanvas(width, height);
    return { canvas, context: canvas.getContext("2d", { willReadFrequently: true }) };
  }
  reset(entry: { canvas: OffscreenCanvas | null }, width: number, height: number) {
    if (!entry.canvas) throw new Error("Canvas is not specified");
    if (width <= 0 || height <= 0) throw new Error("Invalid canvas size");
    entry.canvas.width = width;
    entry.canvas.height = height;
  }
  destroy(entry: { canvas: OffscreenCanvas | null; context: unknown }) {
    if (!entry.canvas) throw new Error("Canvas is not specified");
    entry.canvas.width = entry.canvas.height = 0;
    entry.canvas = null;
    entry.context = null;
  }
}

/** SVG-filter effects (transfer functions, high-contrast page colours) need a DOM to hold the
 * filters; without one pdf.js's own base factory behaviour applies: no filter ("none"). Invoice
 * pages do not use them; the rest of the page renders exactly. */
class NoFilterFactory {
  addFilter() {
    return "none";
  }
  addHCMFilter() {
    return "none";
  }
  addAlphaFilter() {
    return "none";
  }
  addLuminosityFilter() {
    return "none";
  }
  addKnockoutFilter() {
    return "none";
  }
  addHighlightHCMFilter() {
    return "none";
  }
  addSelectionHCMFilter() {
    return "none";
  }
  addSelectionFilter() {
    return "none";
  }
  createSelectionStyle() {
    return null;
  }
  destroy() {}
}

let canvas: OffscreenCanvas | null = null;
let loadingTask: PDFDocumentLoadingTask | null = null;
let doc: PDFDocumentProxy | null = null;
let renderTask: RenderTask | null = null;
let latestRender = 0;

/** Upper bound on opening a document (parser worker start-up plus the PDF fetch and parse). pdf.js
 * itself never gives up on a parser worker that failed to start or a stalled fetch; past this the
 * viewer shows its explicit error state instead of "Loading document…" forever. */
const LOAD_TIMEOUT_MS = 30_000;

async function open(request: Extract<RenderRequest, { type: "open" }>) {
  canvas = request.canvas;
  // A module worker of our own for pdf.js's parser: pdf.js would otherwise look up
  // `window.location` (absent in a worker) to vet a `workerSrc` URL.
  const parser = new Worker(request.parserWorkerUrl, { type: "module" });
  GlobalWorkerOptions.workerPort = parser;
  let settled = false;
  const fail = () => {
    if (settled) return;
    settled = true;
    clearTimeout(timeout);
    scope.postMessage({ type: "error", stage: "load" });
    void loadingTask?.destroy();
  };
  // A parser script that fails to load or evaluate (404, network error, syntax error) surfaces
  // only as an `error` event on this nested worker; pdf.js would keep waiting for its "ready".
  parser.addEventListener("error", fail);
  const timeout = setTimeout(fail, LOAD_TIMEOUT_MS);
  // The bearer token arrives only with this one `open` message (pdf-render-protocol.ts) and goes
  // only to this document's own fetch. There is no `isEvalSupported` to turn off: the pinned
  // pdf.js (6.3) no longer has that option or the eval-compiled PostScript functions it guarded,
  // and neither of its bundles calls `eval` or `new Function`.
  loadingTask = getDocument({
    url: new URL(request.documentUrl),
    httpHeaders: request.authToken ? { Authorization: `Bearer ${request.authToken}` } : undefined,
    standardFontDataUrl: request.standardFontDataUrl,
    // Let the parser worker fetch font data itself: pdf.js's main-side fetch helper resolves URLs
    // against `document.baseURI`, which a worker does not have.
    useWorkerFetch: true,
    // No FontFace/document in a worker: glyphs are drawn from the embedded font programs.
    disableFontFace: true,
    CanvasFactory: OffscreenCanvasFactory,
    FilterFactory: NoFilterFactory,
  } as Parameters<typeof getDocument>[0]);
  try {
    const loaded = await loadingTask.promise;
    if (settled) return;
    settled = true;
    clearTimeout(timeout);
    doc = loaded;
    scope.postMessage({ type: "loaded", pageCount: doc.numPages });
  } catch {
    fail();
  }
}

async function render(request: Extract<RenderRequest, { type: "render" }>) {
  latestRender = request.id;
  renderTask?.cancel();
  renderTask = null;
  if (!doc || !canvas) return;
  try {
    const page = await doc.getPage(request.page + 1);
    if (request.id !== latestRender) return;
    const unscaled = page.getViewport({ scale: 1 });
    const fitScale = Math.min(
      request.boxWidth / unscaled.width,
      request.boxHeight / unscaled.height,
    );
    const viewport = page.getViewport({ scale: fitScale });
    canvas.width = Math.floor(viewport.width * request.pixelRatio);
    canvas.height = Math.floor(viewport.height * request.pixelRatio);
    scope.postMessage({
      type: "sized",
      id: request.id,
      width: Math.floor(viewport.width),
      height: Math.floor(viewport.height),
    });
    const transform =
      request.pixelRatio !== 1 ? [request.pixelRatio, 0, 0, request.pixelRatio, 0, 0] : undefined;
    renderTask = page.render({
      canvas: canvas as unknown as HTMLCanvasElement,
      viewport,
      transform,
    });
    await renderTask.promise;
  } catch (err: unknown) {
    const cancelled = err instanceof Error && err.name === "RenderingCancelledException";
    if (!cancelled && request.id === latestRender) {
      scope.postMessage({ type: "error", stage: "render", id: request.id });
    }
  }
}

scope.onmessage = (event) => {
  const request = event.data;
  if (request.type === "open") void open(request);
  else if (request.type === "render") void render(request);
  else if (request.type === "close") {
    renderTask?.cancel();
    void loadingTask?.destroy();
    self.close();
  }
};
