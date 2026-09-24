/** Messages between the invoice viewer (pdf-canvas.tsx) and its pdf.js render worker
 * (pdf-render-worker.ts). The worker owns the page canvas (transferred as an OffscreenCanvas);
 * the viewer asks for a page fitted to its box and learns the rendered CSS size to lay out the
 * field-box overlay over it. */

export type RenderRequest =
  | {
      type: "open";
      canvas: OffscreenCanvas;
      /** Absolute URLs: a worker has no document to resolve relative ones against. */
      documentUrl: string;
      parserWorkerUrl: string;
      standardFontDataUrl: string;
      authToken: string | null;
    }
  | {
      type: "render";
      /** Increases per request; responses for an older id are stale. */
      id: number;
      /** 0-indexed page number. */
      page: number;
      boxWidth: number;
      boxHeight: number;
      pixelRatio: number;
    }
  | { type: "close" };

export type RenderResponse =
  | { type: "loaded"; pageCount: number }
  | { type: "sized"; id: number; width: number; height: number }
  | { type: "error"; stage: "load" | "render"; id?: number };
