/** Renders one page of a PDF document into a canvas with pdf.js (PRD §10 M5.4 "pdf.js viewer").
 * Loaded dynamically inside an effect so pdfjs-dist — which touches `Path2D`/`DOMMatrix` at
 * import time — never runs during Next.js's server render pass. Reports the on-screen page size
 * so the caller can size a same-size overlay for the grounded field boxes exactly over the
 * rendered page, not over the reserved aspect-ratio frame (which may not match the PDF's own
 * aspect ratio for a scanned/non-A4 source page). */

"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { Loader2 } from "lucide-react";
import type { PDFDocumentLoadingTask, PDFDocumentProxy, RenderTask } from "pdfjs-dist";

export interface PageSize {
  width: number;
  height: number;
}

// pdfjs-dist's worker and base-14 font metrics are served as static files copied from
// node_modules/pdfjs-dist (`web/public/pdfjs`, see that copy's provenance note below) rather than
// resolved via `new URL(..., import.meta.url)`, so the path is one plain string under every
// bundler this app builds with (`next build --turbopack` and plain webpack alike). Re-run:
//   cp node_modules/pdfjs-dist/build/pdf.worker.min.mjs web/public/pdfjs/
//   cp node_modules/pdfjs-dist/standard_fonts/* web/public/pdfjs/standard_fonts/
// after bumping the `pdfjs-dist` dependency.
const WORKER_SRC = "/pdfjs/pdf.worker.min.mjs";
const STANDARD_FONT_DATA_URL = "/pdfjs/standard_fonts/";

let workerConfigured = false;

/** Resolves after the frame that paints the current commit. pdf.js is ~150 KB of JS; requesting
 * it only once the field list is on screen keeps it off /invoices/:id's LCP path (a measured
 * 4.9 s simulated mobile LCP with it requested in the same commit as the field list). */
async function afterPaint(): Promise<void> {
  const nextPaint = () =>
    new Promise<void>((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(resolve, 0))),
    );
  await nextPaint();
  // Text drawn in a web font that is still loading is painted (and counted for LCP) only in the
  // frame after the font arrives, so wait for the fonts and for that frame too.
  await document.fonts.ready;
  await nextPaint();
}

async function loadPdfjs() {
  await afterPaint();
  const pdfjsLib = await import("pdfjs-dist");
  if (!workerConfigured) {
    pdfjsLib.GlobalWorkerOptions.workerSrc = WORKER_SRC;
    workerConfigured = true;
  }
  return pdfjsLib;
}

export type PdfCanvasStatus = "loading" | "ready" | "error";

export function PdfCanvas({
  documentUrl,
  authToken,
  page,
  onDocumentLoad,
  onStatusChange,
  children,
}: {
  documentUrl: string;
  /** Bearer token attached as pdf.js's `httpHeaders` (a "Basic authentication headers" option
   * pdf.js also uses for bearer tokens — mozilla/pdf.js `DocumentInitParameters`), needed
   * because the http adapter's `documentUrl` points at an apx/api route that requires the same
   * `Authorization` header as every other request (apx.api.deps.get_current_user). `null` for
   * the fixture adapter's unauthenticated static sample PDF. */
  authToken: string | null;
  /** 0-indexed page number. */
  page: number;
  onDocumentLoad?: (pageCount: number) => void;
  onStatusChange?: (status: PdfCanvasStatus) => void;
  /** Rendered inside a wrapper sized to exactly match the rendered page in CSS pixels — e.g. the
   * grounded field-box overlay, which positions its boxes as percentages of that same rect. */
  children?: ReactNode;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const docRef = useRef<PDFDocumentProxy | null>(null);
  const taskRef = useRef<PDFDocumentLoadingTask | null>(null);
  const [status, setStatus] = useState<PdfCanvasStatus>("loading");
  const [pageSize, setPageSize] = useState<PageSize | null>(null);

  function setStatusReported(next: PdfCanvasStatus) {
    setStatus(next);
    onStatusChange?.(next);
  }

  // Load the document whenever its URL or auth token changes.
  useEffect(() => {
    let cancelled = false;
    setStatusReported("loading");
    setPageSize(null);
    docRef.current = null;
    void taskRef.current?.destroy();
    taskRef.current = null;

    loadPdfjs()
      .then((pdfjsLib) => {
        const task = pdfjsLib.getDocument({
          url: documentUrl,
          standardFontDataUrl: STANDARD_FONT_DATA_URL,
          httpHeaders: authToken ? { Authorization: `Bearer ${authToken}` } : undefined,
        });
        taskRef.current = task;
        return task.promise;
      })
      .then((doc) => {
        if (cancelled) {
          void taskRef.current?.destroy();
          return;
        }
        docRef.current = doc;
        onDocumentLoad?.(doc.numPages);
        setStatusReported("ready");
      })
      .catch(() => {
        if (!cancelled) setStatusReported("error");
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documentUrl, authToken]);

  // Render the requested page, fit to the reserved container, whenever the doc or page changes.
  useEffect(() => {
    if (status !== "ready") return;
    const doc = docRef.current;
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!doc || !container || !canvas) return;
    if (page < 0 || page >= doc.numPages) return;

    let cancelled = false;
    let renderTask: RenderTask | null = null;

    (async () => {
      const pdfPage = await doc.getPage(page + 1);
      if (cancelled) return;
      const unscaled = pdfPage.getViewport({ scale: 1 });
      const fitScale = Math.min(
        container.clientWidth / unscaled.width,
        container.clientHeight / unscaled.height,
      );
      const viewport = pdfPage.getViewport({ scale: fitScale });
      const outputScale = window.devicePixelRatio || 1;

      canvas.width = Math.floor(viewport.width * outputScale);
      canvas.height = Math.floor(viewport.height * outputScale);
      const cssWidth = Math.floor(viewport.width);
      const cssHeight = Math.floor(viewport.height);
      canvas.style.width = `${cssWidth}px`;
      canvas.style.height = `${cssHeight}px`;
      if (!cancelled) setPageSize({ width: cssWidth, height: cssHeight });

      const transform = outputScale !== 1 ? [outputScale, 0, 0, outputScale, 0, 0] : undefined;
      renderTask = pdfPage.render({ canvas, viewport, transform });
      await renderTask.promise;
    })().catch((err: unknown) => {
      const isCancelled = err instanceof Error && err.name === "RenderingCancelledException";
      if (!cancelled && !isCancelled) setStatusReported("error");
    });

    return () => {
      cancelled = true;
      renderTask?.cancel();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, page]);

  return (
    <div ref={containerRef} className="relative flex h-full w-full items-center justify-center">
      <div aria-live="polite" className="contents">
        {status === "loading" && (
          <div className="text-muted-foreground absolute inset-0 flex flex-col items-center justify-center gap-2">
            <Loader2 className="size-5 animate-spin" aria-hidden />
            <p className="text-xs">Loading document…</p>
          </div>
        )}
        {status === "error" && (
          <p className="text-muted-foreground max-w-xs p-6 text-center text-xs">
            Couldn&apos;t render this document. It may be corrupted or unreachable.
          </p>
        )}
      </div>
      <div
        className="relative"
        style={
          // `display: none` (not a 0x0 box) until the page is sized: a bordered 0x0 box that later
          // grows from the centre registers as a layout shift, a newly displayed one does not.
          pageSize ? { width: pageSize.width, height: pageSize.height } : { display: "none" }
        }
      >
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={`Invoice document, page ${page + 1}`}
          className="border-border rounded-sm border shadow-sm"
        />
        {pageSize && children}
      </div>
    </div>
  );
}
