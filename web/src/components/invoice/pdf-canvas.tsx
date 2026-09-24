/** Renders one page of a PDF document into a canvas with pdf.js (PRD §10 M5.4 "pdf.js viewer").
 * pdf.js runs entirely in a dedicated worker (pdf-render-worker.ts) that paints into this
 * canvas through OffscreenCanvas, so neither its display library nor its parser ever blocks the
 * main thread. Reports the on-screen page size so the caller can size a same-size overlay for
 * the grounded field boxes exactly over the rendered page, not over the reserved aspect-ratio
 * frame (which may not match the PDF's own aspect ratio for a scanned/non-A4 source page). */

"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { Loader2 } from "lucide-react";
import type { RenderRequest, RenderResponse } from "@/components/invoice/pdf-render-protocol";

export interface PageSize {
  width: number;
  height: number;
}

// pdfjs-dist's parser worker and base-14 font metrics are served as static files copied from
// node_modules/pdfjs-dist (`web/public/pdfjs`) rather than resolved via `new URL(...,
// import.meta.url)`, so the path is one plain string under every bundler this app builds with
// (`next build --turbopack` and plain webpack alike). Re-run:
//   cp node_modules/pdfjs-dist/build/pdf.worker.min.mjs web/public/pdfjs/
//   cp node_modules/pdfjs-dist/standard_fonts/* web/public/pdfjs/standard_fonts/
// after bumping the `pdfjs-dist` dependency. The parser runs through `pdf.worker.entry.mjs`,
// which first installs the ES2026 built-ins pdf.js 6's modern build calls but Chromium < 145-class
// browsers lack (`es-polyfills.mjs`; the render worker bundles the same file via ./pdfjs.ts).
// Polyfills, not pdf.js's legacy build: ~2 KB for exactly those three built-ins, versus a legacy
// build ~60 KB larger (plus a larger parser worker) that transpiles and polyfills everything
// for every browser.
const PARSER_WORKER_SRC = "/pdfjs/pdf.worker.entry.mjs";
const STANDARD_FONT_DATA_URL = "/pdfjs/standard_fonts/";

/** Resolves after the frame that paints the current commit. The render worker's chunks (pdf.js
 * is ~150 KB compressed) are requested only once the field list is on screen, so they do not
 * compete with the page's LCP-critical requests for bandwidth. */
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
  const canvasHostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const renderIdRef = useRef(0);
  const [status, setStatus] = useState<PdfCanvasStatus>("loading");
  const [pageSize, setPageSize] = useState<PageSize | null>(null);

  function setStatusReported(next: PdfCanvasStatus) {
    setStatus(next);
    onStatusChange?.(next);
  }

  // Open the document in a fresh worker whenever its URL or auth token changes. The canvas is
  // created here too: control of a canvas can be handed to a worker only once, so every open
  // (including React's dev-mode remount) gets its own canvas element.
  useEffect(() => {
    let cancelled = false;
    let worker: Worker | null = null;
    let canvas: HTMLCanvasElement | null = null;
    setStatusReported("loading");
    setPageSize(null);

    void afterPaint().then(() => {
      const host = canvasHostRef.current;
      if (cancelled || !host) return;
      if (
        typeof OffscreenCanvas === "undefined" ||
        !("transferControlToOffscreen" in HTMLCanvasElement.prototype)
      ) {
        // Every browser the review UI supports has both; say so rather than show a blank pane.
        setStatusReported("error");
        return;
      }
      canvas = document.createElement("canvas");
      canvas.setAttribute("role", "img");
      canvas.className = "border-border rounded-sm border shadow-sm";
      host.append(canvas);
      canvasRef.current = canvas;
      canvas.setAttribute("aria-label", `Invoice document, page ${page + 1}`);

      worker = new Worker(new URL("./pdf-render-worker.ts", import.meta.url), { type: "module" });
      workerRef.current = worker;
      worker.onmessage = (event: MessageEvent<RenderResponse>) => {
        const message = event.data;
        if (cancelled) return;
        if (message.type === "loaded") {
          onDocumentLoad?.(message.pageCount);
          setStatusReported("ready");
        } else if (message.type === "sized" && message.id === renderIdRef.current && canvas) {
          canvas.style.width = `${message.width}px`;
          canvas.style.height = `${message.height}px`;
          setPageSize({ width: message.width, height: message.height });
        } else if (message.type === "error") {
          if (message.stage === "load" || message.id === renderIdRef.current) {
            setStatusReported("error");
          }
        }
      };
      worker.onerror = () => {
        if (!cancelled) setStatusReported("error");
      };
      const offscreen = canvas.transferControlToOffscreen();
      const open: RenderRequest = {
        type: "open",
        canvas: offscreen,
        documentUrl: new URL(documentUrl, window.location.href).href,
        parserWorkerUrl: new URL(PARSER_WORKER_SRC, window.location.href).href,
        standardFontDataUrl: new URL(STANDARD_FONT_DATA_URL, window.location.href).href,
        authToken,
      };
      worker.postMessage(open, [offscreen]);
    });

    return () => {
      cancelled = true;
      if (worker) {
        worker.postMessage({ type: "close" } satisfies RenderRequest);
        // The worker closes itself after cleaning up; terminate if it has not within a second.
        const pending = worker;
        setTimeout(() => pending.terminate(), 1000);
      }
      if (workerRef.current === worker) workerRef.current = null;
      canvas?.remove();
      if (canvasRef.current === canvas) canvasRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documentUrl, authToken]);

  // Render the requested page, fit to the reserved container, whenever the doc or page changes.
  useEffect(() => {
    if (status !== "ready") return;
    const worker = workerRef.current;
    const container = containerRef.current;
    if (!worker || !container) return;
    canvasRef.current?.setAttribute("aria-label", `Invoice document, page ${page + 1}`);
    const id = ++renderIdRef.current;
    worker.postMessage({
      type: "render",
      id,
      page,
      boxWidth: container.clientWidth,
      boxHeight: container.clientHeight,
      pixelRatio: window.devicePixelRatio || 1,
    } satisfies RenderRequest);
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
        {/* React renders no children here: the effect above creates the canvas inside it and
            hands its control to the render worker. */}
        <div ref={canvasHostRef} className="contents" />
        {pageSize && children}
      </div>
    </div>
  );
}
