/** Document pane: pdf.js canvas with the grounded field-box overlay in lockstep (PRD §10 M5.4,
 * web/DESIGN.md "synchronized grounding"). Reserves the A4 aspect ratio before anything loads so
 * there is zero layout shift; when `documentUrl` is null (the http adapter's honest answer until
 * apx/api grows a document endpoint — PRD §10 M5.3 follow-up, no such route exists today) it falls
 * back to the reserved frame with an explicit "not available" state instead of pretending to have
 * a document. */

"use client";

import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, FileWarning } from "lucide-react";
import { Button } from "@/components/ui/button";
import { FieldBoxOverlay, type OverlayBox } from "@/components/invoice/field-box-overlay";
import { PdfCanvas } from "@/components/invoice/pdf-canvas";

export function PdfPane({
  documentUrl,
  authToken,
  pageCount,
  boxes,
  activeField,
  onSelectField,
}: {
  documentUrl: string | null;
  /** Forwarded to `PdfCanvas` as pdf.js's `httpHeaders` bearer token; see its docstring. */
  authToken: string | null;
  /** Metadata page count, used for pagination bounds until the real document is loaded and
   * reports its own (authoritative) `numPages`. */
  pageCount: number;
  boxes: readonly OverlayBox[];
  activeField: string | null;
  onSelectField?: (fieldName: string) => void;
}) {
  const [page, setPage] = useState(0);
  const [loadedPageCount, setLoadedPageCount] = useState<number | null>(null);
  const effectivePageCount = loadedPageCount ?? pageCount;

  useEffect(() => {
    const activeBox = boxes.find((b) => b.fieldName === activeField);
    if (activeBox) setPage(activeBox.box.page);
  }, [activeField, boxes]);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-border flex items-center justify-between border-b px-4 py-2">
        <span className="text-muted-foreground text-xs">
          Page <span className="tabular-mono">{page + 1}</span> of{" "}
          <span className="tabular-mono">{effectivePageCount}</span>
        </span>
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Previous page"
            disabled={page === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
          >
            <ChevronLeft />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Next page"
            disabled={page >= effectivePageCount - 1}
            onClick={() => setPage((p) => Math.min(effectivePageCount - 1, p + 1))}
          >
            <ChevronRight />
          </Button>
        </div>
      </div>
      <div className="bg-muted/40 flex flex-1 items-center justify-center overflow-auto p-6">
        <div
          className="relative flex h-full w-full max-w-[560px] items-center justify-center"
          style={{ aspectRatio: "1 / 1.4142" }}
        >
          {documentUrl ? (
            <PdfCanvas
              documentUrl={documentUrl}
              authToken={authToken}
              page={page}
              onDocumentLoad={setLoadedPageCount}
            >
              <FieldBoxOverlay
                boxes={boxes}
                page={page}
                activeField={activeField}
                onSelectField={onSelectField}
              />
            </PdfCanvas>
          ) : (
            <div
              className="bg-card border-border relative h-full w-full rounded-sm border shadow-sm"
              data-testid="pdf-pane-placeholder"
            >
              <div className="text-muted-foreground/70 absolute inset-0 flex flex-col items-center justify-center gap-2 p-8 text-center">
                <FileWarning className="size-6" aria-hidden />
                <p className="text-xs">
                  No document preview available for this invoice yet. Field boxes below are still
                  positioned from grounded, normalized coordinates.
                </p>
              </div>
              <FieldBoxOverlay
                boxes={boxes}
                page={page}
                activeField={activeField}
                onSelectField={onSelectField}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
