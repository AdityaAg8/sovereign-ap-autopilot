/** Draws grounded field boxes over the document pane from normalized bboxes (0..1, origin
 * top-left — apx.common.types.BBox, I1). The active field's box gets the accent
 * highlight in lockstep with its row in the field list (web/DESIGN.md's "synchronized grounding"
 * signature); every other box stays a quiet hairline so the active one reads instantly. Works
 * against any pane that reports its own pixel size, so pdf.js can replace the placeholder pane
 * without this component changing. */
"use client";

import { useState } from "react";
import type { BBox } from "@/lib/api/types";
import { fieldLabel } from "@/lib/invoice-fields";
import { cn } from "@/lib/utils";

export interface OverlayBox {
  fieldName: string;
  box: BBox;
  accepted: boolean;
}

export function FieldBoxOverlay({
  boxes,
  page,
  activeField,
  onSelectField,
}: {
  boxes: readonly OverlayBox[];
  page: number;
  activeField: string | null;
  onSelectField?: (fieldName: string) => void;
}) {
  const [hovered, setHovered] = useState<string | null>(null);
  const visible = boxes.filter((b) => b.box.page === page);

  return (
    // Pointer-only affordance: each box duplicates its row in the field list, which is the
    // keyboard and screen-reader path (full-size targets, same select action). Keeping the boxes
    // out of the tab order and accessibility tree avoids a second, tiny set of tab stops over a
    // scaled-down page — WCAG 2.5.8's "equivalent control" exception covers their size.
    <div className="pointer-events-none absolute inset-0" aria-hidden>
      {visible.map(({ fieldName, box, accepted }) => {
        const active = fieldName === activeField || fieldName === hovered;
        return (
          <button
            key={fieldName}
            type="button"
            tabIndex={-1}
            title={fieldLabel(fieldName)}
            onClick={() => onSelectField?.(fieldName)}
            onMouseEnter={() => setHovered(fieldName)}
            onMouseLeave={() => setHovered((h) => (h === fieldName ? null : h))}
            className={cn(
              "pointer-events-auto absolute rounded-[3px] border transition-[border-color,background-color,box-shadow] duration-(--duration-base) ease-out",
              active
                ? "border-accent bg-accent/10 ring-accent ring-1"
                : accepted
                  ? "hover:border-accent/60 border-transparent bg-transparent"
                  : "border-warning/70 hover:border-accent/60 bg-warning/[0.06] border-dashed",
            )}
            style={{
              left: `${box.x0 * 100}%`,
              top: `${box.y0 * 100}%`,
              width: `${(box.x1 - box.x0) * 100}%`,
              height: `${(box.y1 - box.y0) * 100}%`,
            }}
          >
            <span className="sr-only">{fieldLabel(fieldName)}</span>
          </button>
        );
      })}
    </div>
  );
}
