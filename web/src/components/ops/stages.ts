/** Stage vocabulary for the operations view: the ordered processing stations an
 * invoice passes through and the outcomes it settles into, with the token each outcome is drawn in. */
import type { InvoiceStatus, OpsSnapshot } from "@/lib/api/types";

export const FLOW: readonly InvoiceStatus[] = [
  "received",
  "scanned",
  "parsed",
  "extracted",
  "gated",
  "validated",
  "fraud_checked",
];

export const OUTCOMES: readonly { stage: InvoiceStatus; swatch: string }[] = [
  { stage: "awaiting_review", swatch: "bg-warning" },
  { stage: "approved", swatch: "bg-success" },
  { stage: "sent", swatch: "bg-success/55" },
  { stage: "archived", swatch: "bg-muted-foreground/45" },
  { stage: "blocked", swatch: "bg-destructive" },
  { stage: "failed", swatch: "bg-destructive/50" },
];

export function stageCounts(stages: OpsSnapshot["stages"]): Map<InvoiceStatus, number> {
  return new Map(stages.map((s) => [s.stage, s.count]));
}

export function inFlightCount(stages: OpsSnapshot["stages"]): number {
  const count = stageCounts(stages);
  return FLOW.reduce((sum, stage) => sum + (count.get(stage) ?? 0), 0);
}
