/** Status, route and flag-count markers for the inbox, review, vendor and operations tables. A
 * status is a small colored dot plus its label, a route an icon plus its label, so color never
 * carries the signal alone (WCAG 2.1 SC 1.4.1) and rows stay quiet enough to scan. */
import { Ban, CircleCheck, CircleDashed, Clock, ShieldAlert } from "lucide-react";
import type { InvoiceStatus, RouteDecision } from "@/lib/api/types";
import { cn } from "@/lib/utils";

export const STATUS_LABEL: Record<InvoiceStatus, string> = {
  received: "Received",
  scanned: "Scanned",
  parsed: "Parsed",
  extracted: "Extracted",
  gated: "Gated",
  validated: "Validated",
  fraud_checked: "Fraud-checked",
  awaiting_review: "Awaiting review",
  approved: "Approved",
  sent: "Sent",
  archived: "Archived",
  blocked: "Blocked",
  failed: "Failed",
};

/** Dot color per status: in-flight stages share the accent, outcomes use the semantic tones. */
export const STATUS_DOT: Record<InvoiceStatus, string> = {
  received: "bg-accent/60",
  scanned: "bg-accent/60",
  parsed: "bg-accent/60",
  extracted: "bg-accent/60",
  gated: "bg-accent/60",
  validated: "bg-accent/60",
  fraud_checked: "bg-accent/60",
  awaiting_review: "bg-warning",
  approved: "bg-success",
  sent: "bg-success/60",
  archived: "bg-muted-foreground/50",
  blocked: "bg-destructive",
  failed: "bg-destructive/60",
};

export function StatusBadge({ status, className }: { status: InvoiceStatus; className?: string }) {
  return (
    <span
      className={cn("inline-flex items-center gap-1.5 text-xs whitespace-nowrap", className)}
      data-slot="status"
    >
      <span aria-hidden className={cn("size-1.5 shrink-0 rounded-full", STATUS_DOT[status])} />
      <span>{STATUS_LABEL[status]}</span>
    </span>
  );
}

const ROUTE_CONFIG: Record<
  RouteDecision,
  { label: string; icon: typeof CircleCheck; className: string }
> = {
  auto: { label: "Auto", icon: CircleCheck, className: "text-success" },
  review: { label: "Review", icon: Clock, className: "text-warning" },
  block: { label: "Blocked", icon: Ban, className: "text-destructive" },
};

export function RouteBadge({ route }: { route: RouteDecision | null }) {
  const {
    label,
    icon: Icon,
    className,
  } = route
    ? ROUTE_CONFIG[route]
    : { label: "Pending", icon: CircleDashed, className: "text-muted-foreground" };
  return (
    <span className="text-muted-foreground inline-flex items-center gap-1.5 text-xs whitespace-nowrap">
      <Icon className={cn("size-3.5 shrink-0", className)} aria-hidden /> {label}
    </span>
  );
}

export function FlagCountBadge({ count }: { count: number }) {
  if (count === 0) return <span className="text-muted-foreground/70 text-xs">—</span>;
  return (
    <span
      className="bg-destructive/10 text-destructive dark:bg-destructive/15 inline-flex h-5 items-center gap-1 rounded-md px-1.5 text-2xs font-semibold"
      title={`${count} ${count === 1 ? "flag" : "flags"}`}
    >
      <ShieldAlert className="size-3" aria-hidden />
      <span className="tabular-mono">{count}</span>
      <span className="sr-only">{count === 1 ? "flag" : "flags"}</span>
    </span>
  );
}
