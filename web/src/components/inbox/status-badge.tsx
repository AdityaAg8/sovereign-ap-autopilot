/** Status and route badges for the inbox/review tables. Color always pairs with text so the
 * signal never depends on hue alone (WCAG 2.1 SC 1.4.1). */
import { Ban, CircleCheck, Clock, Send, ShieldAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
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

export function StatusBadge({ status }: { status: InvoiceStatus }) {
  return (
    <Badge variant="outline" className="whitespace-nowrap">
      {STATUS_LABEL[status]}
    </Badge>
  );
}

const ROUTE_CONFIG: Record<
  RouteDecision,
  { label: string; icon: typeof CircleCheck; className: string }
> = {
  auto: { label: "Auto", icon: CircleCheck, className: "bg-success/20 text-success" },
  review: { label: "Review", icon: Clock, className: "bg-warning/20 text-warning" },
  block: { label: "Blocked", icon: Ban, className: "bg-destructive/20 text-destructive" },
};

export function RouteBadge({ route }: { route: RouteDecision | null }) {
  if (!route) {
    return (
      <Badge variant="secondary" className="gap-1">
        <Send className="size-3" aria-hidden /> Pending
      </Badge>
    );
  }
  const { label, icon: Icon, className } = ROUTE_CONFIG[route];
  return (
    <Badge variant="secondary" className={cn("gap-1 border-transparent", className)}>
      <Icon className="size-3" aria-hidden /> {label}
    </Badge>
  );
}

export function FlagCountBadge({ count }: { count: number }) {
  if (count === 0) return <span className="text-muted-foreground text-xs">—</span>;
  return (
    <Badge
      variant="secondary"
      className="bg-destructive/20 text-destructive gap-1 border-transparent"
    >
      <ShieldAlert className="size-3" aria-hidden />
      {count}
    </Badge>
  );
}
