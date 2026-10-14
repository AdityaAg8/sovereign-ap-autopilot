/** Fraud/validation flags for the current invoice, each with a plain-language explanation
 * (the explanation itself is required, surfaced here ahead of apx/fraud landing). Severity pairs an icon and a word with color, never color alone. */
import { AlertOctagon, AlertTriangle, Info } from "lucide-react";
import type { Flag } from "@/lib/api/types";
import { cn } from "@/lib/utils";

const SEVERITY_CONFIG: Record<
  Flag["severity"],
  { icon: typeof AlertOctagon; label: string; className: string }
> = {
  high: { icon: AlertOctagon, label: "High", className: "text-destructive" },
  medium: { icon: AlertTriangle, label: "Medium", className: "text-warning" },
  low: { icon: Info, label: "Low", className: "text-muted-foreground" },
};

export function FlagsPanel({ flags }: { flags: readonly Flag[] }) {
  if (flags.length === 0) {
    return (
      <p className="text-muted-foreground p-4 text-sm">
        No fraud or validation flags on this invoice.
      </p>
    );
  }
  return (
    <ul className="divide-border divide-y">
      {flags.map((flag) => {
        const { icon: Icon, label, className } = SEVERITY_CONFIG[flag.severity];
        return (
          <li key={flag.id} className="flex items-start gap-2.5 px-4 py-3">
            <Icon className={cn("mt-0.5 size-4 shrink-0", className)} aria-hidden />
            <div className="space-y-0.5">
              <p className="text-sm font-medium first-letter:uppercase">
                {flag.rule.replaceAll("_", " ")}{" "}
                <span className={cn("text-xs font-normal", className)}>· {label}</span>
              </p>
              <p className="text-muted-foreground text-sm">{flag.explanation}</p>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
