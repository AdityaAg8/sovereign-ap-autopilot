/** Tabular confidence readout with a thin proportional bar, the same visual on the inbox and the
 * invoice field list so a reviewer recognizes "confidence" instantly in both places. High
 * confidence draws in the accent; only low values take a warning or danger tone, and the percent
 * always states the value, so color is never the only signal. */
import { confidenceTone, formatConfidence } from "@/lib/format";
import { cn } from "@/lib/utils";

export function ConfidenceMeter({
  value,
  className,
}: {
  value: number | null;
  className?: string;
}) {
  const tone = confidenceTone(value);
  const pct = value === null ? 0 : Math.round(value * 100);
  return (
    <div
      className={cn("flex items-center gap-2", className)}
      title={value === null ? "No confidence data" : `${pct}% confident`}
    >
      <span
        className={cn(
          "tabular-mono w-9 text-right text-xs",
          tone === "warning" && "text-warning",
          tone === "destructive" && "text-destructive",
        )}
      >
        {formatConfidence(value)}
      </span>
      <span className="bg-muted dark:bg-border h-1 w-12 overflow-hidden rounded-full" aria-hidden>
        <span
          className={cn(
            "block h-full rounded-full",
            tone === "success" && "bg-accent",
            tone === "warning" && "bg-warning",
            tone === "destructive" && "bg-destructive",
          )}
          style={{ width: `${pct}%` }}
        />
      </span>
    </div>
  );
}
