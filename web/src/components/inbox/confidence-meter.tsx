/** Tabular confidence readout with a small proportional bar — the same visual language used on
 * the invoice-detail field list, so a reviewer recognizes "confidence" instantly in both places
 * (web/DESIGN.md signature: synchronized grounding). Color is never the only signal: the percent
 * and the bar both encode the value. */
import { confidenceTone, formatConfidence } from "@/lib/format";
import { cn } from "@/lib/utils";

export function ConfidenceMeter({ value }: { value: number | null }) {
  const tone = confidenceTone(value);
  const pct = value === null ? 0 : Math.round(value * 100);
  return (
    <div
      className="flex items-center gap-2"
      title={value === null ? "No confidence data" : `${pct}% confident`}
    >
      <span className="tabular-mono w-9 text-right text-xs">{formatConfidence(value)}</span>
      <span className="bg-muted h-1.5 w-12 overflow-hidden rounded-full" aria-hidden>
        <span
          className={cn(
            "block h-full rounded-full transition-[width]",
            tone === "success" && "bg-success",
            tone === "warning" && "bg-warning",
            tone === "destructive" && "bg-destructive",
          )}
          style={{ width: `${pct}%` }}
        />
      </span>
    </div>
  );
}
