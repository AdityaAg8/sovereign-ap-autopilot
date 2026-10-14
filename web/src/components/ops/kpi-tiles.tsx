/** The five operating KPIs as a row of cards. Each says in one line what it counts, so a number
 * never needs a tooltip, and draws only what the snapshot can back: a share bar for rates, the
 * median against the p95 for latency. Values are tabular so they never jitter as they update. */
import type { OpsKpis } from "@/lib/api/types";
import { formatCount } from "@/lib/format";
import { cn } from "@/lib/utils";

function pct(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function seconds(ms: number | null): string {
  return ms === null ? "—" : `${(ms / 1000).toFixed(1)} s`;
}

/** A thin share bar; `value` is 0..1, clamped. */
function Meter({ value, tone = "accent" }: { value: number | null; tone?: "accent" | "warning" }) {
  const width = value === null ? 0 : Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="bg-muted dark:bg-border h-1.5 overflow-hidden rounded-full" aria-hidden>
      <div
        className={cn("h-full rounded-full", tone === "accent" ? "bg-accent" : "bg-warning")}
        style={{ width: `${width}%` }}
      />
    </div>
  );
}

/** Median latency drawn as a share of the p95, with both ends labelled. */
function LatencyRange({ p50, p95 }: { p50: number | null; p95: number | null }) {
  const share = p50 !== null && p95 ? p50 / p95 : null;
  return (
    <div aria-hidden>
      <Meter value={share} />
      <div className="text-muted-foreground mt-1.5 flex justify-between text-2xs">
        <span>median</span>
        <span>
          p95 <span className="tabular-mono">{seconds(p95)}</span>
        </span>
      </div>
    </div>
  );
}

export function KpiTiles({ kpis }: { kpis: OpsKpis }) {
  const tiles = [
    {
      id: "stp",
      label: "Straight-through",
      value: pct(kpis.stpRate),
      note: `of ${formatCount(kpis.routedInvoices)} routed invoices went through untouched`,
      visual: <Meter value={kpis.stpRate} />,
    },
    {
      id: "auto-accept-error",
      label: "Auto-accept error",
      value: pct(kpis.autoAcceptErrorRate),
      note: `of ${formatCount(kpis.automatedFields)} model-filled fields were later corrected`,
      visual: <Meter value={kpis.autoAcceptErrorRate} tone="warning" />,
    },
    {
      id: "latency",
      label: "Model latency, median",
      value: seconds(kpis.llmLatencyP50Ms),
      note: `across ${formatCount(kpis.llmCalls)} model calls`,
      visual: <LatencyRange p50={kpis.llmLatencyP50Ms} p95={kpis.llmLatencyP95Ms} />,
    },
    {
      id: "attacks",
      label: "Attacks blocked",
      value: formatCount(kpis.attacksBlocked),
      note: "invoices stopped by the injection scan",
      visual: null,
    },
    {
      id: "queue",
      label: "Review queue",
      value: formatCount(kpis.queueSize),
      note: kpis.routedInvoices
        ? `waiting for a person, ${Math.round((kpis.queueSize / kpis.routedInvoices) * 100)}% of routed invoices`
        : "invoices waiting for a person",
      visual: kpis.routedInvoices ? (
        <Meter value={kpis.queueSize / kpis.routedInvoices} tone="warning" />
      ) : null,
    },
  ];
  return (
    <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-5">
      {tiles.map((tile) => (
        <div
          key={tile.id}
          className="panel flex flex-col px-4 py-3.5 last:col-span-2 sm:last:col-span-1"
          data-testid={`kpi-${tile.id}`}
        >
          <dt className="text-muted-foreground text-xs font-medium">{tile.label}</dt>
          <dd className="mt-1 text-[1.625rem] leading-tight font-semibold tracking-[-0.02em] tabular-nums">
            {tile.value}
          </dd>
          {tile.visual && <dd className="mt-2.5">{tile.visual}</dd>}
          <dd className="text-muted-foreground mt-auto pt-2.5 text-2xs text-pretty">{tile.note}</dd>
        </div>
      ))}
    </dl>
  );
}
