/** The five operating KPIs (PRD §10 M10.3). Each tile says in one line what it counts, so a
 * number never needs a tooltip, and carries a two-minute trend line; values are tabular mono so
 * they don't jitter as they update. */
import type { OpsKpis } from "@/lib/api/types";
import { Sparkline } from "@/components/ops/sparkline";
import type { OpsPoint } from "@/components/ops/use-ops-feed";

function pct(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function seconds(ms: number | null): string {
  return ms === null ? "—" : `${(ms / 1000).toFixed(1)} s`;
}

export function KpiTiles({ kpis, history }: { kpis: OpsKpis; history: readonly OpsPoint[] }) {
  const tiles = [
    {
      id: "stp",
      label: "Straight-through",
      value: pct(kpis.stpRate),
      note: `of ${kpis.routedInvoices} routed invoices went through untouched`,
      trend: history.map((p) => p.stp),
    },
    {
      id: "auto-accept-error",
      label: "Auto-accept error",
      value: pct(kpis.autoAcceptErrorRate),
      note: `of ${kpis.automatedFields} model-filled fields were later corrected`,
      trend: history.map((p) => p.autoAcceptError),
    },
    {
      id: "latency",
      label: "Model latency, median",
      value: seconds(kpis.llmLatencyP50Ms),
      note: `p95 ${seconds(kpis.llmLatencyP95Ms)} across ${kpis.llmCalls} model calls`,
      trend: history.map((p) => p.latencyP50),
    },
    {
      id: "attacks",
      label: "Attacks blocked",
      value: String(kpis.attacksBlocked),
      note: "invoices stopped by the injection scan",
      trend: history.map((p) => p.attacks),
    },
    {
      id: "queue",
      label: "Review queue",
      value: String(kpis.queueSize),
      note: "invoices waiting for a person",
      trend: history.map((p) => p.queue),
    },
  ];
  return (
    // Tiles grow to fill each row, so a short last row never leaves an empty cell; the -1px
    // margins tuck the outer tile borders under the card border.
    <div className="bg-card overflow-hidden rounded-xl border">
      <dl className="-mr-px -mb-px flex flex-wrap">
        {tiles.map((tile) => (
          <div
            key={tile.id}
            className="flex grow basis-36 flex-col border-r border-b px-4 py-4 sm:basis-44 sm:px-5"
            data-testid={`kpi-${tile.id}`}
          >
            <dt className="text-muted-foreground text-xs">{tile.label}</dt>
            <dd className="mt-1.5 font-mono text-2xl leading-tight tabular-nums">{tile.value}</dd>
            <dd className="mt-3" aria-hidden>
              <Sparkline values={tile.trend} className="h-6 w-full opacity-80" />
            </dd>
            <dd className="text-muted-foreground mt-2 text-2xs text-pretty">{tile.note}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
