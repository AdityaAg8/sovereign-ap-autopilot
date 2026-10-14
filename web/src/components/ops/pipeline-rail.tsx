/** The pipeline as a stage flow: each processing station in order, with its live count and a bar
 * sized against the busiest station, so where work is piling up reads at a glance. Columns from
 * md up, rows on phones; `SettledBreakdown` shows how finished invoices split across outcomes. */
import type { CSSProperties } from "react";
import { ChevronRight } from "lucide-react";
import type { OpsSnapshot } from "@/lib/api/types";
import { STATUS_LABEL } from "@/components/inbox/status-badge";
import { FLOW, OUTCOMES, inFlightCount, stageCounts } from "@/components/ops/stages";
import { formatCount } from "@/lib/format";
import { cn } from "@/lib/utils";

export function PipelineRail({ stages }: { stages: OpsSnapshot["stages"] }) {
  const count = stageCounts(stages);
  const moving = inFlightCount(stages);
  const max = Math.max(1, ...FLOW.map((stage) => count.get(stage) ?? 0));
  return (
    <section aria-labelledby="ops-pipeline" className="panel flex flex-col">
      {/* The heading stays a direct child of the header, and the header of the section, so the
          stage list is found from the heading's section. */}
      <header className="flex items-baseline justify-between gap-4 px-5 pt-4">
        <h2 id="ops-pipeline" className="text-sm font-semibold">
          Pipeline
        </h2>
        <p className="text-muted-foreground shrink-0 text-xs">
          <span className="tabular-mono text-foreground">{formatCount(moving)}</span> in flight
        </p>
      </header>
      <p className="text-muted-foreground mt-0.5 px-5 text-xs">
        Invoices at each processing station right now
      </p>

      <ol className="grid flex-1 grid-cols-1 content-end gap-y-1 px-5 pt-4 pb-5 md:grid-cols-7 md:gap-x-5">
        {FLOW.map((stage, index) => {
          const n = count.get(stage) ?? 0;
          const busiest = n > 0 && n === max;
          return (
            <li
              key={stage}
              className="relative grid grid-cols-[7.5rem_minmax(0,1fr)_2.5rem] items-center gap-3 md:flex md:flex-col-reverse md:items-stretch md:gap-2"
              data-testid={`stage-${stage}`}
              style={{ "--share": n / max } as CSSProperties}
            >
              <span
                className={cn(
                  "min-w-0 truncate text-xs md:text-center",
                  n > 0 ? "text-foreground" : "text-muted-foreground",
                )}
              >
                {STATUS_LABEL[stage]}
              </span>
              <span
                className={cn(
                  "tabular-mono order-last text-right text-sm md:order-none md:text-center md:text-lg md:font-medium",
                  n > 0 ? "text-foreground" : "text-muted-foreground",
                )}
              >
                {n}
              </span>
              <span
                aria-hidden
                className="bg-muted dark:bg-border/60 md:border-border relative h-2 overflow-hidden rounded-full md:h-16 md:rounded-none md:border-b md:bg-transparent md:dark:bg-transparent"
              >
                {n > 0 && (
                  <span
                    className={cn(
                      "absolute inset-y-0 start-0 rounded-full md:inset-x-[12%] md:inset-y-auto md:bottom-0 md:rounded-none md:rounded-t-[3px]",
                      "w-[calc(var(--share)*100%)] md:h-[max(3px,calc(var(--share)*100%))] md:w-auto",
                      busiest ? "bg-accent" : "bg-accent/40",
                    )}
                  />
                )}
              </span>
              {index < FLOW.length - 1 && (
                <ChevronRight
                  aria-hidden
                  className="text-muted-foreground/50 absolute top-[3.25rem] -right-[18px] z-10 hidden size-4 md:block rtl:rotate-180"
                />
              )}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

export function SettledBreakdown({ stages }: { stages: OpsSnapshot["stages"] }) {
  const count = stageCounts(stages);
  const settled = OUTCOMES.reduce((sum, o) => sum + (count.get(o.stage) ?? 0), 0);
  return (
    <section aria-labelledby="ops-settled" className="panel flex flex-col px-5 pt-4 pb-5">
      <header className="flex items-baseline justify-between gap-4">
        <h2 id="ops-settled" className="text-sm font-semibold">
          Settled
        </h2>
        <p className="text-muted-foreground text-xs">
          <span className="tabular-mono text-foreground">{formatCount(settled)}</span> invoices
        </p>
      </header>
      <div className="bg-muted mt-4 flex h-2.5 gap-0.5 overflow-hidden rounded-full" aria-hidden>
        {OUTCOMES.map(({ stage, swatch }) => {
          const n = count.get(stage) ?? 0;
          return n > 0 ? (
            <span key={stage} className={cn("h-full", swatch)} style={{ flexGrow: n }} />
          ) : null;
        })}
      </div>
      <ul className="mt-4 grid flex-1 content-start gap-y-2.5">
        {OUTCOMES.map(({ stage, swatch }) => {
          const n = count.get(stage) ?? 0;
          return (
            <li key={stage} className="flex items-center gap-2.5 text-sm">
              <span aria-hidden className={cn("size-2 shrink-0 rounded-full", swatch)} />
              <span className="text-muted-foreground min-w-0 truncate">{STATUS_LABEL[stage]}</span>
              <span className="tabular-mono ms-auto" data-testid={`outcome-${stage}`}>
                {n}
              </span>
              <span className="tabular-mono text-muted-foreground w-10 text-right text-xs">
                {settled ? `${Math.round((n / settled) * 100)}%` : "—"}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
