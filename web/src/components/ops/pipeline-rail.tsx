/** The pipeline drawn as a transit line (PRD §10 M10.3): each processing stage is a station on
 * one track with its live count, invoices in flight ride the segment towards the next station,
 * and the line ends in a terminus bar showing how settled invoices split across outcomes.
 * Horizontal from md up, vertical on phones; tokens are hidden under reduced motion. */
import type { OpsSnapshot } from "@/lib/api/types";
import { STATUS_LABEL } from "@/components/inbox/status-badge";
import { FLOW, OUTCOMES, inFlightCount, stageCounts } from "@/components/ops/stages";
import { cn } from "@/lib/utils";

/** At most this many tokens ride a segment, so a busy stage reads as busy without clutter. */
const MAX_TOKENS = 3;

export function PipelineRail({ stages }: { stages: OpsSnapshot["stages"] }) {
  const count = stageCounts(stages);
  const moving = inFlightCount(stages);
  const settled = OUTCOMES.reduce((sum, o) => sum + (count.get(o.stage) ?? 0), 0);
  return (
    <section aria-labelledby="ops-pipeline" className="bg-card rounded-xl border">
      <header className="flex items-baseline justify-between px-5 pt-4 sm:px-6">
        <h2 id="ops-pipeline" className="text-sm font-semibold">
          Pipeline
        </h2>
        <p className="text-muted-foreground text-xs">
          <span className="text-foreground font-mono tabular-nums">{moving}</span> in flight
        </p>
      </header>

      <ol className="grid grid-cols-1 px-5 pt-4 pb-6 sm:px-6 md:grid-cols-7 md:pt-6">
        {FLOW.map((stage, index) => {
          const n = count.get(stage) ?? 0;
          const active = n > 0;
          const last = index === FLOW.length - 1;
          return (
            <li
              key={stage}
              className="relative flex items-center gap-4 py-2.5 md:flex-col md:gap-2 md:py-0"
              data-testid={`stage-${stage}`}
            >
              {!last && (
                <span
                  aria-hidden
                  className={cn(
                    "absolute rounded-full [clip-path:inset(-4px_0)]",
                    "top-1/2 left-[7px] h-full w-0.5",
                    "md:top-[55px] md:left-1/2 md:h-0.5 md:w-full",
                    active ? "bg-accent/35" : "bg-border",
                  )}
                >
                  {active &&
                    Array.from({ length: Math.min(n, MAX_TOKENS) }, (_, token) => (
                      <span
                        key={token}
                        className="ops-ride absolute inset-0 hidden md:block"
                        style={{ animationDelay: `${(token * 2.1) / MAX_TOKENS}s` }}
                      >
                        <span className="bg-accent absolute top-1/2 right-0 size-1.5 -translate-y-1/2 rounded-full shadow-[0_0_6px_var(--accent)]" />
                      </span>
                    ))}
                </span>
              )}
              <span
                className={cn(
                  "order-last font-mono text-2xl leading-10 tabular-nums md:order-first md:h-10 md:text-3xl",
                  active ? "text-foreground" : "text-muted-foreground",
                )}
              >
                {n}
              </span>
              <span className="relative z-10 grid size-4 shrink-0 place-items-center" aria-hidden>
                {active && <span className="bg-accent/40 ops-ping absolute inset-0 rounded-full" />}
                <span
                  className={cn(
                    "relative size-3.5 rounded-full border-2",
                    active ? "border-accent bg-accent" : "border-border bg-card",
                  )}
                />
              </span>
              <span
                className={cn(
                  "min-w-0 flex-1 text-xs md:flex-none md:text-center md:text-2xs md:tracking-wide md:uppercase",
                  active ? "text-foreground" : "text-muted-foreground",
                )}
              >
                {STATUS_LABEL[stage]}
              </span>
            </li>
          );
        })}
      </ol>

      <div className="border-t px-5 py-4 sm:px-6">
        <div className="flex items-baseline justify-between">
          <p className="text-muted-foreground text-2xs tracking-wide uppercase">Settled</p>
          <p className="text-muted-foreground text-xs">
            <span className="text-foreground font-mono tabular-nums">{settled}</span> invoices
          </p>
        </div>
        <div className="bg-muted mt-2.5 flex h-2 gap-px overflow-hidden rounded-full" aria-hidden>
          {OUTCOMES.map(({ stage, swatch }) => {
            const n = count.get(stage) ?? 0;
            return n > 0 ? (
              <span key={stage} className={cn("h-full", swatch)} style={{ flexGrow: n }} />
            ) : null;
          })}
        </div>
        <ul className="mt-3 grid grid-cols-1 gap-x-8 gap-y-2 sm:grid-cols-3 2xl:grid-cols-6">
          {OUTCOMES.map(({ stage, swatch }) => {
            const n = count.get(stage) ?? 0;
            return (
              <li key={stage} className="flex items-center gap-2 text-sm">
                <span aria-hidden className={cn("size-2 shrink-0 rounded-sm", swatch)} />
                <span className="text-muted-foreground min-w-0 truncate">
                  {STATUS_LABEL[stage]}
                </span>
                <span className="ml-auto font-mono tabular-nums" data-testid={`outcome-${stage}`}>
                  {n}
                </span>
                <span className="text-muted-foreground w-9 text-right font-mono text-2xs tabular-nums">
                  {settled ? `${Math.round((n / settled) * 100)}%` : "—"}
                </span>
              </li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}
