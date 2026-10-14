/** The "now" band at the top of the operations view: one plain sentence about what the system is
 * doing this second (invoices moving, the busiest station, which models hold the GPU) beside a
 * trace of work in flight over the polling history this page has seen, up to two minutes. */
import type { OpsSnapshot } from "@/lib/api/types";
import { STATUS_LABEL } from "@/components/inbox/status-badge";
import { Sparkline } from "@/components/ops/sparkline";
import { FLOW, inFlightCount, stageCounts } from "@/components/ops/stages";
import { MODEL_NAMES, pipelineModels } from "@/components/ops/model-strip";
import { formatCount } from "@/lib/format";
import { cn } from "@/lib/utils";

function gpuSentence(snapshot: OpsSnapshot): string {
  if (!snapshot.modelsReachable) return "The model server is not answering.";
  const ready = pipelineModels(snapshot.models)
    .filter((m) => m.state === "ready")
    .map((m) => MODEL_NAMES[m.model] ?? m.model);
  const loading = pipelineModels(snapshot.models)
    .filter((m) => m.state !== "ready")
    .map((m) => MODEL_NAMES[m.model] ?? m.model);
  if (ready.length === 0 && loading.length === 0) return "GPU idle. Models load when work arrives.";
  const parts = [];
  if (ready.length)
    parts.push(`${ready.join(" and ")} ${ready.length === 1 ? "is" : "are"} on the GPU`);
  if (loading.length) parts.push(`${loading.join(" and ")} loading`);
  return `${parts.join(", ")}.`;
}

export function NowStrip({
  snapshot,
  history,
  error,
}: {
  snapshot: OpsSnapshot;
  history: readonly number[];
  error: string | null;
}) {
  const count = stageCounts(snapshot.stages);
  const moving = inFlightCount(snapshot.stages);
  const busiest = FLOW.reduce<{ stage: (typeof FLOW)[number]; n: number } | null>((best, stage) => {
    const n = count.get(stage) ?? 0;
    return n > 0 && (best === null || n > best.n) ? { stage, n } : best;
  }, null);

  return (
    <section aria-label="Now" className="panel overflow-hidden">
      <div className="grid md:grid-cols-[minmax(0,1fr)_20rem]">
        <div className="min-w-0 px-5 py-4">
          <p className="text-muted-foreground flex items-center gap-2 text-xs">
            <span className="relative flex size-2" aria-hidden>
              {!error && <span className="bg-success ops-ping absolute inset-0 rounded-full" />}
              <span
                className={cn(
                  "relative size-2 rounded-full",
                  error ? "bg-destructive" : "bg-success",
                )}
              />
            </span>
            {error ? `Paused: ${error}` : "Live"}
            <span aria-hidden>·</span>
            <span>
              updated{" "}
              <time dateTime={snapshot.generatedAt} className="tabular-mono">
                {new Date(snapshot.generatedAt).toLocaleTimeString()}
              </time>
            </span>
          </p>
          <p className="mt-2 text-xl leading-tight font-semibold tracking-[-0.02em] text-balance">
            {moving === 0 ? (
              "Nothing in flight. The pipeline is clear."
            ) : (
              <>
                <span className="text-accent tabular-nums">{formatCount(moving)}</span>{" "}
                {moving === 1 ? "invoice" : "invoices"} moving through the pipeline
              </>
            )}
          </p>
          <p className="text-muted-foreground mt-1 text-sm text-pretty">
            {busiest && (
              <>
                Busiest station:{" "}
                <span className="text-foreground">{STATUS_LABEL[busiest.stage]}</span>{" "}
                <span className="tabular-mono">({busiest.n})</span>.{" "}
              </>
            )}
            {gpuSentence(snapshot)}
          </p>
        </div>
        <div className="border-border bg-muted/30 dark:bg-muted/20 flex flex-col justify-end border-t px-5 py-4 md:border-t-0 md:border-s">
          {history.length >= 2 ? (
            <Sparkline
              values={history}
              area
              className="h-12 w-full"
              label={`Invoices in flight over the recent updates, now ${moving}`}
            />
          ) : (
            <div className="text-muted-foreground flex h-12 items-center text-xs">
              The trace starts with the next update.
            </div>
          )}
          <p className="text-muted-foreground mt-1.5 flex justify-between text-2xs">
            <span>In flight, up to 2 min back</span>
            <span>now</span>
          </p>
        </div>
      </div>
    </section>
  );
}
