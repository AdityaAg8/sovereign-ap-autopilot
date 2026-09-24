/** The "now" band at the top of the operations view (PRD §10 M10.3): one plain sentence about
 * what the system is doing this second -- invoices moving, the busiest station, which models hold
 * the GPU -- beside a two-minute trace of work in flight. */
import type { OpsSnapshot } from "@/lib/api/types";
import { STATUS_LABEL } from "@/components/inbox/status-badge";
import { Sparkline } from "@/components/ops/sparkline";
import { FLOW, inFlightCount, stageCounts } from "@/components/ops/stages";
import { MODEL_NAMES } from "@/components/ops/model-strip";
import { cn } from "@/lib/utils";

function gpuSentence(snapshot: OpsSnapshot): string {
  if (!snapshot.modelsReachable) return "The model server is not answering.";
  const ready = snapshot.models
    .filter((m) => m.state === "ready")
    .map((m) => MODEL_NAMES[m.model] ?? m.model);
  const loading = snapshot.models
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
    <section
      aria-label="Now"
      className="bg-card relative overflow-hidden rounded-xl border px-5 py-5 sm:px-6"
    >
      <div className="grid gap-5 md:grid-cols-[minmax(0,1fr)_18rem] md:items-end">
        <div className="min-w-0">
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
              <time dateTime={snapshot.generatedAt} className="font-mono">
                {new Date(snapshot.generatedAt).toLocaleTimeString()}
              </time>
            </span>
          </p>
          <p className="mt-3 text-2xl leading-tight font-semibold tracking-tight text-balance sm:text-[1.75rem]">
            {moving === 0 ? (
              "Nothing in flight. The pipeline is clear."
            ) : (
              <>
                <span className="text-accent font-mono tabular-nums">{moving}</span>{" "}
                {moving === 1 ? "invoice" : "invoices"} moving through the pipeline
              </>
            )}
          </p>
          <p className="text-muted-foreground mt-1.5 text-sm text-pretty">
            {busiest && (
              <>
                Busiest station:{" "}
                <span className="text-foreground">{STATUS_LABEL[busiest.stage]}</span>{" "}
                <span className="font-mono tabular-nums">({busiest.n})</span>.{" "}
              </>
            )}
            {gpuSentence(snapshot)}
          </p>
        </div>
        <div>
          <Sparkline
            values={history}
            area
            className="h-14 w-full"
            label={`Invoices in flight over the last two minutes, now ${moving}`}
          />
          <p className="text-muted-foreground mt-1.5 flex justify-between text-2xs">
            <span>2 min ago</span>
            <span>in flight</span>
            <span>now</span>
          </p>
        </div>
      </div>
    </section>
  );
}
