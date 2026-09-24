/** GPU residency (PRD §10 M10.3): which models llama-swap holds in memory right now and for how
 * long. Models load on demand and unload when their stage finishes, so this shows the GPU
 * following the work. State is spelled out in words, never colour alone. */
import type { OpsSnapshot } from "@/lib/api/types";
import { cn } from "@/lib/utils";

export const MODEL_NAMES: Readonly<Record<string, string>> = {
  "extractor-a": "Extractor A",
  "extractor-b": "Extractor B",
  arbiter: "Arbiter",
};

const KNOWN: readonly { model: string; detail: string }[] = [
  { model: "extractor-a", detail: "Qwen3.5-9B · reads every invoice" },
  { model: "extractor-b", detail: "Qwen3.5-4B + LoRA · second opinion" },
  { model: "arbiter", detail: "Gemma 4 12B · settles disagreements" },
];

function held(since: number | undefined, now: number): string | null {
  if (since === undefined) return null;
  const s = Math.max(0, Math.round((now - since) / 1000));
  return s < 60 ? `for ${s}s` : `for ${Math.floor(s / 60)}m`;
}

export function ModelStrip({
  models,
  reachable,
  residentSince,
  now,
}: {
  models: OpsSnapshot["models"];
  reachable: boolean;
  residentSince: ReadonlyMap<string, number>;
  now: number;
}) {
  const state = new Map(models.map((m) => [m.model, m.state]));
  const extra = models
    .filter((m) => !KNOWN.some((k) => k.model === m.model))
    .map((m) => ({ model: m.model, detail: m.model }));
  return (
    <section
      aria-labelledby="ops-models"
      className="bg-card h-full overflow-hidden rounded-xl border"
    >
      <header className="px-5 pt-4 sm:px-6">
        <h2 id="ops-models" className="text-sm font-semibold">
          Models on the GPU
        </h2>
        <p className="text-muted-foreground mt-0.5 text-xs">
          {reachable
            ? "Loaded when a stage needs them, unloaded when it's done."
            : "The model server is not answering; states are unknown."}
        </p>
      </header>
      <ul className="mt-3 divide-y border-t">
        {[...KNOWN, ...extra].map(({ model, detail }) => {
          const current = state.get(model);
          const loaded = reachable && current === "ready";
          const loading = reachable && current !== undefined && current !== "ready";
          const label = !reachable ? "Unknown" : loaded ? "Loaded" : loading ? "Loading" : "Asleep";
          return (
            <li
              key={model}
              className={cn(
                "flex items-center gap-3 px-5 py-3 transition-colors duration-500 sm:px-6",
                loaded && "bg-accent/[0.06]",
              )}
              data-testid={`model-${model}`}
            >
              <span className="relative grid size-2.5 shrink-0 place-items-center" aria-hidden>
                {loaded && <span className="bg-accent/50 ops-ping absolute inset-0 rounded-full" />}
                <span
                  className={cn(
                    "relative size-2.5 rounded-full border",
                    loaded && "border-accent bg-accent",
                    loading && "border-accent bg-accent/30 animate-pulse",
                    !loaded && !loading && "border-muted-foreground/60",
                  )}
                />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium">
                  {MODEL_NAMES[model] ?? model}
                </span>
                <span className="text-muted-foreground block truncate text-xs">{detail}</span>
              </span>
              <span className="text-right">
                <span
                  className={cn(
                    "block text-xs font-medium",
                    loaded ? "text-accent" : "text-muted-foreground",
                  )}
                >
                  {label}
                </span>
                {loaded && (
                  <span className="text-muted-foreground block font-mono text-2xs tabular-nums">
                    {held(residentSince.get(model), now)}
                  </span>
                )}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
