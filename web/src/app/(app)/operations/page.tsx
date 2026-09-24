/** Operations view (PRD §10 M10.3): a live dispatch board for the pipeline -- what is moving
 * now, the operating KPIs with their recent trend, the pipeline as a transit line, the models
 * holding the GPU and the latest activity. Data comes from the 2 s polling feed. */

"use client";

import { ActivityTable } from "@/components/ops/activity-table";
import { KpiTiles } from "@/components/ops/kpi-tiles";
import { ModelStrip } from "@/components/ops/model-strip";
import { NowStrip } from "@/components/ops/now-strip";
import { PipelineRail } from "@/components/ops/pipeline-rail";
import { useOpsFeed } from "@/components/ops/use-ops-feed";
import { Skeleton } from "@/components/ui/skeleton";
import { useSetPageHeader } from "@/lib/shell/page-header-context";

export default function OperationsPage() {
  useSetPageHeader({ title: "Operations" });
  const { snapshot, error, now, history, fresh, residentSince } = useOpsFeed();

  if (!snapshot) {
    return error ? (
      <p className="text-muted-foreground p-6 text-sm">{error}</p>
    ) : (
      <div className="mx-auto max-w-[90rem] space-y-4 p-4 sm:p-6" aria-busy="true">
        <Skeleton className="h-36 w-full rounded-xl" />
        <Skeleton className="h-32 w-full rounded-xl" />
        <Skeleton className="h-64 w-full rounded-xl" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-[90rem] space-y-4 p-4 sm:p-6">
      <NowStrip snapshot={snapshot} history={history.map((p) => p.inFlight)} error={error} />
      <KpiTiles kpis={snapshot.kpis} history={history} />
      <PipelineRail stages={snapshot.stages} />
      <div className="grid grid-cols-[minmax(0,1fr)] gap-4 xl:grid-cols-[minmax(0,1fr)_24rem]">
        <div className="xl:order-last">
          <ModelStrip
            models={snapshot.models}
            reachable={snapshot.modelsReachable}
            residentSince={residentSince}
            now={now}
          />
        </div>
        <ActivityTable rows={snapshot.recent} now={now} fresh={fresh} />
      </div>
    </div>
  );
}
