/** Live feed behind the operations view (PRD §10 M10.3): polls the ops snapshot every 2 s while
 * the tab is visible (never stacking requests) and derives, per tick, a short rolling history for
 * the sparklines, which activity rows just changed, and since when each model has been resident. */

"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError, getApiClient } from "@/lib/api/client";
import type { OpsSnapshot } from "@/lib/api/types";
import { inFlightCount } from "@/components/ops/stages";

export const POLL_MS = 2000;
/** 60 ticks at 2 s: the sparklines cover the last two minutes. */
const HISTORY_POINTS = 60;

export interface OpsPoint {
  inFlight: number;
  stp: number | null;
  autoAcceptError: number | null;
  latencyP50: number | null;
  attacks: number;
  queue: number;
}

export interface OpsFeed {
  snapshot: OpsSnapshot | null;
  error: string | null;
  now: number;
  history: readonly OpsPoint[];
  /** Ids of activity rows that are new or moved since the previous tick. */
  fresh: ReadonlySet<string>;
  /** Model name -> epoch ms it was first seen ready, for models loaded right now. */
  residentSince: ReadonlyMap<string, number>;
}

const EMPTY: OpsFeed = {
  snapshot: null,
  error: null,
  now: 0,
  history: [],
  fresh: new Set(),
  residentSince: new Map(),
};

export function useOpsFeed(): OpsFeed {
  const [feed, setFeed] = useState<OpsFeed>(EMPTY);
  const latest = useRef<OpsFeed>(EMPTY);
  const inFlight = useRef(false);

  useEffect(() => {
    let cancelled = false;
    const publish = (next: OpsFeed) => {
      latest.current = next;
      setFeed(next);
    };
    const load = async (force = false) => {
      if (inFlight.current || (document.hidden && !force)) return;
      inFlight.current = true;
      try {
        const next = await getApiClient().getOpsSnapshot();
        if (cancelled) return;
        const prev = latest.current;
        const now = Date.now();
        const before = new Map(prev.snapshot?.recent.map((r) => [r.id, r.updatedAt]) ?? []);
        const fresh = new Set(
          prev.snapshot === null
            ? []
            : next.recent.filter((r) => before.get(r.id) !== r.updatedAt).map((r) => r.id),
        );
        const residentSince = new Map<string, number>();
        for (const model of next.models) {
          if (model.state === "ready") {
            residentSince.set(model.model, prev.residentSince.get(model.model) ?? now);
          }
        }
        const point: OpsPoint = {
          inFlight: inFlightCount(next.stages),
          stp: next.kpis.stpRate,
          autoAcceptError: next.kpis.autoAcceptErrorRate,
          latencyP50: next.kpis.llmLatencyP50Ms,
          attacks: next.kpis.attacksBlocked,
          queue: next.kpis.queueSize,
        };
        publish({
          snapshot: next,
          error: null,
          now,
          history: [...prev.history, point].slice(-HISTORY_POINTS),
          fresh,
          residentSince,
        });
      } catch (err: unknown) {
        if (!cancelled) {
          publish({
            ...latest.current,
            error: err instanceof ApiError ? err.message : "Could not reach the API.",
          });
        }
      } finally {
        inFlight.current = false;
      }
    };
    void load(true);
    const timer = window.setInterval(() => void load(), POLL_MS);
    const onVisible = () => void load();
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  return feed;
}
