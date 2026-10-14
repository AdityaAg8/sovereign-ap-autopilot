/** An invoice's audit trail: the hash-chained events that decided it, newest
 * first, with their system facts and the exception agent's review note. The event a citation
 * pointed at is scrolled into view and marked, so "why was this blocked?" lands on its evidence. */

"use client";

import { useEffect, useRef, useState } from "react";
import { CitedText } from "@/components/assistant/citation-chip";
import { Skeleton } from "@/components/ui/skeleton";
import { getApiClient } from "@/lib/api/client";
import { ApiError } from "@/lib/api/contract";
import type { AuditEventView, CitationSource } from "@/lib/api/types";
import { auditActionLabel } from "@/lib/audit-actions";
import { formatAuditTime } from "@/lib/format";
import { cn } from "@/lib/utils";

/** Fact keys are snake_case system names ("max_score"); shown as plain words. */
function factLabel(key: string): string {
  return key.replace(/_/g, " ");
}

export function AuditTrail({ invoiceId, focusId }: { invoiceId: string; focusId: number | null }) {
  const [events, setEvents] = useState<readonly AuditEventView[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const focusRef = useRef<HTMLLIElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    getApiClient()
      .getInvoiceAudit(invoiceId)
      .then((rows) => !cancelled && setEvents(rows))
      .catch((err: unknown) => {
        if (!cancelled)
          setError(err instanceof ApiError ? err.message : "Could not load the trail.");
      });
    return () => {
      cancelled = true;
    };
  }, [invoiceId]);

  useEffect(() => {
    if (events && focusId !== null && focusRef.current) {
      focusRef.current.scrollIntoView({ block: "center" });
      focusRef.current.focus({ preventScroll: true });
    }
  }, [events, focusId]);

  if (error) return <p className="text-muted-foreground px-4 py-3 text-sm">{error}</p>;
  if (!events) {
    return (
      <div className="space-y-3 px-4 py-3" aria-busy="true">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
    );
  }
  if (events.length === 0) {
    return <p className="text-muted-foreground px-4 py-3 text-sm">No audit events yet.</p>;
  }

  const hrefFor = (source: CitationSource, ref: string) =>
    source === "audit" ? `/invoices/${invoiceId}?tab=audit&event=${ref}` : null;

  return (
    <ol className="relative px-4 py-3" aria-label="Audit trail, newest first">
      {events.map((event) => {
        const focused = event.id === focusId;
        const facts = Object.entries(event.facts).filter(([, value]) => value !== null);
        return (
          <li
            key={event.id}
            id={`audit-${event.id}`}
            ref={focused ? focusRef : undefined}
            tabIndex={focused ? -1 : undefined}
            aria-current={focused ? "true" : undefined}
            className={cn(
              "relative border-s ps-4 pb-4 outline-none last:pb-0",
              focused && "ops-fresh",
            )}
            data-testid={`audit-event-${event.id}`}
          >
            <span
              aria-hidden
              className={cn(
                "absolute top-1 -start-[5px] size-2.5 rounded-full border-2",
                focused ? "border-accent bg-accent" : "border-border bg-card",
              )}
            />
            <div
              className={cn(
                "rounded-md px-2 py-1.5 -mx-2",
                focused && "ring-accent/60 bg-accent/5 ring-1",
              )}
            >
              <p className="flex flex-wrap items-baseline gap-x-2 text-sm">
                <span className="font-medium" data-testid="audit-label" title={event.action}>
                  {auditActionLabel(event.action)}
                </span>
                <span className="text-muted-foreground tabular-mono text-[0.6875rem]">
                  #{event.id}
                </span>
              </p>
              <p className="text-muted-foreground text-xs">
                <span className="tabular-mono">{formatAuditTime(event.ts)}</span> · {event.actor}
              </p>
              {facts.length > 0 && (
                <dl className="mt-1.5 flex flex-wrap gap-1.5">
                  {facts.map(([key, value]) => (
                    <div
                      key={key}
                      className="bg-muted flex gap-1 rounded px-1.5 py-0.5 text-[0.6875rem]"
                    >
                      <dt className="text-muted-foreground">{factLabel(key)}</dt>
                      <dd className={cn(typeof value === "number" && "tabular-mono")}>
                        {String(value)}
                      </dd>
                    </div>
                  ))}
                </dl>
              )}
              {event.note && (
                <p className="mt-1.5 text-sm leading-relaxed text-pretty">
                  <CitedText text={event.note} hrefFor={hrefFor} />
                </p>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
