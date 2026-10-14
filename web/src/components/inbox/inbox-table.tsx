/** Virtualized inbox table: invoice number, vendor, status, amount, confidence and flags for every
 * invoice, with j/k/enter keyboard navigation; columns drop out by container width
 * (globals.css `.inbox-grid`). Uses @tanstack/react-virtual so the DOM only ever
 * holds the visible rows regardless of dataset size. While the rows load, the filter bar and
 * column labels already render above row-height skeletons, so the page paints its real layout
 * first and the rows then fill it without shifting anything. */

"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  StatusBadge,
  RouteBadge,
  FlagCountBadge,
  STATUS_LABEL,
} from "@/components/inbox/status-badge";
import { ConfidenceMeter } from "@/components/inbox/confidence-meter";
import { InboxFilters, type InboxFilterState } from "@/components/inbox/inbox-filters";
import { useHotkeys } from "@/hooks/use-hotkeys";
import { useRegisterShortcuts } from "@/lib/shortcuts/registry";
import { formatAed, formatReceivedCompact } from "@/lib/format";
import type { InvoiceStatus, InvoiceSummary } from "@/lib/api/types";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

const ROW_HEIGHT = 44;

const SHORTCUTS = [
  { combo: "j", description: "Next invoice" },
  { combo: "k", description: "Previous invoice" },
  { combo: "Enter", description: "Open the highlighted invoice" },
] as const;

/** The supplier's invoice number, or, when none is known yet (the http adapter until apx/api
 * carries it), this system's own ID: muted and announced as internal, so it is never mistaken
 * for the supplier's number yet every row still names a searchable invoice. */
function InvoiceRef({ row }: { row: InvoiceSummary }) {
  if (row.invoiceNumber) return <>{row.invoiceNumber}</>;
  return (
    <span className="text-muted-foreground" data-testid="inbox-internal-id">
      <span className="sr-only">Internal ID </span>
      {row.id}
    </span>
  );
}

export function InboxTable({
  invoices,
  lockStatus,
  shortcutScope = "Inbox",
}: {
  /** `null` while the list is still loading. */
  invoices: readonly InvoiceSummary[] | null;
  /** Review queue passes "awaiting_review" so the list is pre-filtered and the picker is hidden. */
  lockStatus?: InvoiceStatus;
  shortcutScope?: string;
}) {
  const router = useRouter();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [filters, setFilters] = useState<InboxFilterState>({
    query: "",
    status: lockStatus ?? "all",
    route: "all",
  });
  const [activeIndex, setActiveIndex] = useState(0);

  useRegisterShortcuts(shortcutScope, SHORTCUTS);

  const rows = useMemo(() => {
    if (!invoices) return [];
    const q = filters.query.trim().toLowerCase();
    return invoices.filter((row) => {
      if (lockStatus && row.status !== lockStatus) return false;
      if (!lockStatus && filters.status !== "all" && row.status !== filters.status) return false;
      if (filters.route !== "all" && row.route !== filters.route) return false;
      if (
        q &&
        !row.id.toLowerCase().includes(q) &&
        !(row.invoiceNumber ?? "").toLowerCase().includes(q) &&
        !(row.vendorName ?? "").toLowerCase().includes(q)
      )
        return false;
      return true;
    });
  }, [invoices, filters, lockStatus]);

  useEffect(() => {
    setActiveIndex((i) => Math.min(i, Math.max(rows.length - 1, 0)));
  }, [rows.length]);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 12,
  });

  useHotkeys([
    {
      combo: "j",
      description: "Next invoice",
      handler: () =>
        setActiveIndex((i) => {
          const next = Math.min(i + 1, rows.length - 1);
          virtualizer.scrollToIndex(next, { align: "auto" });
          return next;
        }),
    },
    {
      combo: "k",
      description: "Previous invoice",
      handler: () =>
        setActiveIndex((i) => {
          const next = Math.max(i - 1, 0);
          virtualizer.scrollToIndex(next, { align: "auto" });
          return next;
        }),
    },
    {
      combo: "enter",
      description: "Open the highlighted invoice",
      handler: () => {
        const row = rows[activeIndex];
        if (row) router.push(`/invoices/${row.id}`);
      },
    },
  ]);

  const activeId = rows[activeIndex]?.id;

  return (
    <div className="inbox-table flex h-full min-h-0 flex-col">
      <InboxFilters value={filters} onChange={setFilters} hideStatus={Boolean(lockStatus)} />
      {/* Visual column labels only — the interactive rows below are an ARIA listbox (each row's
          own accessible name carries its data), so this row stays out of the accessibility tree
          rather than presenting an orphaned `role="row"` with no table/grid ancestor. */}
      <div
        aria-hidden="true"
        className="inbox-grid text-muted-foreground border-border border-y px-6 py-2 text-xs font-medium tracking-wide uppercase"
        data-testid="inbox-header"
      >
        <span className="ic-num">Invoice no.</span>
        <span className="ic-vendor">Vendor</span>
        <span className="ic-status">Status</span>
        <span className="ic-route">Route</span>
        <span className="ic-amount text-right">Amount</span>
        <span className="ic-conf">Confidence</span>
        <span className="ic-flags text-center">Flags</span>
        <span className="ic-received text-right">Received</span>
      </div>
      <div
        ref={scrollRef}
        className="min-h-0 flex-1 overflow-y-auto px-3"
        data-testid="inbox-scroll"
        aria-busy={invoices === null}
      >
        {invoices === null ? (
          <div aria-hidden="true" className="space-y-2 py-2">
            {Array.from({ length: 10 }).map((_, i) => (
              <Skeleton key={i} className="h-9 w-full" />
            ))}
          </div>
        ) : rows.length === 0 ? (
          <p className="text-muted-foreground py-12 text-center text-sm">
            No invoices match these filters.
          </p>
        ) : (
          <div
            role="listbox"
            aria-label="Invoices"
            aria-activedescendant={activeId ? `inbox-row-${activeId}` : undefined}
            tabIndex={0}
            // No ring on the list itself (the scroll container would clip it): keyboard focus is
            // shown on the active row instead, as a 2px ring drawn inside the row's padding.
            className="group/list relative outline-none"
            style={{ height: virtualizer.getTotalSize() }}
          >
            {virtualizer.getVirtualItems().map((item) => {
              const row = rows[item.index];
              if (!row) return null;
              const active = item.index === activeIndex;
              return (
                <div
                  key={row.id}
                  id={`inbox-row-${row.id}`}
                  role="option"
                  aria-selected={active}
                  onClick={() => {
                    setActiveIndex(item.index);
                    router.push(`/invoices/${row.id}`);
                  }}
                  onMouseEnter={() => setActiveIndex(item.index)}
                  className={cn(
                    "inbox-grid border-border/70 cursor-pointer items-center border-b px-3 text-sm transition-colors",
                    active
                      ? "bg-accent/10 outline-ring outline-1 -outline-offset-1 outline-solid group-focus-visible/list:outline-2 group-focus-visible/list:-outline-offset-2"
                      : "hover:bg-muted/60",
                  )}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    right: 0,
                    height: item.size,
                    transform: `translateY(${item.start}px)`,
                  }}
                >
                  <span
                    className="ic-num tabular-mono truncate text-xs"
                    title={
                      row.invoiceNumber
                        ? `Invoice ID ${row.id}`
                        : `No supplier invoice number yet; internal ID ${row.id}`
                    }
                    data-testid="inbox-invoice-number"
                  >
                    <InvoiceRef row={row} />
                  </span>
                  <span className="ic-vendor">
                    <span className="block truncate font-medium">
                      {row.vendorName ?? "Unknown vendor"}
                    </span>
                    {/* Only the number truncates: the received time is short, fixed-width and
                        always shown whole, however long a supplier's invoice number is. */}
                    <span
                      className="ic-sub text-muted-foreground flex min-w-0 gap-1 text-2xs"
                      data-testid="inbox-sub"
                    >
                      <span className="tabular-mono min-w-0 truncate">
                        <InvoiceRef row={row} />
                      </span>
                      <span className="sr-only">, received</span>
                      <span aria-hidden="true">·</span>
                      <span className="tabular-mono shrink-0" data-testid="inbox-sub-received">
                        {formatReceivedCompact(row.receivedAt)}
                      </span>
                    </span>
                  </span>
                  <span className="ic-status">
                    <StatusBadge status={row.status} />
                  </span>
                  <span className="ic-route">
                    <RouteBadge route={row.route} />
                  </span>
                  <span className="ic-amount text-right">
                    <span
                      className="tabular-mono block truncate"
                      data-testid="inbox-amount"
                      title={formatAed(row.total)}
                    >
                      {formatAed(row.total)}
                    </span>
                    <span className="ic-sub text-muted-foreground block truncate text-2xs">
                      {STATUS_LABEL[row.status]}
                      {row.flagCount > 0 &&
                        ` · ${row.flagCount} flag${row.flagCount === 1 ? "" : "s"}`}
                    </span>
                  </span>
                  <span className="ic-conf">
                    <ConfidenceMeter value={row.confidence} />
                  </span>
                  <span className="ic-flags text-center">
                    <FlagCountBadge count={row.flagCount} />
                  </span>
                  <span
                    className="ic-received tabular-mono text-muted-foreground text-right text-xs"
                    data-testid="inbox-received"
                  >
                    {formatReceivedCompact(row.receivedAt)}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
