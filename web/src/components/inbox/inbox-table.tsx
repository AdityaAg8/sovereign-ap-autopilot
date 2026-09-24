/** Virtualized inbox table: status, vendor, amount, confidence and flags for every invoice, with
 * j/k/enter keyboard navigation (PRD §10 M5.4). Uses @tanstack/react-virtual so the DOM only ever
 * holds the visible rows regardless of dataset size. */

"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useVirtualizer } from "@tanstack/react-virtual";
import { StatusBadge, RouteBadge, FlagCountBadge } from "@/components/inbox/status-badge";
import { ConfidenceMeter } from "@/components/inbox/confidence-meter";
import { InboxFilters, type InboxFilterState } from "@/components/inbox/inbox-filters";
import { useHotkeys } from "@/hooks/use-hotkeys";
import { useRegisterShortcuts } from "@/lib/shortcuts/registry";
import { formatAed, formatReceivedCompact } from "@/lib/format";
import type { InvoiceStatus, InvoiceSummary } from "@/lib/api/types";
import { cn } from "@/lib/utils";

const ROW_HEIGHT = 44;

// Vendor is the only flexible track (minmax(0, 1fr)) so it soaks up whatever room the fixed,
// content-sized columns leave — sized from the longest real value in each ("inv-20260101-0419",
// "Awaiting review", "AED 1,234,567.89"), not guessed. `gap-x-3` keeps a right-aligned column's
// text from touching the next column's left-aligned label (Amount/Confidence otherwise read as
// one run-on word).
const GRID_COLS = "9.5rem minmax(0,1fr) 8.5rem 6rem 8rem 6.5rem 4rem 7rem";

const SHORTCUTS = [
  { combo: "j", description: "Next invoice" },
  { combo: "k", description: "Previous invoice" },
  { combo: "Enter", description: "Open the highlighted invoice" },
] as const;

export function InboxTable({
  invoices,
  lockStatus,
  shortcutScope = "Inbox",
}: {
  invoices: readonly InvoiceSummary[];
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
    const q = filters.query.trim().toLowerCase();
    return invoices.filter((row) => {
      if (lockStatus && row.status !== lockStatus) return false;
      if (!lockStatus && filters.status !== "all" && row.status !== filters.status) return false;
      if (filters.route !== "all" && row.route !== filters.route) return false;
      if (
        q &&
        !row.id.toLowerCase().includes(q) &&
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
    <div className="flex h-full min-h-0 flex-col">
      <InboxFilters value={filters} onChange={setFilters} hideStatus={Boolean(lockStatus)} />
      {/* Visual column labels only — the interactive rows below are an ARIA listbox (each row's
          own accessible name carries its data), so this row stays out of the accessibility tree
          rather than presenting an orphaned `role="row"` with no table/grid ancestor. */}
      <div
        aria-hidden="true"
        className="text-muted-foreground border-border grid gap-x-3 border-y px-6 py-2 text-xs font-medium tracking-wide uppercase"
        style={{ gridTemplateColumns: GRID_COLS }}
      >
        <span>Invoice #</span>
        <span>Vendor</span>
        <span>Status</span>
        <span>Route</span>
        <span className="text-right">Amount</span>
        <span>Confidence</span>
        <span className="text-center">Flags</span>
        <span className="text-right">Received</span>
      </div>
      <div
        ref={scrollRef}
        className="min-h-0 flex-1 overflow-y-auto px-3"
        data-testid="inbox-scroll"
      >
        {rows.length === 0 ? (
          <p className="text-muted-foreground py-12 text-center text-sm">
            No invoices match these filters.
          </p>
        ) : (
          <div
            role="listbox"
            aria-label="Invoices"
            aria-activedescendant={activeId ? `inbox-row-${activeId}` : undefined}
            tabIndex={0}
            className="relative outline-none"
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
                    "border-border/70 grid cursor-pointer items-center gap-x-3 border-b px-3 text-sm transition-colors",
                    active
                      ? "bg-accent/10 outline-ring outline-1 -outline-offset-1"
                      : "hover:bg-muted/60",
                  )}
                  style={{
                    gridTemplateColumns: GRID_COLS,
                    position: "absolute",
                    top: 0,
                    left: 0,
                    right: 0,
                    height: item.size,
                    transform: `translateY(${item.start}px)`,
                  }}
                >
                  <span className="tabular-mono text-muted-foreground truncate pe-2 text-xs">
                    {row.id}
                  </span>
                  <span className="truncate pe-2 font-medium">
                    {row.vendorName ?? "Unknown vendor"}
                  </span>
                  <span>
                    <StatusBadge status={row.status} />
                  </span>
                  <span>
                    <RouteBadge route={row.route} />
                  </span>
                  <span className="tabular-mono text-right">{formatAed(row.total)}</span>
                  <span>
                    <ConfidenceMeter value={row.confidence} />
                  </span>
                  <span className="text-center">
                    <FlagCountBadge count={row.flagCount} />
                  </span>
                  <span className="tabular-mono text-muted-foreground text-right text-xs">
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
