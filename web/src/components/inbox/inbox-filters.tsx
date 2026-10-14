/** Search + status/route filters above the inbox table. Pure controlled
 * inputs — filtering itself runs client-side over the already-fetched page in inbox-table.tsx.
 * The two pickers are `FilterSelect`s, whose Radix select loads on the first interaction. */

"use client";

import { SearchIcon, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { FilterSelect, type FilterSelectOption } from "@/components/inbox/filter-select";
import type { InvoiceStatus, RouteDecision } from "@/lib/api/types";
import { formatCount } from "@/lib/format";

export interface InboxFilterState {
  query: string;
  status: InvoiceStatus | "all";
  route: RouteDecision | "all";
}

const STATUSES: readonly InvoiceStatus[] = [
  "received",
  "scanned",
  "parsed",
  "extracted",
  "gated",
  "validated",
  "fraud_checked",
  "awaiting_review",
  "approved",
  "sent",
  "archived",
  "blocked",
  "failed",
];

const STATUS_OPTIONS: readonly FilterSelectOption[] = [
  { value: "all", label: "All statuses" },
  ...STATUSES.map((s) => ({ value: s, label: s.replaceAll("_", " "), className: "capitalize" })),
];

const ROUTE_OPTIONS: readonly FilterSelectOption[] = [
  { value: "all", label: "All routes" },
  { value: "auto", label: "Auto" },
  { value: "review", label: "Review" },
  { value: "block", label: "Blocked" },
];

export function InboxFilters({
  value,
  onChange,
  hideStatus = false,
  shown = null,
  total = null,
}: {
  value: InboxFilterState;
  onChange: (next: InboxFilterState) => void;
  /** Review queue pre-filters to "awaiting_review"; the status selector would be redundant. */
  hideStatus?: boolean;
  /** Rows left after filtering and rows loaded; `null` while the list is still loading. */
  shown?: number | null;
  total?: number | null;
}) {
  const dirty =
    value.query !== "" || (!hideStatus && value.status !== "all") || value.route !== "all";
  return (
    <div className="flex flex-wrap items-center gap-2 px-4 py-3 sm:px-5">
      <div className="relative min-w-56 flex-1 sm:max-w-md">
        <SearchIcon className="text-muted-foreground pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2" />
        <Input
          value={value.query}
          onChange={(e) => onChange({ ...value, query: e.target.value })}
          placeholder="Search invoice number, ID or vendor…"
          aria-label="Search invoices"
          className="ps-8"
        />
      </div>
      {!hideStatus && (
        <FilterSelect
          value={value.status}
          onValueChange={(status) =>
            onChange({ ...value, status: status as InboxFilterState["status"] })
          }
          options={STATUS_OPTIONS}
          ariaLabel="Filter by status"
          className="w-44"
          placeholder="Status"
        />
      )}
      <FilterSelect
        value={value.route}
        onValueChange={(route) => onChange({ ...value, route: route as InboxFilterState["route"] })}
        options={ROUTE_OPTIONS}
        ariaLabel="Filter by route"
        className="w-36"
        placeholder="Route"
      />
      {dirty && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() =>
            onChange({ query: "", status: hideStatus ? value.status : "all", route: "all" })
          }
        >
          <X /> Clear
        </Button>
      )}
      {/* Fixed-width slot, filled once rows arrive, so nothing beside it moves. */}
      <p className="text-muted-foreground ms-auto hidden min-w-28 text-right text-xs sm:block">
        {shown !== null && total !== null && (
          <>
            <span className="tabular-mono text-foreground">{formatCount(shown)}</span>
            {shown !== total && (
              <>
                {" "}
                of <span className="tabular-mono">{formatCount(total)}</span>
              </>
            )}{" "}
            {total === 1 ? "invoice" : "invoices"}
          </>
        )}
      </p>
    </div>
  );
}
