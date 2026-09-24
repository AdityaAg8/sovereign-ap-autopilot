/** Search + status/route filters above the inbox table (PRD §10 M5.4: "filters"). Pure controlled
 * inputs — filtering itself runs client-side over the already-fetched page in inbox-table.tsx. */

"use client";

import { SearchIcon, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { InvoiceStatus, RouteDecision } from "@/lib/api/types";

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

export function InboxFilters({
  value,
  onChange,
  hideStatus = false,
}: {
  value: InboxFilterState;
  onChange: (next: InboxFilterState) => void;
  /** Review queue pre-filters to "awaiting_review"; the status selector would be redundant. */
  hideStatus?: boolean;
}) {
  const dirty =
    value.query !== "" || (!hideStatus && value.status !== "all") || value.route !== "all";
  return (
    <div className="flex flex-wrap items-center gap-2 px-6 py-3">
      <div className="relative min-w-56 flex-1">
        <SearchIcon className="text-muted-foreground pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2" />
        <Input
          value={value.query}
          onChange={(e) => onChange({ ...value, query: e.target.value })}
          placeholder="Search invoice ID or vendor…"
          aria-label="Search invoices"
          className="ps-8"
        />
      </div>
      {!hideStatus && (
        <Select
          value={value.status}
          onValueChange={(status) =>
            onChange({ ...value, status: status as InboxFilterState["status"] })
          }
        >
          <SelectTrigger className="w-44" aria-label="Filter by status">
            <SelectValue placeholder="Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All statuses</SelectItem>
            {STATUSES.map((s) => (
              <SelectItem key={s} value={s} className="capitalize">
                {s.replaceAll("_", " ")}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
      <Select
        value={value.route}
        onValueChange={(route) => onChange({ ...value, route: route as InboxFilterState["route"] })}
      >
        <SelectTrigger className="w-36" aria-label="Filter by route">
          <SelectValue placeholder="Route" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All routes</SelectItem>
          <SelectItem value="auto">Auto</SelectItem>
          <SelectItem value="review">Review</SelectItem>
          <SelectItem value="block">Blocked</SelectItem>
        </SelectContent>
      </Select>
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
    </div>
  );
}
