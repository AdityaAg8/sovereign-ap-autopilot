/** Review queue route: invoices awaiting a human decision, in received order (PRD §10 M5.4). Open
 * one to approve, fix a field, or reject — see the invoice detail page's review action bar. */

"use client";

import { useEffect, useState } from "react";
import { InboxTable } from "@/components/inbox/inbox-table";
import { Skeleton } from "@/components/ui/skeleton";
import { getApiClient } from "@/lib/api/client";
import { useSetPageHeader } from "@/lib/shell/page-header-context";
import type { InvoiceSummary } from "@/lib/api/types";

export default function ReviewQueuePage() {
  useSetPageHeader({ title: "Review queue" });
  const [invoices, setInvoices] = useState<readonly InvoiceSummary[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    getApiClient()
      .listInvoices({ status: "awaiting_review", limit: 2000 })
      .then((rows) => {
        if (!cancelled) setInvoices(rows);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!invoices) {
    return (
      <div className="space-y-2 p-6">
        {Array.from({ length: 10 }).map((_, i) => (
          <Skeleton key={i} className="h-11 w-full" />
        ))}
      </div>
    );
  }

  if (invoices.length === 0) {
    return (
      <div className="text-muted-foreground flex h-full items-center justify-center text-sm">
        Nothing waiting on you — the queue is clear.
      </div>
    );
  }

  return (
    <InboxTable invoices={invoices} lockStatus="awaiting_review" shortcutScope="Review queue" />
  );
}
