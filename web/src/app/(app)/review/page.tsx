/** Review queue route: invoices awaiting a human decision, in received order. Open
 * one to approve, fix a field, or reject — see the invoice detail page's review action bar. */

"use client";

import { useEffect, useState } from "react";
import { InboxTable } from "@/components/inbox/inbox-table";
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

  // While loading (`invoices === null`) the table renders its filter bar and column labels over
  // skeleton rows, so the page's real layout paints before the data arrives.
  if (invoices?.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-1 p-6 text-center">
        <p className="text-sm font-medium">The review queue is clear</p>
        <p className="text-muted-foreground text-sm">
          New invoices that need a person will show up here.
        </p>
      </div>
    );
  }

  return (
    <InboxTable invoices={invoices} lockStatus="awaiting_review" shortcutScope="Review queue" />
  );
}
