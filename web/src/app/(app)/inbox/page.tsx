/** Inbox route: every invoice regardless of status, in the virtualized table (PRD §10 M5.4). */

"use client";

import { useEffect, useState } from "react";
import { InboxTable } from "@/components/inbox/inbox-table";
import { Skeleton } from "@/components/ui/skeleton";
import { getApiClient } from "@/lib/api/client";
import { useSetPageHeader } from "@/lib/shell/page-header-context";
import type { InvoiceSummary } from "@/lib/api/types";

export default function InboxPage() {
  useSetPageHeader({ title: "Inbox" });
  const [invoices, setInvoices] = useState<readonly InvoiceSummary[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    getApiClient()
      .listInvoices({ limit: 2000 })
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

  return <InboxTable invoices={invoices} />;
}
