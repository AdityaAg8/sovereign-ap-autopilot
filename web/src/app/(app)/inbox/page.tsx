/** Inbox route: every invoice regardless of status, in the virtualized table. */

"use client";

import { useEffect, useState } from "react";
import { InboxTable } from "@/components/inbox/inbox-table";
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

  // Rendered while loading too (`invoices === null`): the table shows its filter bar and column
  // labels over skeleton rows, so the page's real layout paints before the data arrives.
  return <InboxTable invoices={invoices} />;
}
