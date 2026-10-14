/** Vendor detail route: identity, bank history and recent invoices. */

"use client";

import { use, useEffect, useMemo, useState } from "react";
import { VendorDetail } from "@/components/vendor/vendor-detail";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, getApiClient } from "@/lib/api/client";
import { useSetPageHeader } from "@/lib/shell/page-header-context";
import type { InvoiceSummary, Vendor } from "@/lib/api/types";

export default function VendorDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [vendor, setVendor] = useState<Vendor | null>(null);
  const [invoices, setInvoices] = useState<readonly InvoiceSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getApiClient()
      .getVendor(id)
      .then(setVendor)
      .catch((err: unknown) => {
        setError(err instanceof ApiError ? err.message : "Could not load this vendor.");
      });
    getApiClient()
      .listInvoices({ vendorId: id, limit: 200 })
      .then(setInvoices)
      .catch(() => setInvoices([]));
  }, [id]);

  useSetPageHeader(useMemo(() => ({ title: vendor?.nameEn ?? "Vendor" }), [vendor]));

  if (error) {
    return <p className="text-muted-foreground p-6 text-sm">{error}</p>;
  }

  if (!vendor) {
    return (
      <div className="mx-auto max-w-4xl space-y-4 p-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  return <VendorDetail vendor={vendor} invoices={invoices} />;
}
