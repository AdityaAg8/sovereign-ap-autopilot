/** Vendor list route. */

"use client";

import { useEffect, useState } from "react";
import { VendorTable } from "@/components/vendor/vendor-table";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, getApiClient } from "@/lib/api/client";
import { useSetPageHeader } from "@/lib/shell/page-header-context";
import type { Vendor } from "@/lib/api/types";

export default function VendorsPage() {
  useSetPageHeader({ title: "Vendors" });
  const [vendors, setVendors] = useState<readonly Vendor[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getApiClient()
      .listVendors()
      .then(setVendors)
      .catch((err: unknown) => {
        setError(err instanceof ApiError ? err.message : "Could not load vendors.");
      });
  }, []);

  if (error) {
    return <p className="text-muted-foreground p-6 text-sm">{error}</p>;
  }

  if (!vendors) {
    return (
      <div className="space-y-2 p-3 sm:p-5">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-12 w-full" />
        ))}
      </div>
    );
  }

  return (
    <div className="p-3 sm:p-5">
      <VendorTable vendors={vendors} />
    </div>
  );
}
