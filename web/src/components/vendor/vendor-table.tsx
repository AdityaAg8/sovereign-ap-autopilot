/** Vendor list: EN/AR name, TRN, verified bank status and invoice volume ("vendor
 * view"). Small, fixed dataset — a plain (non-virtualized) table reads better here than the
 * inbox's virtualized one. */

"use client";

import Link from "next/link";
import { ShieldAlert, ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatAed, formatCount } from "@/lib/format";
import type { Vendor } from "@/lib/api/types";

export function VendorTable({ vendors }: { vendors: readonly Vendor[] }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Vendor</TableHead>
          <TableHead>TRN</TableHead>
          <TableHead>Bank details</TableHead>
          <TableHead className="text-right">Invoices</TableHead>
          <TableHead className="text-right">Total (AED)</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {vendors.map((vendor) => {
          const verified = vendor.verified;
          return (
            <TableRow key={vendor.id} className="hover:bg-muted/60">
              <TableCell>
                <Link
                  href={`/vendors/${vendor.id}`}
                  className="hover:text-accent focus-visible:text-accent block"
                >
                  <span className="font-medium">{vendor.nameEn}</span>
                  {vendor.nameAr && (
                    <span dir="rtl" lang="ar" className="text-muted-foreground block text-sm">
                      {vendor.nameAr}
                    </span>
                  )}
                  {vendor.isNew && (
                    <Badge
                      variant="secondary"
                      className="bg-warning/20 text-warning mt-1 border-transparent"
                    >
                      New vendor
                    </Badge>
                  )}
                </Link>
              </TableCell>
              <TableCell className="tabular-mono">{vendor.trn ?? "—"}</TableCell>
              <TableCell>
                <span
                  className={
                    verified
                      ? "text-success flex items-center gap-1.5"
                      : "text-warning flex items-center gap-1.5"
                  }
                >
                  {verified ? (
                    <ShieldCheck className="size-3.5" />
                  ) : (
                    <ShieldAlert className="size-3.5" />
                  )}
                  {verified ? "Verified" : "Needs verification"}
                </span>
              </TableCell>
              <TableCell className="tabular-mono text-right">
                {formatCount(vendor.invoiceCount)}
              </TableCell>
              <TableCell className="tabular-mono text-right">
                {formatAed(vendor.totalAed)}
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
