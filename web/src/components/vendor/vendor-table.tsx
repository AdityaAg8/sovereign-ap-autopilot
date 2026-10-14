/** Vendor list: EN/AR name, TRN, verified bank status and invoice volume ("vendor
 * view"). Small, fixed dataset, so a plain (non-virtualized) table inside one panel reads better
 * here than the inbox's virtualized one. */

"use client";

import Link from "next/link";
import { ShieldAlert, ShieldCheck } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Amount } from "@/components/inbox/amount";
import { formatAed, formatCount } from "@/lib/format";
import type { Vendor } from "@/lib/api/types";

const HEAD = "text-muted-foreground h-9 px-5 text-2xs";

export function VendorTable({ vendors }: { vendors: readonly Vendor[] }) {
  return (
    <div className="panel overflow-hidden">
      <Table>
        <TableHeader className="bg-muted/40 dark:bg-muted/30">
          <TableRow className="hover:bg-transparent">
            <TableHead className={HEAD}>Vendor</TableHead>
            <TableHead className={HEAD}>TRN</TableHead>
            <TableHead className={HEAD}>Bank details</TableHead>
            <TableHead className={`${HEAD} text-right`}>Invoices</TableHead>
            <TableHead className={`${HEAD} text-right`}>Total (AED)</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {vendors.map((vendor) => {
            const verified = vendor.verified;
            return (
              <TableRow key={vendor.id} className="border-border/60 hover:bg-muted/50">
                <TableCell className="px-5 py-3">
                  <Link
                    href={`/vendors/${vendor.id}`}
                    className="hover:text-accent focus-visible:text-accent group flex flex-wrap items-baseline gap-x-3 gap-y-0.5"
                  >
                    <span className="font-medium">{vendor.nameEn}</span>
                    {vendor.isNew && (
                      <span className="text-warning bg-warning/10 rounded px-1.5 py-0.5 text-2xs font-medium">
                        New vendor
                      </span>
                    )}
                    {vendor.nameAr && (
                      <span
                        dir="rtl"
                        lang="ar"
                        className="text-muted-foreground basis-full text-left text-xs"
                      >
                        {vendor.nameAr}
                      </span>
                    )}
                  </Link>
                </TableCell>
                <TableCell className="tabular-mono px-5 text-xs">{vendor.trn ?? "—"}</TableCell>
                <TableCell className="px-5">
                  <span className="inline-flex items-center gap-1.5 text-xs">
                    {verified ? (
                      <ShieldCheck className="text-success size-3.5" aria-hidden />
                    ) : (
                      <ShieldAlert className="text-warning size-3.5" aria-hidden />
                    )}
                    <span
                      className={verified ? "text-muted-foreground" : "text-warning font-medium"}
                    >
                      {verified ? "Verified" : "Needs verification"}
                    </span>
                  </span>
                </TableCell>
                <TableCell className="tabular-mono px-5 text-right">
                  {formatCount(vendor.invoiceCount)}
                </TableCell>
                <TableCell className="tabular-mono px-5 text-right">
                  <Amount text={formatAed(vendor.totalAed)} />
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
