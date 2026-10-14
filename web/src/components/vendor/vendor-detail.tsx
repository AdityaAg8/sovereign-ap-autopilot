/** Vendor detail: bilingual identity, bank-detail history and recent invoices from this vendor
 * ("vendor view"; "bank_details (verified flag, history)"). */

"use client";

import Link from "next/link";
import { ShieldAlert, ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { StatusBadge, RouteBadge } from "@/components/inbox/status-badge";
import { Amount } from "@/components/inbox/amount";
import { formatAed, formatCount, formatReceivedAt } from "@/lib/format";
import type { InvoiceSummary, Vendor } from "@/lib/api/types";

export function VendorDetail({
  vendor,
  invoices,
}: {
  vendor: Vendor;
  invoices: readonly InvoiceSummary[];
}) {
  return (
    <div className="mx-auto max-w-4xl space-y-8 p-3 sm:p-5">
      <div>
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold">{vendor.nameEn}</h2>
          {vendor.isNew && (
            <Badge variant="secondary" className="bg-warning/10 text-warning border-transparent">
              New vendor
            </Badge>
          )}
        </div>
        {vendor.nameAr && (
          <p dir="rtl" lang="ar" className="text-muted-foreground mt-0.5">
            {vendor.nameAr}
          </p>
        )}
        <dl className="mt-4 grid grid-cols-3 gap-4 text-sm">
          <div>
            <dt className="text-muted-foreground text-xs">TRN</dt>
            <dd className="tabular-mono">{vendor.trn ?? "—"}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground text-xs">Invoices</dt>
            <dd className="tabular-mono">{formatCount(vendor.invoiceCount)}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground text-xs">Total (AED)</dt>
            <dd className="tabular-mono">
              <Amount text={formatAed(vendor.totalAed)} />
            </dd>
          </div>
        </dl>
      </div>

      <section>
        <h3 className="mb-2 text-sm font-semibold">Bank details</h3>
        <ul className="panel divide-border/60 divide-y overflow-hidden">
          {vendor.bankDetails.map((bank) => (
            <li key={bank.id} className="flex items-center justify-between px-4 py-3">
              <div>
                <p className="tabular-mono text-sm">{bank.ibanMasked}</p>
                {bank.bankName && <p className="text-muted-foreground text-xs">{bank.bankName}</p>}
              </div>
              <span
                className={
                  bank.verified
                    ? "text-success flex items-center gap-1.5 text-sm"
                    : "text-warning flex items-center gap-1.5 text-sm"
                }
              >
                {bank.verified ? (
                  <ShieldCheck className="size-4" />
                ) : (
                  <ShieldAlert className="size-4" />
                )}
                {bank.verified
                  ? `Verified ${bank.verifiedAt ? formatReceivedAt(bank.verifiedAt) : ""}`
                  : "Pending verification"}
              </span>
            </li>
          ))}
          {vendor.bankDetails.length === 0 && (
            <li className="text-muted-foreground px-4 py-6 text-center text-sm">
              No bank details on file yet.
            </li>
          )}
        </ul>
      </section>

      <section>
        <h3 className="mb-2 text-sm font-semibold">Recent invoices</h3>
        <ul className="panel divide-border/60 divide-y overflow-hidden">
          {invoices.slice(0, 20).map((inv) => (
            <li key={inv.id}>
              <Link
                href={`/invoices/${inv.id}`}
                className="hover:bg-muted/60 flex items-center justify-between gap-3 px-4 py-2.5 text-sm"
              >
                <span className="tabular-mono">{inv.id}</span>
                <StatusBadge status={inv.status} />
                <RouteBadge route={inv.route} />
                <span className="tabular-mono ms-auto">
                  <Amount text={formatAed(inv.total)} />
                </span>
              </Link>
            </li>
          ))}
          {invoices.length === 0 && (
            <li className="text-muted-foreground px-4 py-6 text-center text-sm">
              No invoices yet.
            </li>
          )}
        </ul>
      </section>
    </div>
  );
}
