/** Most recent invoice activity (PRD §10 M10.3): where each of the latest invoices is now. Rows
 * that are new or moved since the last tick glow briefly (keyed by update time so the glow
 * replays on every move); ids and amounts are mono so the columns line up. */
import Link from "next/link";
import type { OpsRecentInvoice } from "@/lib/api/types";
import { StatusBadge, RouteBadge } from "@/components/inbox/status-badge";
import { cn } from "@/lib/utils";

function ago(iso: string, now: number): string {
  const s = Math.max(0, Math.round((now - Date.parse(iso)) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function ActivityTable({
  rows,
  now,
  fresh,
}: {
  rows: readonly OpsRecentInvoice[];
  now: number;
  fresh: ReadonlySet<string>;
}) {
  return (
    <section aria-labelledby="ops-activity" className="bg-card overflow-hidden rounded-xl border">
      <h2 id="ops-activity" className="px-5 pt-4 pb-3 text-sm font-semibold sm:px-6">
        Latest activity
      </h2>
      {rows.length === 0 ? (
        <p className="text-muted-foreground px-5 pb-6 text-sm sm:px-6">
          No invoices yet. Drop a PDF into the watched folder or upload one from the inbox.
        </p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm whitespace-nowrap">
            <thead className="text-muted-foreground border-y text-xs">
              <tr>
                <th scope="col" className="py-2 pr-4 pl-5 text-left font-medium sm:pl-6">
                  Invoice
                </th>
                <th scope="col" className="px-4 py-2 text-left font-medium">
                  Vendor
                </th>
                <th scope="col" className="px-4 py-2 text-left font-medium">
                  Stage
                </th>
                <th scope="col" className="px-4 py-2 text-left font-medium">
                  Route
                </th>
                <th scope="col" className="px-4 py-2 text-right font-medium">
                  Total
                </th>
                <th scope="col" className="py-2 pr-5 pl-4 text-right font-medium sm:pr-6">
                  Updated
                </th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {rows.map((row) => (
                <tr
                  key={`${row.id}:${row.updatedAt}`}
                  className={cn("hover:bg-muted/40", fresh.has(row.id) && "ops-fresh")}
                >
                  <td className="py-2.5 pr-4 pl-5 font-mono sm:pl-6">
                    <Link
                      href={`/invoices/${row.id}`}
                      className="hover:text-accent underline-offset-4 hover:underline"
                    >
                      {row.number}
                    </Link>
                  </td>
                  <td className="max-w-52 truncate px-4 py-2.5">{row.vendorName ?? "—"}</td>
                  <td className="px-4 py-2.5">
                    <StatusBadge status={row.status} />
                  </td>
                  <td className="px-4 py-2.5">
                    <RouteBadge route={row.route} />
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono tabular-nums">
                    {row.total}{" "}
                    <span className="text-muted-foreground text-xs">{row.currency}</span>
                  </td>
                  <td className="text-muted-foreground py-2.5 pr-5 pl-4 text-right text-xs tabular-nums sm:pr-6">
                    {ago(row.updatedAt, now)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
