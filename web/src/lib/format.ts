/** Locale-aware formatting helpers for money, dates and confidence values shared by the inbox,
 * invoice detail and vendor screens (PRD §10 M5.4). Keeps AED amounts and timestamps consistent
 * everywhere they render. */

const aed = new Intl.NumberFormat("en-AE", {
  style: "currency",
  currency: "AED",
  currencyDisplay: "code",
  minimumFractionDigits: 2,
});

export function formatAed(value: string | number | null): string {
  if (value === null) return "—";
  const n = typeof value === "string" ? Number(value) : value;
  if (Number.isNaN(n)) return "—";
  return aed.format(n);
}

const groupedNumber = new Intl.NumberFormat("en-AE", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

/** Grouped digits with no currency prefix, for a money field shown next to a label that already
 * says "(AED)" (lib/invoice-fields.ts's `isMoneyField`) — `formatAed` would repeat the currency. */
export function formatGroupedAmount(raw: string): string {
  const n = Number(raw);
  return Number.isNaN(n) ? raw : groupedNumber.format(n);
}

const groupedCount = new Intl.NumberFormat("en-AE");

/** Grouped integer count (vendor invoice totals etc.) — the same thousands-separator convention
 * as every AED amount in the app, applied consistently even though today's fixture counts are
 * all small enough that it rarely shows visually. `null` (apx/api does not aggregate this yet,
 * see lib/api/types.ts's `Vendor.invoiceCount`) renders as the same "—" every other missing
 * value in this app uses. */
export function formatCount(n: number | null): string {
  return n === null ? "—" : groupedCount.format(n);
}

export function formatConfidence(value: number | null): string {
  if (value === null) return "—";
  return `${Math.round(value * 100)}%`;
}

const dateTime = new Intl.DateTimeFormat("en-AE", {
  dateStyle: "medium",
  timeStyle: "short",
  timeZone: "Asia/Dubai",
});

export function formatReceivedAt(iso: string): string {
  return dateTime.format(new Date(iso));
}

// "22 Sep, 15:45" — day/month/time only, 24h clock, so the inbox table's Received column always
// fits on one line at its fixed column width (web/DESIGN.md dense-table density) instead of the
// longer `formatReceivedAt` wrapping onto two.
const dateTimeCompact = new Intl.DateTimeFormat("en-AE", {
  day: "2-digit",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
  timeZone: "Asia/Dubai",
});

export function formatReceivedCompact(iso: string): string {
  return dateTimeCompact.format(new Date(iso)).replace(",", "");
}

export function confidenceTone(value: number | null): "success" | "warning" | "destructive" {
  if (value === null) return "warning";
  if (value >= 0.85) return "success";
  if (value >= 0.6) return "warning";
  return "destructive";
}
