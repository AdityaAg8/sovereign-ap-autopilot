/** Locale-aware formatting helpers for money, dates and confidence values shared by the inbox,
 * invoice detail and vendor screens. Keeps AED amounts and timestamps consistent
 * everywhere they render. */

// The one number formatter every amount and count in the app goes through: "en-AE" grouping
// (comma thousands, period decimals) pinned to Latin digits (`nu-latn`), so an amount reads the
// same on an Arabic invoice's field list as in the English inbox, and never switches to
// Arabic-Indic digits with the browser locale.
const NUMBER_LOCALE = "en-AE-u-nu-latn";
const money = new Intl.NumberFormat(NUMBER_LOCALE, {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const count = new Intl.NumberFormat(NUMBER_LOCALE, { maximumFractionDigits: 0 });

const ARABIC_INDIC_ZERO = 0x0660;
const EASTERN_ARABIC_INDIC_ZERO = 0x06f0;
/** apx.common.normalize._CURRENCY_TOKENS: AED in Latin and Arabic (U+062F. U+0625), and Dhs. */
const CURRENCY_TOKENS = ["AED", "aed", "Aed", "د.إ", "Dhs", "DHS", "dhs"] as const;
/** Canonical form after separators are resolved; at most 2 decimals, since the display shows 2
 * and silently rounding "1,234.567" would misstate the extracted amount. */
const CANONICAL_AMOUNT = /^\d+(\.\d{1,2})?$/;

/** True when the float `value` still reads back as the canonical `text` it came from: the integer
 * part is a safe integer and no digit was lost to double precision. apx keeps amounts as exact
 * Decimals, so a value a float cannot hold exactly is shown raw rather than as a nearby number. */
function isExact(value: number, text: string): boolean {
  if (!Number.isSafeInteger(Math.trunc(value))) return false;
  // `whole` and `fraction` are digits only: callers pass text matching CANONICAL_AMOUNT.
  const [whole = "", fraction = ""] = text.split(".");
  const canonical = whole.replace(/^0+(?=\d)/, "") + (fraction ? `.${fraction}` : "");
  return value.toFixed(fraction.length) === canonical;
}

/** Parses an amount exactly as apx.common.normalize.normalize_amount does: ASCII or Arabic-Indic
 * (U+0660-0669, U+06F0-06F9) digits, Arabic thousands/decimal separators (U+066C/U+066B), an
 * optional AED marker, a sign or accounting parentheses, and the backend's comma-vs-period rule
 * (both present: whichever comes last is the decimal point, so "1.234,56" is 1234.56; a lone
 * comma is a thousands separator). Returns null, never a guess, for anything else, for more
 * than 2 decimals, or for a value a float cannot hold exactly (integer part past
 * Number.MAX_SAFE_INTEGER, or digits lost to precision), so callers show the raw text instead.
 * The first two are deliberate differences from the backend, which keeps any number of
 * decimals and any magnitude as a Decimal (see e2e/amount-parity.spec.ts). */
export function parseAmount(raw: string | number | null): number | null {
  if (raw === null) return null;
  if (typeof raw === "number") {
    return Number.isFinite(raw) &&
      Number.isSafeInteger(Math.trunc(raw)) &&
      Math.round(raw * 100) / 100 === raw
      ? raw
      : null;
  }
  let text = raw
    .replace(/[٠-٩]/g, (d) => String(d.charCodeAt(0) - ARABIC_INDIC_ZERO))
    .replace(/[۰-۹]/g, (d) => String(d.charCodeAt(0) - EASTERN_ARABIC_INDIC_ZERO))
    .replace(/٬/g, ",")
    .replace(/٫/g, ".");
  for (const token of CURRENCY_TOKENS) text = text.replaceAll(token, "");
  text = text.replace(/\u00a0/g, " ").trim();

  let negative = false;
  if (text.startsWith("(") && text.endsWith(")")) {
    negative = true;
    text = text.slice(1, -1).trim();
  }
  if (text.startsWith("-") || text.startsWith("−")) {
    negative = true;
    text = text.slice(1).trim();
  } else if (text.startsWith("+")) {
    text = text.slice(1).trim();
  }
  text = text.replace(/\s+/g, "");
  if (!text) return null;

  if (text.includes(",") && text.includes(".")) {
    text =
      text.lastIndexOf(",") > text.lastIndexOf(".")
        ? text.replaceAll(".", "").replaceAll(",", ".")
        : text.replaceAll(",", "");
  } else if (text.includes(",")) {
    text = text.replaceAll(",", "");
  }
  if (!CANONICAL_AMOUNT.test(text)) return null;
  const value = Number(text);
  if (!isExact(value, text)) return null;
  return negative ? -value : value;
}

/** Grouped digits with no currency prefix, for a money field shown next to a label that already
 * says "(AED)" (lib/invoice-fields.ts's `isMoneyField`) — `formatAed` would repeat the currency.
 * Unparseable input is shown as-is, so a reviewer still sees exactly what was extracted. */
export function formatGroupedAmount(raw: string): string {
  const n = parseAmount(raw);
  return n === null ? raw : money.format(n);
}

/** "AED 1,234,567.89": the currency code, a space and the shared grouped format. A value that
 * does not parse is shown as its raw text, never reformatted into a different number; only a
 * missing one (null or blank) renders as "—". */
export function formatAed(value: string | number | null): string {
  const n = parseAmount(value);
  if (n !== null) return `AED ${money.format(n)}`;
  if (typeof value === "string" && value.trim()) return value.trim();
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return "—";
}

/** Grouped integer count (vendor invoice totals, approval limits) — the same thousands-separator
 * convention as every AED amount in the app. `null` (apx/api does not aggregate this yet, see
 * lib/api/types.ts's `Vendor.invoiceCount`) renders as the same "—" every other missing value in
 * this app uses. */
export function formatCount(n: number | null): string {
  return n === null ? "—" : count.format(n);
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

// "22 Sep, 15:45:07" for the audit trail, where events seconds apart need telling apart. Pinned
// to Latin digits and Gulf time like every other timestamp here, not the browser's locale.
const auditTime = new Intl.DateTimeFormat("en-AE-u-nu-latn", {
  day: "numeric",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
  timeZone: "Asia/Dubai",
});

export function formatAuditTime(iso: string): string {
  return auditTime.format(new Date(iso));
}

export function formatReceivedCompact(iso: string): string {
  return dateTimeCompact.format(new Date(iso)).replace(",", "");
}

export function confidenceTone(value: number | null): "success" | "warning" | "destructive" {
  if (value === null) return "warning";
  if (value >= 0.85) return "success";
  if (value >= 0.6) return "warning";
  return "destructive";
}
