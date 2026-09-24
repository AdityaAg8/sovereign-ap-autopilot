/** Human-readable labels for extracted field names (apx.extract.schema.InvoiceFields, PRD §10
 * M3.2). Shared by the fixture generator and the field-list UI so both draw from one source
 * instead of duplicating the field vocabulary. */

export const FIELD_LABELS: Readonly<Record<string, string>> = {
  invoice_number: "Invoice number",
  issue_date: "Issue date",
  supplier_name: "Supplier name",
  supplier_trn: "Supplier TRN",
  buyer_name: "Buyer name",
  line_item_description: "Line item 1",
  subtotal: "Subtotal (AED)",
  vat_amount: "VAT 5% (AED)",
  total: "Total (AED)",
  supplier_iban: "Supplier IBAN",
};

export function fieldLabel(name: string): string {
  return FIELD_LABELS[name] ?? classifiedLabel(name) ?? name.replaceAll("_", " ");
}

/** Free-text/name fields that must render in the UI sans face, never mono (web/DESIGN.md: "mono
 * face … for amounts, dates, TRNs, IBANs, invoice IDs and timestamps" — company names and
 * descriptions are prose, not tabular data, and setting them in mono reads as a design mistake,
 * not a deliberate one). Every other field (ids, dates, money, TRN, IBAN) is mono by default. */
const TEXT_FIELDS: ReadonlySet<string> = new Set([
  "supplier_name",
  "buyer_name",
  "line_item_description",
]);

export function isMonoField(name: string): boolean {
  return !TEXT_FIELDS.has(name);
}

/** Money-valued fields whose raw extracted text (a plain decimal like "182345.67" — apx never
 * puts grouping separators in a stored value) must still render grouped, matching every other
 * AED amount in the app (inbox/vendor tables' `formatAed`) instead of the one inconsistent,
 * ungrouped-looking number on the field list. */
const MONEY_FIELDS: ReadonlySet<string> = new Set(["subtotal", "vat_amount", "total"]);

export function isMoneyField(name: string): boolean {
  return MONEY_FIELDS.has(name);
}

/* ADR-0012 Amendment 1 classified fields: the document type, the currency and each line's VAT
 * category and rate are ordinary field rows a reviewer corrects before approval. Names mirror
 * apx.workflows.corrections (line indices are 1-based over Invoice.lines). */

export const INVOICE_TYPE_FIELD = "invoice_type_code";
export const CURRENCY_FIELD = "currency";

export const INVOICE_TYPE_OPTIONS: readonly { value: string; label: string }[] = [
  { value: "380", label: "Invoice" },
  { value: "381", label: "Credit note" },
];

export const VAT_CATEGORY_OPTIONS: readonly { value: string; label: string }[] = [
  { value: "S", label: "Standard" },
  { value: "Z", label: "Zero-rated" },
  { value: "E", label: "Exempt" },
  { value: "AE", label: "Reverse charge" },
  { value: "O", label: "Out of scope" },
];

const LINE_FIELD = /^line_items\[(\d+)\]\.(vat_category|vat_rate)$/;

export interface LineFieldRef {
  index: number;
  attr: "vat_category" | "vat_rate";
}

export function parseLineField(name: string): LineFieldRef | null {
  const m = LINE_FIELD.exec(name);
  return m ? { index: Number(m[1]), attr: m[2] as LineFieldRef["attr"] } : null;
}

export function lineFieldName(index: number, attr: LineFieldRef["attr"]): string {
  return `line_items[${index}].${attr}`;
}

const CLASSIFIED_LABELS: Readonly<Record<string, string>> = {
  [INVOICE_TYPE_FIELD]: "Document type",
  [CURRENCY_FIELD]: "Currency",
  vat_category: "VAT category",
  vat_rate: "VAT rate (%)",
};

/** Label for a classified field; a line field's label omits the line, which its group heading
 * carries. `withLine` adds it back for contexts outside the group (the approve error). */
export function classifiedLabel(name: string, withLine = false): string | null {
  const line = parseLineField(name);
  if (line) {
    const label = CLASSIFIED_LABELS[line.attr]!;
    return withLine ? `Line ${line.index} ${label}` : label;
  }
  return CLASSIFIED_LABELS[name] ?? null;
}

export function optionLabel(
  options: readonly { value: string; label: string }[],
  value: string | null,
): string | null {
  const hit = options.find((o) => o.value === value);
  return hit ? `${hit.value} · ${hit.label}` : null;
}

/** Allowed VAT rates per category (PINT AE; AE may state 0 or 5, ADR-0012 Amendment 1). */
export function vatRateGuidance(category: string | null): string {
  switch (category) {
    case "S":
      return "Standard rate: 5";
    case "Z":
    case "E":
    case "O":
      return "This category takes 0";
    case "AE":
      return "Reverse charge: 0 or 5";
    default:
      return "S = 5 · Z, E, O = 0 · AE = 0 or 5";
  }
}

/** A currency value the backend treats as missing: blank, not three letters, or "XXX". */
export function isMissingCurrency(raw: string | null): boolean {
  const v = (raw ?? "").trim().toUpperCase();
  return !/^[A-Z]{3}$/.test(v) || v === "XXX";
}

const UNRESOLVED_PREFIX = "correct these fields first:";

/** Field names from the decisions endpoint's 422 detail (MissingCorrectionsError). */
export function parseUnresolvedFields(detail: string): readonly string[] {
  const at = detail.indexOf(UNRESOLVED_PREFIX);
  if (at < 0) return [];
  return detail
    .slice(at + UNRESOLVED_PREFIX.length)
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}
