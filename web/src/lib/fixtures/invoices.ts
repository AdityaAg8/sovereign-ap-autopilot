/** Generates a deterministic set of fixture invoices (summaries + full detail with grounded
 * fields, boxes and fraud flags) so the inbox, review queue and invoice-detail screens have
 * realistic, virtualization-scale data before apx/api (M5.1-M5.3) exists. Seeded PRNG keeps the
 * output stable across dev, Playwright and Lighthouse runs (PRD §10 M5.4). */
import type {
  BBox,
  Flag,
  FieldValue,
  InvoiceDetail,
  InvoiceStatus,
  RouteDecision,
} from "@/lib/api/types";
import { FIELD_LABELS } from "@/lib/invoice-fields";
import { fixtureDocumentIban, FIXTURE_VENDORS } from "@/lib/fixtures/vendors";
import { intBetween, mulberry32, pick } from "@/lib/fixtures/random";

const SEED = 20260101;
const COUNT = 420;

const STATUS_WEIGHTS: readonly {
  status: InvoiceStatus;
  route: RouteDecision | null;
  weight: number;
}[] = [
  { status: "awaiting_review", route: "review", weight: 5 },
  { status: "approved", route: "auto", weight: 4 },
  { status: "sent", route: "auto", weight: 3 },
  { status: "archived", route: "auto", weight: 3 },
  { status: "blocked", route: "block", weight: 1 },
  { status: "received", route: null, weight: 1 },
  { status: "validated", route: null, weight: 1 },
  { status: "fraud_checked", route: null, weight: 1 },
];

const FLAG_LIBRARY: readonly Omit<Flag, "id">[] = [
  {
    rule: "duplicate_invoice",
    severity: "high",
    explanation: "Same invoice number, vendor and amount as an invoice received 6 days ago.",
  },
  {
    rule: "bank_change",
    severity: "high",
    explanation: "Vendor's IBAN differs from the verified bank details on file.",
  },
  {
    rule: "new_vendor_high_amount",
    severity: "medium",
    explanation: "First invoice from this vendor is above AED 100,000.",
  },
  {
    rule: "split_invoice",
    severity: "medium",
    explanation:
      "Three invoices from this vendor this week each sit just under the approval limit.",
  },
  {
    rule: "price_variance",
    severity: "low",
    explanation: "Unit price is 8% above the matched purchase order line.",
  },
];

interface FieldSpec {
  name: string;
  page: number;
  box: BBox;
  sample: readonly string[];
}

const FIELD_SPECS: readonly FieldSpec[] = [
  {
    name: "invoice_number",
    page: 0,
    box: { page: 0, x0: 0.62, y0: 0.08, x1: 0.92, y1: 0.11 },
    sample: ["INV-2026-", "AR-", "GT-"],
  },
  {
    name: "issue_date",
    page: 0,
    box: { page: 0, x0: 0.62, y0: 0.12, x1: 0.92, y1: 0.15 },
    sample: ["2026-01-", "2026-02-", "2026-03-"],
  },
  {
    name: "supplier_name",
    page: 0,
    box: { page: 0, x0: 0.08, y0: 0.08, x1: 0.5, y1: 0.12 },
    sample: [],
  },
  {
    name: "supplier_trn",
    page: 0,
    box: { page: 0, x0: 0.08, y0: 0.13, x1: 0.35, y1: 0.16 },
    sample: [],
  },
  {
    name: "buyer_name",
    page: 0,
    box: { page: 0, x0: 0.08, y0: 0.2, x1: 0.45, y1: 0.23 },
    sample: ["Sovereign Manufacturing FZE"],
  },
  {
    name: "line_item_description",
    page: 0,
    box: { page: 0, x0: 0.08, y0: 0.36, x1: 0.6, y1: 0.39 },
    sample: [
      "Structural steel beams — grade 50",
      "A4 copier paper, 80gsm, 500 sheets",
      "Quarterly HVAC maintenance",
    ],
  },
  {
    name: "subtotal",
    page: 0,
    box: { page: 0, x0: 0.68, y0: 0.72, x1: 0.92, y1: 0.75 },
    sample: [],
  },
  {
    name: "vat_amount",
    page: 0,
    box: { page: 0, x0: 0.68, y0: 0.76, x1: 0.92, y1: 0.79 },
    sample: [],
  },
  {
    name: "total",
    page: 0,
    box: { page: 0, x0: 0.68, y0: 0.8, x1: 0.92, y1: 0.84 },
    sample: [],
  },
  {
    name: "supplier_iban",
    page: 0,
    box: { page: 0, x0: 0.08, y0: 0.88, x1: 0.5, y1: 0.91 },
    sample: [],
  },
];

function money(rng: () => number, min: number, max: number): string {
  return (min + rng() * (max - min)).toFixed(2);
}

function buildFields(
  rng: () => number,
  vendor: (typeof FIXTURE_VENDORS)[number],
  forceLowConfidence: boolean,
) {
  const subtotal = Number(money(rng, 500, 480000));
  const vat = Number((subtotal * 0.05).toFixed(2));
  const total = (subtotal + vat).toFixed(2);
  const fields: FieldValue[] = FIELD_SPECS.map((spec) => {
    const confidence = forceLowConfidence ? 0.4 + rng() * 0.35 : 0.9 + rng() * 0.099;
    let raw: string;
    switch (spec.name) {
      case "invoice_number":
        raw = `${pick(rng, spec.sample)}${intBetween(rng, 1000, 9999)}`;
        break;
      case "issue_date":
        raw = `${pick(rng, spec.sample)}${String(intBetween(rng, 1, 28)).padStart(2, "0")}`;
        break;
      case "supplier_name":
        raw = vendor.nameEn;
        break;
      case "supplier_trn":
        raw = vendor.trn ?? "";
        break;
      case "buyer_name":
        raw = "Sovereign Manufacturing FZE";
        break;
      case "line_item_description":
        raw = pick(rng, spec.sample);
        break;
      case "subtotal":
        raw = subtotal.toFixed(2);
        break;
      case "vat_amount":
        raw = vat.toFixed(2);
        break;
      case "total":
        raw = total;
        break;
      case "supplier_iban":
        raw = fixtureDocumentIban(vendor.id);
        break;
      default:
        raw = "";
    }
    const accepted = confidence >= 0.85;
    return {
      name: spec.name,
      raw,
      normalized: raw,
      wordIds: [`${spec.name}-w0`],
      boxes: [spec.box],
      confidence: Number(confidence.toFixed(3)),
      accepted,
      source: accepted ? (rng() > 0.5 ? "extractor_a" : "extractor_b") : "arbiter",
    } satisfies FieldValue;
  });
  return { fields, total, subtotal: subtotal.toFixed(2), vat: vat.toFixed(2) };
}

interface GeneratedInvoice {
  detail: InvoiceDetail;
}

function generate(): readonly GeneratedInvoice[] {
  const rng = mulberry32(SEED);
  const totalWeight = STATUS_WEIGHTS.reduce((sum, s) => sum + s.weight, 0);
  const out: GeneratedInvoice[] = [];

  for (let i = 0; i < COUNT; i++) {
    const vendor = pick(rng, FIXTURE_VENDORS);
    let roll = rng() * totalWeight;
    const bucket = STATUS_WEIGHTS.find((s) => (roll -= s.weight) <= 0) ?? STATUS_WEIGHTS[0]!;
    const forceLowConfidence = bucket.status === "awaiting_review" && rng() > 0.4;
    const { fields, total } = buildFields(rng, vendor, forceLowConfidence);

    const flagCount = bucket.status === "blocked" ? intBetween(rng, 1, 2) : rng() > 0.85 ? 1 : 0;
    const flags: Flag[] = Array.from({ length: flagCount }, (_, idx) => {
      const template = pick(rng, FLAG_LIBRARY);
      return { id: `flag-${i}-${idx}`, ...template };
    });

    const day = intBetween(rng, 1, 27);
    const month = intBetween(rng, 1, 9);
    const receivedAt = new Date(
      Date.UTC(2026, month - 1, day, intBetween(rng, 6, 18), intBetween(rng, 0, 59)),
    );
    // Every fixture field sets a real confidence (never null — that's only possible from the
    // http adapter, apx.api.interfaces.InvoiceFieldView.confidence being `float | None`), so the
    // `?? 0` here never actually fires; it only satisfies FieldValue's shared, nullable type.
    const meanConfidence = fields.reduce((sum, f) => sum + (f.confidence ?? 0), 0) / fields.length;

    const id = `inv-${SEED}-${i.toString().padStart(4, "0")}`;
    out.push({
      detail: {
        summary: {
          id,
          status: bucket.status,
          route: bucket.route,
          vendorId: vendor.id,
          vendorName: vendor.nameEn,
          total,
          currency: "AED",
          confidence: Number(meanConfidence.toFixed(3)),
          flagCount: flags.length,
          receivedAt: receivedAt.toISOString(),
        },
        fields,
        flags,
        reasons:
          flags.length > 0
            ? flags.map((f) => f.explanation)
            : forceLowConfidence
              ? ["Two or more fields fell below the auto-accept confidence threshold."]
              : [],
        proposals: [],
        pageCount: rng() > 0.85 ? 2 : 1,
        language: rng() > 0.75 ? "ar" : rng() > 0.5 ? "bilingual" : "en",
        // Every fixture invoice points at the same bundled sample PDF (web/public/fixtures) so
        // the pdf.js viewer always has a real document to render in dev and Playwright e2e; its
        // 2 pages cover both the 1- and 2-page cases above.
        documentUrl: "/fixtures/sample-invoice.pdf",
      },
    });
  }
  return out;
}

let generated: readonly InvoiceDetail[] | null = null;

/** ID of the fixed ADR-0012 review case below. */
export const DISAGREEMENT_INVOICE_ID = "inv-adr12-disagreement";

/** An A/B disagreement on the classified fields (ADR-0012 Amendment 1): the document type and
 * both lines' VAT category and rate came back unresolved, so approval is refused (422) until a
 * reviewer fills them in. Received early in the year so it never heads the inbox or queue that
 * other e2e specs pick their "first row" from. */
function disagreementInvoice(base: InvoiceDetail): InvoiceDetail {
  const unresolved = (name: string): FieldValue => ({
    name,
    raw: null,
    normalized: null,
    wordIds: [],
    boxes: [],
    confidence: 0.31,
    accepted: false,
    source: "arbiter",
  });
  const currency: FieldValue = {
    name: "currency",
    raw: "AED",
    normalized: "AED",
    wordIds: ["currency-w0"],
    boxes: [{ page: 0, x0: 0.62, y0: 0.16, x1: 0.72, y1: 0.19 }],
    confidence: 0.97,
    accepted: true,
    source: "extractor_a",
  };
  return {
    ...base,
    summary: {
      ...base.summary,
      id: DISAGREEMENT_INVOICE_ID,
      status: "awaiting_review",
      route: "review",
      flagCount: 0,
      receivedAt: "2026-01-01T06:00:00.000Z",
    },
    fields: [
      ...base.fields,
      unresolved("invoice_type_code"),
      currency,
      unresolved("line_items[1].vat_category"),
      unresolved("line_items[1].vat_rate"),
      unresolved("line_items[2].vat_category"),
      unresolved("line_items[2].vat_rate"),
    ],
    flags: [],
    reasons: ["Extractors A and B disagree on the document type and the lines' VAT treatment."],
  };
}

/** The ~420 generated fixture invoices plus the ADR-0012 disagreement case, built on first call
 * and memoized. Generation is deferred (not run at module load) because every route imports this
 * module via the auth provider, and running it eagerly cost ~140ms of main-thread time before
 * first paint on /login under Lighthouse's mobile CPU throttling — on pages that never show an
 * invoice. */
export function fixtureInvoices(): readonly InvoiceDetail[] {
  if (generated === null) {
    const details = generate().map((g) => g.detail);
    generated = [...details, disagreementInvoice(details[0]!)];
  }
  return generated;
}

export function findInvoice(id: string): InvoiceDetail | undefined {
  return fixtureInvoices().find((inv) => inv.summary.id === id);
}

// Fails fast in dev if a generated field name has no UI label (lib/invoice-fields.ts).
if (process.env.NODE_ENV !== "production") {
  for (const spec of FIELD_SPECS) {
    if (!FIELD_LABELS[spec.name])
      throw new Error(`Fixture field "${spec.name}" has no label in lib/invoice-fields.ts`);
  }
}
