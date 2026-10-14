/** Wire types for the review UI's data layer. Mirrors apx/api/interfaces.py, apx/common/types.py,
 * apx/trust/interfaces.py and apx/policy/interfaces.py so the fixture
 * adapter and the future HTTP adapter share one contract with the FastAPI service. Fields marked
 * "UI extension" are not yet on the Python `Frozen` models above but are the values M5.3 will
 * attach once the trust gate and policy engine are wired into the API responses; the fixture
 * adapter populates them today so the confidence/flag UI can be built ahead of the backend. */

export type Role = "reviewer" | "approver" | "admin";

export type InvoiceStatus =
  | "received"
  | "scanned"
  | "parsed"
  | "extracted"
  | "gated"
  | "validated"
  | "fraud_checked"
  | "awaiting_review"
  | "approved"
  | "sent"
  | "archived"
  | "blocked"
  | "failed";

export type RouteDecision = "auto" | "review" | "block";

export interface BBox {
  page: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface FieldValue {
  name: string;
  raw: string | null;
  normalized: string | null;
  wordIds: readonly string[];
  boxes: readonly BBox[];
  /** UI extension: apx.trust.interfaces.FieldDecision.p_correct, 0..1. Null when the backend
   * hasn't recorded a confidence for this field (apx.api.interfaces.InvoiceFieldView.confidence
   * is `float | None` — M5.1's schema has no trust-decision table yet). */
  confidence: number | null;
  /** UI extension: apx.trust.interfaces.FieldDecision.accepted (auto-accepted by the trust gate). */
  accepted: boolean;
  /** UI extension: which extractor/arbiter produced the accepted value, for the "source" column. */
  source: "extractor_a" | "extractor_b" | "arbiter" | "reviewer";
}

export interface Flag {
  id: string;
  rule: string;
  severity: "low" | "medium" | "high";
  explanation: string;
}

export interface InvoiceSummary {
  id: string;
  /** UI extension: the supplier's own invoice number (the extracted `invoice_number` field), the
   * number an AP clerk actually searches by. apx.api.interfaces.InvoiceSummary does not carry it
   * yet, so the http adapter reports null rather than a guess; the inbox then shows the internal
   * `id`, visibly marked as such (inbox-table.tsx's `InvoiceRef`). */
  invoiceNumber: string | null;
  status: InvoiceStatus;
  route: RouteDecision | null;
  /** Null when the backend hasn't resolved a vendor for this invoice.
   * apx.api.interfaces.InvoiceSummary.vendor_id — populated by the http
   * adapter now that the field exists on the wire. */
  vendorId: string | null;
  /** Null for the same reasons as `vendorId` — apx.api.interfaces.InvoiceSummary calls this
   * `supplier_name` and it is optional even when a vendor is resolved. */
  vendorName: string | null;
  total: string | null;
  currency: string;
  /** UI extension: mean p_correct across fields, surfaced as the inbox confidence column. Only
   * the fixture adapter can compute this without an extra round trip per row; the http adapter
   * reports null until apx/api aggregates it server-side. */
  confidence: number | null;
  flagCount: number;
  receivedAt: string;
}

export interface InvoiceDetail {
  summary: InvoiceSummary;
  fields: readonly FieldValue[];
  flags: readonly Flag[];
  reasons: readonly string[];
  proposals: readonly Proposal[];
  pageCount: number;
  language: "en" | "ar" | "bilingual";
  /** URL of the source document for the pdf.js viewer, or null when none is available. The
   * fixture adapter always serves a bundled sample PDF; the http adapter points this at
   * `GET /invoices/{id}/document`, which needs the same bearer token as
   * every other apx/api route — callers pass the session token through to `PdfPane`/`PdfCanvas`,
   * which forward it as pdf.js's `httpHeaders` (see pdf-canvas.tsx). */
  documentUrl: string | null;
}

export interface FieldEdit {
  field: string;
  /** At most 1,000 characters (the API answers 422 beyond that); a reviewer's supplier_address
   * is refused on approval above 500 characters or with control characters (ADR-0012 amdt 4). */
  value: string;
  reason: string;
}

export type DecisionAction = "approve" | "reject" | "send_to_review" | "execute_proposal";

export interface DecisionRequest {
  action: DecisionAction;
  proposalId?: string;
  comment?: string;
}

export interface Proposal {
  id: string;
  invoiceId: string | null;
  kind: "edit_field" | "route" | "request_info" | "approve_payment" | "change_bank";
  payload: Record<string, unknown>;
  createdBy: "agent" | "chat" | string;
  createdAt: string;
  status: "pending" | "executed" | "rejected";
}

export interface SseEvent {
  event: string;
  data: Record<string, unknown>;
}

export interface BankDetail {
  id: string;
  /** Masked to the last 4 characters, always — apx.api.interfaces.VendorBankDetailView.iban_masked
   *: the vendor master-data view never returns a full IBAN to any role, unlike
   * the per-invoice field view's admin exception (ADR-0008 T2). */
  ibanMasked: string;
  /** UI-only decoration. apx.db.models.BankDetail has no bank-name column — the http adapter
   * always reports null, not a guess at one; the fixture adapter keeps its illustrative names. */
  bankName: string | null;
  /** The G-3 `Computed` column: true only if `verification_id` names a real
   * `bank_verifications` row. */
  verified: boolean;
  verifiedAt: string | null;
  isCurrent: boolean;
  changedAt: string;
}

export interface Vendor {
  id: string;
  nameEn: string;
  /** No Arabic name column on apx.db.models.Vendor today — the http adapter always reports
   * null; the fixture adapter keeps its bilingual names. */
  nameAr: string | null;
  trn: string | null;
  /** True if the vendor has at least one current, G-3-verified bank account
   * (apx.api.interfaces.VendorSummary.verified / VendorDetail derived from its bank_details). */
  verified: boolean;
  /** Populated on `getVendor`; empty on a `listVendors` row, which the backend keeps lean
   * (apx.api.routers.vendors.list_vendors returns `VendorSummary`, not the full detail) — use
   * `verified` above for the list view's status badge, not `bankDetails.every(...)`. */
  bankDetails: readonly BankDetail[];
  /** Not aggregated anywhere in apx/api yet — a documented gap, not a guessed count. */
  invoiceCount: number | null;
  /** Not aggregated anywhere in apx/api yet — a documented gap, not a guessed total. */
  totalAed: string | null;
  /** No such concept on the backend vendor master record — the http adapter always reports
   * false; the fixture adapter keeps its illustrative flag. */
  isNew: boolean;
}

export interface AuthUser {
  id: string;
  name: string;
  email: string;
  roles: readonly Role[];
  approvalLimitAed: number | null;
}

export interface Session {
  user: AuthUser;
  token: string;
}

export interface ListInvoicesParams {
  status?: InvoiceStatus;
  route?: RouteDecision;
  vendorId?: string;
  limit?: number;
}

/** apx.observability.dashboard.OpsSnapshot: the live operations view. */
export interface OpsKpis {
  stpRate: number | null;
  routedInvoices: number;
  autoAcceptErrorRate: number | null;
  automatedFields: number;
  llmLatencyP50Ms: number | null;
  llmLatencyP95Ms: number | null;
  llmCalls: number;
  e2eLatencyP50S: number | null;
  attacksBlocked: number;
  queueSize: number;
}

export interface OpsRecentInvoice {
  id: string;
  number: string;
  vendorName: string | null;
  status: InvoiceStatus;
  route: RouteDecision | null;
  total: string;
  currency: string;
  updatedAt: string;
}

export interface OpsSnapshot {
  generatedAt: string;
  kpis: OpsKpis;
  stages: readonly { stage: InvoiceStatus; count: number }[];
  recent: readonly OpsRecentInvoice[];
  /** Models llama-swap holds in memory right now; `state` is e.g. "ready" or "starting". */
  models: readonly { model: string; state: string }[];
  modelsReachable: boolean;
}

/** apx.api.routers.chat: one audit_log event on an invoice's audit tab. `facts`
 * holds only scalar, system-written values; `note` is the exception agent's review note. */
export interface AuditEventView {
  id: number;
  ts: string;
  actor: string;
  action: string;
  facts: Readonly<Record<string, string | number | boolean | null>>;
  note: string | null;
}

export type CitationSource = "invoice" | "audit" | "vendor" | "policy" | "document" | "proposal";

/** A citation in an assistant reply; `href` links to the cited record when it has a page. */
export interface ChatCitation {
  source: CitationSource;
  ref: string;
  href: string | null;
}

/** One tool call the assistant made while answering, shown live in the panel. */
export interface ChatStep {
  tool: string | null;
  label: string;
  outcome: "ok" | "refused";
}

export interface ChatReply {
  text: string;
  /** Set when the assistant declined (e.g. "unverifiable", "action_not_permitted"). */
  refusal: string | null;
  citations: readonly ChatCitation[];
  proposals: readonly { id: string; kind: string }[];
}

export interface ChatTurn {
  role: "user" | "assistant";
  text: string;
}
