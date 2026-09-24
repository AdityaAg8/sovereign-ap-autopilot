/** Dev-only ApiClient adapter backed by in-memory fixture data (PRD §10 M5.4). Used whenever
 * NEXT_PUBLIC_API_MODE is unset or "fixture" — the default in local dev and in the Playwright
 * smoke suite, so e2e can run before apx/api exists and later flip to the HTTP adapter unchanged. */
import type { ApiClient } from "@/lib/api/contract";
import { ApiError } from "@/lib/api/contract";
import type {
  AuditEventView,
  ChatReply,
  ChatStep,
  ChatTurn,
  DecisionRequest,
  FieldEdit,
  FieldValue,
  InvoiceDetail,
  InvoiceSummary,
  InvoiceStatus,
  ListInvoicesParams,
  OpsSnapshot,
  Session,
  SseEvent,
  Vendor,
} from "@/lib/api/types";
import { fixtureAssistant, fixtureAudit } from "@/lib/fixtures/assistant";
import { fixtureInvoices } from "@/lib/fixtures/invoices";
import { findVendor, FIXTURE_VENDORS } from "@/lib/fixtures/vendors";
import { findUserByEmail } from "@/lib/fixtures/users";
import {
  CURRENCY_FIELD,
  INVOICE_TYPE_FIELD,
  lineFieldName,
  parseLineField,
  VAT_CATEGORY_OPTIONS,
} from "@/lib/invoice-fields";

/** Simulated network latency; 0 under Playwright so assertions don't need arbitrary waits. */
const LATENCY_MS = process.env.NEXT_PUBLIC_E2E === "1" ? 0 : 120;

function delay<T>(value: T): Promise<T> {
  return LATENCY_MS === 0
    ? Promise.resolve(value)
    : new Promise((r) => setTimeout(() => r(value), LATENCY_MS));
}

// Mutable copy so approve/reject/edit-field calls persist for the lifetime of the tab. A shallow
// copy is enough -- and far cheaper at startup than `structuredClone`, which was doing a full
// recursive clone of all ~420 invoices' field/box arrays on every page load (a measured multi-
// -second main-thread cost under Lighthouse's mobile CPU throttling, since every route that
// calls getApiClient() imports this module). `editField`/`decide` below only ever replace a
// whole top-level key (`{ ...invoice, fields }` / `{ ...invoice, summary: {...} }`), never mutate
// a nested fields/flags/proposals array in place, so nothing here ever reaches back into
// `fixtureInvoices()`'s own arrays. Built lazily on the first data call, so pages that never
// read invoices (e.g. /login) never pay for generating them.
let storeCache: Map<string, InvoiceDetail> | null = null;
function store(): Map<string, InvoiceDetail> {
  storeCache ??= new Map(fixtureInvoices().map((inv) => [inv.summary.id, { ...inv }]));
  return storeCache;
}

const sessions = new Map<string, Session>();

/** Stages a document passes through before it settles, in pipeline order. */
const FLOW: readonly InvoiceStatus[] = [
  "received",
  "scanned",
  "parsed",
  "extracted",
  "gated",
  "validated",
  "fraud_checked",
];
const ALL_STAGES: readonly InvoiceStatus[] = [
  ...FLOW,
  "awaiting_review",
  "approved",
  "sent",
  "archived",
  "blocked",
  "failed",
];
const TICK_MS = 2000;

/** Synthetic arrivals for tick `t`: deterministic (a hash of the tick, so every tab agrees), with
 * quiet ticks, single arrivals and the odd burst of three, like a real inbox. */
function arrivalsAt(t: number): number {
  const h = Math.imul(t, 2654435761) >>> 0;
  if (h % 7 === 0) return 3;
  return h % 3 === 0 ? 0 : 1;
}

/** Dev-mode stand-in for GET /ops/snapshot: the stored invoices plus synthetic documents that
 * arrive on an uneven schedule and advance one stage per 2 s tick, and the model llama-swap would hold for the stage they
 * are in -- so the operations view moves in fixture mode the way it does against the API. */
function fixtureOpsSnapshot(now: number): OpsSnapshot {
  const tick = Math.floor(now / TICK_MS);
  const invoices = [...store().values()].map((inv) => inv.summary);
  const counts = new Map<InvoiceStatus, number>(ALL_STAGES.map((s) => [s, 0]));
  for (const inv of invoices) counts.set(inv.status, (counts.get(inv.status) ?? 0) + 1);
  const flight = FLOW.flatMap((stage, age) =>
    Array.from({ length: arrivalsAt(tick - age) }, () => stage),
  );
  for (const stage of flight) counts.set(stage, (counts.get(stage) ?? 0) + 1);

  const extracting = flight.includes("extracted") || flight.includes("parsed");
  const models = extracting
    ? [
        { model: "extractor-a", state: tick % 4 === 0 ? "starting" : "ready" },
        { model: "extractor-b", state: "ready" },
      ]
    : flight.includes("gated")
      ? [{ model: "arbiter", state: "ready" }]
      : [];

  const routed = invoices.filter((inv) => inv.route !== null);
  const fields = [...store().values()].flatMap((inv) => inv.fields);
  const automated = fields.filter((f) => f.source !== "reviewer");
  return {
    generatedAt: new Date(now).toISOString(),
    kpis: {
      stpRate: routed.length
        ? routed.filter((i) => i.route === "auto").length / routed.length
        : null,
      routedInvoices: routed.length,
      autoAcceptErrorRate: automated.length
        ? automated.filter((f) => !f.accepted).length / automated.length
        : null,
      automatedFields: automated.length,
      llmLatencyP50Ms: 21400 + (tick % 7) * 310,
      llmLatencyP95Ms: 61800 + (tick % 5) * 900,
      llmCalls: 1480 + (tick % 1000),
      e2eLatencyP50S: 48.2,
      attacksBlocked: invoices.filter((i) => i.status === "blocked").length,
      queueSize: counts.get("awaiting_review") ?? 0,
    },
    stages: ALL_STAGES.map((stage) => ({ stage, count: counts.get(stage) ?? 0 })),
    recent: [...store().values()]
      .sort((a, b) => b.summary.receivedAt.localeCompare(a.summary.receivedAt))
      .slice(0, 12)
      .map(({ summary: s, fields: f }) => ({
        id: s.id,
        number: f.find((field) => field.name === "invoice_number")?.raw ?? s.id.slice(0, 12),
        vendorName: s.vendorName,
        status: s.status,
        route: s.route,
        total: s.total ?? "0.00",
        currency: s.currency,
        updatedAt: s.receivedAt,
      })),
    models,
    modelsReachable: true,
  };
}

const CURRENCY_CODE = /^[A-Z]{3}$/;
const VAT_CATEGORIES: ReadonlySet<string> = new Set(VAT_CATEGORY_OPTIONS.map((o) => o.value));

/** Stand-in for apx.workflows.corrections.corrections_from_fields: the classified fields an
 * approval would be refused over (the API's 422 MissingCorrectionsError), in the server's order.
 * Only invoices that carry ADR-0012 field rows are checked; the legacy fixture set has none. */
function unresolvedClassifiedFields(fields: readonly FieldValue[]): string[] {
  const values = new Map(fields.map((f) => [f.name, f.raw?.trim() ?? ""]));
  const indices = [
    ...new Set(
      fields.flatMap((f) => (parseLineField(f.name) ? [parseLineField(f.name)!.index] : [])),
    ),
  ].sort((a, b) => a - b);
  if (!values.has(INVOICE_TYPE_FIELD) && indices.length === 0) return [];
  const problems: string[] = [];
  if (!["380", "381"].includes(values.get(INVOICE_TYPE_FIELD) ?? ""))
    problems.push(INVOICE_TYPE_FIELD);
  const currency = (values.get(CURRENCY_FIELD) ?? "").toUpperCase();
  if (!CURRENCY_CODE.test(currency) || currency === "XXX") problems.push(CURRENCY_FIELD);
  if (indices.length === 0) problems.push("line_items[*].vat_category");
  for (const index of indices) {
    const category = values.get(lineFieldName(index, "vat_category")) ?? "";
    const rateText = (values.get(lineFieldName(index, "vat_rate")) ?? "").replace(/%$/, "").trim();
    const rate = rateText === "" ? NaN : Number(rateText);
    if (!VAT_CATEGORIES.has(category)) problems.push(lineFieldName(index, "vat_category"));
    if (!Number.isFinite(rate) || rate < 0) problems.push(lineFieldName(index, "vat_rate"));
  }
  return problems;
}

class FixtureApiClient implements ApiClient {
  async login(email: string, _password: string): Promise<Session> {
    const user = findUserByEmail(email);
    if (!user) throw new ApiError("No account matches that email.", 401);
    const token = `fixture-token.${user.id}.${Date.now()}`;
    const session: Session = { user, token };
    sessions.set(token, session);
    return delay(session);
  }

  async restoreSession(session: Session): Promise<boolean> {
    const user = findUserByEmail(session.user.email);
    if (!user || !session.token.startsWith(`fixture-token.${user.id}.`)) return false;
    sessions.set(session.token, { user, token: session.token });
    return true;
  }

  async logout(token: string): Promise<void> {
    sessions.delete(token);
    return delay(undefined);
  }

  async listInvoices(params: ListInvoicesParams): Promise<readonly InvoiceSummary[]> {
    let rows = Array.from(store().values()).map((inv) => inv.summary);
    if (params.status) rows = rows.filter((r) => r.status === params.status);
    if (params.route) rows = rows.filter((r) => r.route === params.route);
    if (params.vendorId) rows = rows.filter((r) => r.vendorId === params.vendorId);
    rows = [...rows].sort((a, b) => b.receivedAt.localeCompare(a.receivedAt));
    if (params.limit) rows = rows.slice(0, params.limit);
    return delay(rows);
  }

  async getInvoice(id: string): Promise<InvoiceDetail> {
    const invoice = store().get(id);
    if (!invoice) throw new ApiError(`Invoice ${id} not found.`, 404);
    return delay(invoice);
  }

  async editField(invoiceId: string, edit: FieldEdit): Promise<void> {
    const invoice = store().get(invoiceId);
    if (!invoice) throw new ApiError(`Invoice ${invoiceId} not found.`, 404);
    const fields = invoice.fields.map((f) =>
      f.name === edit.field
        ? {
            ...f,
            raw: edit.value,
            normalized: edit.value,
            accepted: true,
            confidence: 1,
            source: "reviewer" as const,
          }
        : f,
    );
    store().set(invoiceId, { ...invoice, fields });
    return delay(undefined);
  }

  async decide(invoiceId: string, request: DecisionRequest): Promise<void> {
    const invoice = store().get(invoiceId);
    if (!invoice) throw new ApiError(`Invoice ${invoiceId} not found.`, 404);
    const statusByAction = {
      approve: "approved",
      reject: "blocked",
      send_to_review: "awaiting_review",
      execute_proposal: invoice.summary.status,
    } as const;
    if (request.action === "approve") {
      const unresolved = unresolvedClassifiedFields(invoice.fields);
      if (unresolved.length > 0)
        throw new ApiError(
          `cannot approve: correct these fields first: ${unresolved.join(", ")}`,
          422,
        );
    }
    const nextStatus = statusByAction[request.action];
    store().set(invoiceId, {
      ...invoice,
      summary: { ...invoice.summary, status: nextStatus },
    });
    return delay(undefined);
  }

  async getOpsSnapshot(): Promise<OpsSnapshot> {
    return delay(fixtureOpsSnapshot(Date.now()));
  }

  async listVendors(): Promise<readonly Vendor[]> {
    return delay(FIXTURE_VENDORS);
  }

  async getVendor(id: string): Promise<Vendor> {
    const vendor = findVendor(id);
    if (!vendor) throw new ApiError(`Vendor ${id} not found.`, 404);
    return delay(vendor);
  }

  async getInvoiceAudit(id: string): Promise<readonly AuditEventView[]> {
    const invoice = store().get(id);
    if (!invoice) throw new ApiError(`invoice ${id} not found`, 404);
    return delay(fixtureAudit(invoice));
  }

  async askAssistant(
    history: readonly ChatTurn[],
    onStep: (step: ChatStep) => void,
    signal: AbortSignal,
  ): Promise<ChatReply> {
    const pause = (ms: number) =>
      new Promise<void>((resolve, reject) => {
        if (signal.aborted) return reject(new ApiError("cancelled", 499));
        setTimeout(resolve, LATENCY_MS === 0 ? 0 : ms);
      });
    return fixtureAssistant(history, store().values(), onStep, pause);
  }

  subscribeEvents(_invoiceId: string | null, _onEvent: (event: SseEvent) => void): () => void {
    // The fixture adapter has no server pushing SSE events; the HTTP adapter fulfils this once
    // apx/api's event stream exists. Returning a no-op keeps callers identical across adapters.
    return () => {};
  }
}

export const fixtureApiClient: ApiClient = new FixtureApiClient();
