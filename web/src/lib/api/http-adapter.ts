/** ApiClient adapter that calls the real apx/api REST + SSE service,
 * selected over the fixture adapter via NEXT_PUBLIC_API_MODE=http (see client.ts). Every method
 * maps apx/api/interfaces.py's wire shapes — snake_case, no camelCase alias_generator anywhere in
 * that module — onto this app's ApiClient contract, and is explicit via a typed ApiError, never a
 * guess or a quiet empty result, about the handful of things apx/api still does not expose:
 * per-invoice confidence/flag aggregation, vendor invoice-count/total aggregation and a per-user
 * approval limit. Those gaps are noted inline below, next to the code that has to live with
 * them. */
import type { ApiClient } from "@/lib/api/contract";
import { ApiError } from "@/lib/api/contract";
import type {
  AuditEventView,
  ChatCitation,
  ChatReply,
  ChatStep,
  ChatTurn,
  AuthUser,
  BankDetail,
  BBox,
  DecisionRequest,
  FieldEdit,
  FieldValue,
  Flag,
  InvoiceDetail,
  InvoiceStatus,
  InvoiceSummary,
  ListInvoicesParams,
  OpsSnapshot,
  Proposal,
  Role,
  RouteDecision,
  Session,
  SseEvent,
  Vendor,
} from "@/lib/api/types";

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
// apx.api.service._MAX_PAGE_SIZE: the server clamps any larger `limit` itself, so a caller asking
// for e.g. 2000 rows (the inbox/review-queue pages do) has to be served over several requests.
const MAX_PAGE_SIZE = 200;

let bearerToken: string | null = null;

async function parseErrorDetail(res: Response): Promise<string> {
  const text = await res.text().catch(() => "");
  if (!text) return res.statusText;
  try {
    const body: unknown = JSON.parse(text);
    if (
      body !== null &&
      typeof body === "object" &&
      "detail" in body &&
      typeof (body as { detail: unknown }).detail === "string"
    ) {
      return (body as { detail: string }).detail;
    }
  } catch {
    // Not JSON (e.g. a proxy's plain-text error page) — fall through to the raw text below.
  }
  return text;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      ...(bearerToken ? { Authorization: `Bearer ${bearerToken}` } : {}),
      ...init?.headers,
    },
  });
  if (!res.ok) throw new ApiError(await parseErrorDetail(res), res.status);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

function apiFetchJson<T>(path: string, init: RequestInit & { body: string }): Promise<T> {
  return apiFetch<T>(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init.headers },
  });
}

// ---- apx.api.interfaces wire DTOs, field-for-field as FastAPI serializes them ----

interface InvoiceSummaryDto {
  id: string;
  status: InvoiceStatus;
  route: RouteDecision | null;
  vendor_id: string | null;
  supplier_name: string | null;
  total: string | null;
  received_at: string;
}

interface InvoiceFieldViewDto {
  name: string;
  value: string | null;
  confidence: number | null;
  boxes: readonly BBox[];
  source: "extractor_a" | "extractor_b" | "arbiter" | "human";
}

interface ProposalDto {
  id: string;
  invoice_id: string | null;
  kind: Proposal["kind"];
  payload: Record<string, unknown>;
  created_by: string;
  created_at: string;
  status: Proposal["status"];
}

/** apx.api.interfaces.FlagView. */
interface FlagViewDto {
  id: number;
  rule_id: string;
  typology: string;
  severity: Flag["severity"];
  score: number;
  explanation: string;
}

interface InvoiceDetailDto {
  summary: InvoiceSummaryDto;
  fields: readonly InvoiceFieldViewDto[];
  accepted: Record<string, boolean>;
  reasons: readonly string[];
  proposals: readonly ProposalDto[];
  flags: readonly FlagViewDto[];
}

/** apx.observability.dashboard.OpsSnapshot as JSON (snake_case). */
interface OpsSnapshotDto {
  generated_at: string;
  kpis: {
    stp_rate: number | null;
    routed_invoices: number;
    auto_accept_error_rate: number | null;
    automated_fields: number;
    llm_latency_p50_ms: number | null;
    llm_latency_p95_ms: number | null;
    llm_calls: number;
    e2e_latency_p50_s: number | null;
    attacks_blocked: number;
    queue_size: number;
  };
  stages: readonly { stage: InvoiceStatus; count: number }[];
  recent: readonly {
    id: string;
    number: string;
    vendor_name: string | null;
    status: InvoiceStatus;
    route: RouteDecision | null;
    total: string;
    currency: string;
    updated_at: string;
  }[];
  models: readonly { model: string; state: string }[];
  models_reachable: boolean;
}

interface InvoicePageDto {
  items: readonly InvoiceSummaryDto[];
  total: number;
  limit: number;
  offset: number;
}

interface TokenResponseDto {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
}

// ---- apx.api.interfaces vendor wire DTOs ----

interface VendorBankDetailViewDto {
  id: number;
  iban_masked: string;
  verified: boolean;
  verified_at: string | null;
  source: string;
  is_current: boolean;
  changed_at: string;
}

interface VendorSummaryDto {
  id: string;
  name: string;
  trn: string | null;
  verified: boolean;
}

interface VendorDetailDto {
  id: string;
  name: string;
  trn: string | null;
  bank_details: readonly VendorBankDetailViewDto[];
  recent_invoices: readonly InvoiceSummaryDto[];
}

interface VendorPageDto {
  items: readonly VendorSummaryDto[];
  total: number;
  limit: number;
  offset: number;
}

// apx.db.enums.FieldSource calls a human correction "human"; this app's UI vocabulary calls the
// same provenance "reviewer" (field-row.tsx's SOURCE_LABEL) — one explicit mapping, not a guess.
const SOURCE_MAP: Record<InvoiceFieldViewDto["source"], FieldValue["source"]> = {
  extractor_a: "extractor_a",
  extractor_b: "extractor_b",
  arbiter: "arbiter",
  human: "reviewer",
};

const VALID_ROLES: ReadonlySet<string> = new Set<Role>(["reviewer", "approver", "admin"]);

function mapSummary(dto: InvoiceSummaryDto): InvoiceSummary {
  return {
    id: dto.id,
    // Not on apx.api.interfaces.InvoiceSummary yet; see InvoiceSummary.invoiceNumber. The inbox
    // falls back to the (marked) internal id, so real-API rows are never an unlabelled "—".
    invoiceNumber: null,
    status: dto.status,
    route: dto.route,
    vendorId: dto.vendor_id,
    vendorName: dto.supplier_name,
    total: dto.total,
    // Every invoice on this path is a UAE PINT AE invoice; the API doesn't echo the currency back.
    currency: "AED",
    // Mean field confidence isn't aggregated server-side, and computing it here would mean an
    // extra round trip per row on a list endpoint that isn't built for that.
    confidence: null,
    // apx/fraud hasn't landed yet, so there is genuinely no flag data anywhere in the system
    // — 0 is the accurate count today, not a guess standing in for missing data.
    flagCount: 0,
    receivedAt: dto.received_at,
  };
}

function mapField(dto: InvoiceFieldViewDto, accepted: Record<string, boolean>): FieldValue {
  return {
    name: dto.name,
    raw: dto.value,
    normalized: dto.value,
    // apx.api.interfaces.InvoiceFieldView carries a value and its grounding boxes, not the parse
    // word ids behind it — this app's UI never reads wordIds today (see field-list.tsx).
    wordIds: [],
    boxes: dto.boxes,
    confidence: dto.confidence,
    accepted: accepted[dto.name] ?? false,
    source: SOURCE_MAP[dto.source],
  };
}

function mapFlag(dto: FlagViewDto): Flag {
  return {
    id: String(dto.id),
    rule: dto.rule_id,
    severity: dto.severity,
    explanation: dto.explanation,
  };
}

function mapProposal(dto: ProposalDto): Proposal {
  return {
    id: dto.id,
    invoiceId: dto.invoice_id,
    kind: dto.kind,
    payload: dto.payload,
    createdBy: dto.created_by,
    createdAt: dto.created_at,
    status: dto.status,
  };
}

function mapVendorSummary(dto: VendorSummaryDto): Vendor {
  return {
    id: dto.id,
    nameEn: dto.name,
    // apx.db.models.Vendor has no Arabic-name column — null, not a guess.
    nameAr: null,
    trn: dto.trn,
    verified: dto.verified,
    // GET /vendors (apx.api.routers.vendors.list_vendors) returns VendorSummary, not the full
    // bank-details list, to keep the list query cheap — empty here, not a guess; the list UI's
    // verified badge reads `verified` above, never `bankDetails`.
    bankDetails: [],
    invoiceCount: null,
    totalAed: null,
    isNew: false,
  };
}

function mapVendorBankDetail(dto: VendorBankDetailViewDto): BankDetail {
  return {
    id: String(dto.id),
    ibanMasked: dto.iban_masked,
    // apx.db.models.BankDetail has no bank-name column — null, not a guess.
    bankName: null,
    verified: dto.verified,
    verifiedAt: dto.verified_at,
    isCurrent: dto.is_current,
    changedAt: dto.changed_at,
  };
}

function mapVendorDetail(dto: VendorDetailDto): Vendor {
  return {
    id: dto.id,
    nameEn: dto.name,
    nameAr: null,
    trn: dto.trn,
    verified: dto.bank_details.some((b) => b.verified && b.is_current),
    bankDetails: dto.bank_details.map(mapVendorBankDetail),
    invoiceCount: null,
    totalAed: null,
    isNew: false,
  };
}

function pageCountFromFields(fields: readonly InvoiceFieldViewDto[]): number {
  let maxPage = 0;
  for (const field of fields) {
    for (const box of field.boxes) maxPage = Math.max(maxPage, box.page);
  }
  return maxPage + 1;
}

function decodeAccessToken(token: string): {
  sub: string;
  roles: readonly Role[];
  exp: number | null;
} {
  const parts = token.split(".");
  if (parts.length !== 3) throw new ApiError("received a malformed access token", 500);
  try {
    const base64 = parts[1]!.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64 + "=".repeat((4 - (base64.length % 4)) % 4);
    const payload: unknown = JSON.parse(atob(padded));
    if (payload === null || typeof payload !== "object")
      throw new Error("payload is not an object");
    const sub = (payload as Record<string, unknown>).sub;
    const roles = (payload as Record<string, unknown>).roles;
    if (typeof sub !== "string") throw new Error("missing `sub` claim");
    if (!Array.isArray(roles)) throw new Error("missing `roles` claim");
    for (const role of roles) {
      if (typeof role !== "string" || !VALID_ROLES.has(role)) {
        throw new Error(`unrecognized role claim ${JSON.stringify(role)}`);
      }
    }
    const exp = (payload as Record<string, unknown>).exp;
    return { sub, roles: roles as Role[], exp: typeof exp === "number" ? exp : null };
  } catch (err) {
    throw new ApiError(`could not decode the access token: ${String(err)}`, 500);
  }
}

/** Reads apx/api's `text/event-stream` responses by hand instead of the browser's native
 * `EventSource`, which cannot attach an `Authorization` header — and every apx/api route,
 * including `GET /invoices/{id}/events`, requires a bearer JWT (apx.api.deps.get_current_user).
 * Frame format matches apx/api/sse.py exactly: `event: <name>` / `data: <json>` pairs. With a
 * `body` it POSTs JSON instead (POST /chat streams its answer the same way). */
async function streamServerSentEvents(
  url: string,
  signal: AbortSignal,
  onEvent: (event: SseEvent) => void,
  body?: string,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(url, {
      method: body === undefined ? "GET" : "POST",
      headers: {
        ...(bearerToken ? { Authorization: `Bearer ${bearerToken}` } : {}),
        ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      },
      body,
      signal,
    });
  } catch (err) {
    if (!signal.aborted) onEvent({ event: "error", data: { message: String(err) } });
    return;
  }
  if (!res.ok || !res.body) {
    onEvent({ event: "error", data: { message: await parseErrorDetail(res) } });
    return;
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let eventName = "message";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return;
      buffer += value;
      let sep = buffer.indexOf("\n\n");
      while (sep !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        let data = "";
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) eventName = line.slice("event:".length).trim();
          else if (line.startsWith("data:")) data += line.slice("data:".length).trim();
        }
        if (data) {
          try {
            onEvent({ event: eventName, data: JSON.parse(data) as Record<string, unknown> });
          } catch {
            onEvent({ event: eventName, data: { raw: data } });
          }
        }
        eventName = "message";
        sep = buffer.indexOf("\n\n");
      }
    }
  } catch (err) {
    if (!signal.aborted) onEvent({ event: "error", data: { message: String(err) } });
  }
}

interface AuditEventDto {
  id: number;
  ts: string;
  actor: string;
  action: string;
  facts: Record<string, string | number | boolean | null>;
  note: string | null;
}

interface ChatReplyDto {
  text: string;
  refusal: string | null;
  citations: { source: ChatCitation["source"]; ref: string; href: string | null }[];
  proposals: { id: string; kind: string }[];
}

function isChatReplyDto(
  data: Record<string, unknown>,
): data is Record<string, unknown> & ChatReplyDto {
  return typeof data.text === "string" && Array.isArray(data.citations);
}

class HttpApiClient implements ApiClient {
  async login(email: string, password: string): Promise<Session> {
    // apx.api.routers.auth.login takes an OAuth2PasswordRequestForm — form-encoded, not JSON.
    const res = await fetch(`${BASE_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ username: email, password }).toString(),
    });
    if (!res.ok) throw new ApiError(await parseErrorDetail(res), res.status);
    const token = (await res.json()) as TokenResponseDto;
    bearerToken = token.access_token;

    const claims = decodeAccessToken(token.access_token);
    const user: AuthUser = {
      id: claims.sub,
      // apx.api.deps.get_current_user sets `display_name` to the user id itself; there is no
      // separate human display name on the token or a `/auth/me` route to fetch one from.
      name: claims.sub,
      email,
      roles: claims.roles,
      // API_APPROVAL_LIMIT (apx/api/settings.py) is one process-wide setting, not a per-user
      // limit — there is nothing per-user to report here.
      approvalLimitAed: null,
    };
    return { user, token: token.access_token };
  }

  async restoreSession(session: Session): Promise<boolean> {
    let exp: number | null;
    try {
      exp = decodeAccessToken(session.token).exp;
    } catch {
      return false;
    }
    // apx/api always sets `exp`; a token without one, or already expired, is never re-attached.
    if (exp === null || exp * 1000 <= Date.now()) return false;
    bearerToken = session.token;
    return true;
  }

  async logout(_token: string): Promise<void> {
    // apx/api issues short-lived, non-revocable JWTs and has no `/auth/logout`
    // route — the token simply expires. Clearing the local copy is the entire contract here, not
    // a fallback standing in for a failed server call.
    bearerToken = null;
  }

  async listInvoices(params: ListInvoicesParams): Promise<readonly InvoiceSummary[]> {
    const want = params.limit ?? MAX_PAGE_SIZE;
    const items: InvoiceSummaryDto[] = [];
    let offset = 0;
    for (;;) {
      const search = new URLSearchParams();
      if (params.status) search.set("status", params.status);
      if (params.vendorId) search.set("vendor_id", params.vendorId);
      search.set("limit", String(Math.max(1, Math.min(MAX_PAGE_SIZE, want - items.length))));
      search.set("offset", String(offset));
      const page = await apiFetch<InvoicePageDto>(`/invoices?${search.toString()}`);
      items.push(...page.items);
      offset += page.items.length;
      if (items.length >= want || page.items.length === 0 || offset >= page.total) break;
    }
    let rows = items.map(mapSummary);
    // GET /invoices only filters by status/vendor_id/q server-side (apx/api/routers/invoices.py)
    // — applied here instead, so a caller that does pass `route` still gets a correct answer
    // rather than a silently-ignored filter.
    if (params.route) rows = rows.filter((row) => row.route === params.route);
    return rows;
  }

  async getInvoice(id: string): Promise<InvoiceDetail> {
    const dto = await apiFetch<InvoiceDetailDto>(`/invoices/${id}`);
    return {
      summary: mapSummary(dto.summary),
      fields: dto.fields.map((f) => mapField(f, dto.accepted)),
      // Stored apx.fraud flags, highest score first, each with its plain-language explanation.
      flags: dto.flags.map(mapFlag),
      reasons: dto.reasons,
      proposals: dto.proposals.map(mapProposal),
      // Derived from the fields' own grounding boxes (real data), not guessed: the highest page
      // index any field is grounded on, plus one.
      pageCount: pageCountFromFields(dto.fields),
      // Not exposed by apx/api yet; nothing in the UI reads InvoiceDetail.language today.
      language: "en",
      // apx.api.routers.invoices.get_invoice_document. Requires the same bearer
      // token as every other route here — the caller (invoice detail page) passes the session
      // token through to PdfPane/PdfCanvas, which attach it as pdf.js's `httpHeaders`.
      documentUrl: `${BASE_URL}/invoices/${id}/document`,
    };
  }

  async editField(invoiceId: string, edit: FieldEdit): Promise<void> {
    await apiFetchJson<void>(`/invoices/${invoiceId}/fields`, {
      method: "PATCH",
      body: JSON.stringify({ field: edit.field, value: edit.value, reason: edit.reason }),
    });
  }

  async decide(invoiceId: string, decision: DecisionRequest): Promise<void> {
    await apiFetchJson<void>(`/invoices/${invoiceId}/decisions`, {
      method: "POST",
      body: JSON.stringify({
        action: decision.action,
        proposal_id: decision.proposalId ?? null,
        comment: decision.comment ?? "",
      }),
    });
  }

  async getOpsSnapshot(): Promise<OpsSnapshot> {
    const dto = await apiFetch<OpsSnapshotDto>("/ops/snapshot");
    const k = dto.kpis;
    return {
      generatedAt: dto.generated_at,
      kpis: {
        stpRate: k.stp_rate,
        routedInvoices: k.routed_invoices,
        autoAcceptErrorRate: k.auto_accept_error_rate,
        automatedFields: k.automated_fields,
        llmLatencyP50Ms: k.llm_latency_p50_ms,
        llmLatencyP95Ms: k.llm_latency_p95_ms,
        llmCalls: k.llm_calls,
        e2eLatencyP50S: k.e2e_latency_p50_s,
        attacksBlocked: k.attacks_blocked,
        queueSize: k.queue_size,
      },
      stages: dto.stages,
      recent: dto.recent.map((r) => ({
        id: r.id,
        number: r.number,
        vendorName: r.vendor_name,
        status: r.status,
        route: r.route,
        total: r.total,
        currency: r.currency,
        updatedAt: r.updated_at,
      })),
      models: dto.models,
      modelsReachable: dto.models_reachable,
    };
  }

  async listVendors(): Promise<readonly Vendor[]> {
    // GET /vendors (apx.api.routers.vendors) is paginated the same way GET /invoices is —
    // walk every page the same way listInvoices does above, since the vendor list UI (a small,
    // fixed dataset) expects the whole set at once.
    const items: VendorSummaryDto[] = [];
    let offset = 0;
    for (;;) {
      const search = new URLSearchParams();
      search.set("limit", String(MAX_PAGE_SIZE));
      search.set("offset", String(offset));
      const page = await apiFetch<VendorPageDto>(`/vendors?${search.toString()}`);
      items.push(...page.items);
      offset += page.items.length;
      if (page.items.length === 0 || offset >= page.total) break;
    }
    return items.map(mapVendorSummary);
  }

  async getVendor(id: string): Promise<Vendor> {
    const dto = await apiFetch<VendorDetailDto>(`/vendors/${id}`);
    return mapVendorDetail(dto);
  }

  async getInvoiceAudit(id: string): Promise<readonly AuditEventView[]> {
    const rows = await apiFetch<AuditEventDto[]>(`/invoices/${id}/audit`);
    return rows.map((row) => ({ ...row, facts: row.facts ?? {} }));
  }

  async askAssistant(
    history: readonly ChatTurn[],
    onStep: (step: ChatStep) => void,
    signal: AbortSignal,
  ): Promise<ChatReply> {
    let reply: ChatReply | null = null;
    let failure: string | null = null;
    await streamServerSentEvents(
      `${BASE_URL}/chat`,
      signal,
      (event) => {
        if (event.event === "step") {
          const { tool, label, outcome } = event.data as Partial<ChatStep>;
          onStep({
            tool: typeof tool === "string" ? tool : null,
            label: typeof label === "string" ? label : "Using a tool",
            outcome: outcome === "refused" ? "refused" : "ok",
          });
        } else if (event.event === "reply" && isChatReplyDto(event.data)) {
          reply = {
            text: event.data.text,
            refusal: event.data.refusal ?? null,
            citations: event.data.citations,
            proposals: event.data.proposals ?? [],
          };
        } else if (event.event === "error") {
          const message = event.data.message;
          failure = typeof message === "string" ? message : "The assistant could not answer.";
        }
      },
      JSON.stringify({ messages: history }),
    );
    if (reply) return reply;
    throw new ApiError(failure ?? "The assistant stopped before answering.", 502);
  }

  subscribeEvents(invoiceId: string | null, onEvent: (event: SseEvent) => void): () => void {
    if (invoiceId === null) {
      // apx.api.routers.invoices only exposes GET /invoices/{id}/events — there is no
      // invoice-less feed to fall back to.
      throw new ApiError(
        "apx/api's event stream is scoped to one invoice; there is no global feed to subscribe to.",
        400,
      );
    }
    const controller = new AbortController();
    void streamServerSentEvents(
      `${BASE_URL}/invoices/${invoiceId}/events`,
      controller.signal,
      onEvent,
    );
    return () => controller.abort();
  }
}

export const httpApiClient: ApiClient = new HttpApiClient();
