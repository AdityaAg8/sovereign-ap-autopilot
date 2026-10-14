/** The `ApiClient` contract every adapter (fixture, HTTP) implements, matching the REST surface
 * apx/api will expose. Kept separate from client.ts so adapters can implement
 * this interface without importing the factory, avoiding a circular module graph. */
import type {
  AuditEventView,
  ChatReply,
  ChatStep,
  ChatTurn,
  DecisionRequest,
  FieldEdit,
  InvoiceDetail,
  InvoiceSummary,
  ListInvoicesParams,
  OpsSnapshot,
  Session,
  SseEvent,
  Vendor,
} from "@/lib/api/types";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface ApiClient {
  login(email: string, password: string): Promise<Session>;
  logout(token: string): Promise<void>;
  /** Re-attaches a session persisted before a reload (the bearer token lives only in memory).
   * Returns false when it is no longer usable — expired or malformed — and the caller must then
   * discard it rather than show a signed-in UI whose every request would 401. Async so a lazily
   * loaded adapter can answer; it rejects only when the adapter itself cannot be reached. */
  restoreSession(session: Session): Promise<boolean>;
  listInvoices(params: ListInvoicesParams): Promise<readonly InvoiceSummary[]>;
  getInvoice(id: string): Promise<InvoiceDetail>;
  editField(invoiceId: string, edit: FieldEdit): Promise<void>;
  decide(invoiceId: string, request: DecisionRequest): Promise<void>;
  listVendors(): Promise<readonly Vendor[]>;
  getVendor(id: string): Promise<Vendor>;
  /** One live operations snapshot (GET /ops/snapshot). */
  getOpsSnapshot(): Promise<OpsSnapshot>;
  /** An invoice's audit trail, newest first (GET /invoices/{id}/audit). */
  getInvoiceAudit(id: string): Promise<readonly AuditEventView[]>;
  /** Asks the ops assistant (POST /chat). `onStep` receives each tool call as it
   * happens; resolves with the cited reply, rejects with ApiError when it cannot answer. */
  askAssistant(
    history: readonly ChatTurn[],
    onStep: (step: ChatStep) => void,
    signal: AbortSignal,
  ): Promise<ChatReply>;
  /** Subscribes to invoice events (apx.api SSE stream); returns an unsubscribe function. */
  subscribeEvents(invoiceId: string | null, onEvent: (event: SseEvent) => void): () => void;
}
