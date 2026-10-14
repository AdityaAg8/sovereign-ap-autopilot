/** Factory that picks the API adapter for this environment. The UI only ever imports `ApiClient`
 * and `getApiClient` from this module — never a concrete adapter directly — so swapping the
 * fixture adapter for the HTTP adapter once apx/api lands is a one-line change here, not a
 * page-by-page rewrite. The fixture adapter (and its generated data) is imported
 * on first use and only in fixture mode, so it never ships on the critical path of any route. */
import { httpApiClient } from "@/lib/api/http-adapter";
import type { ApiClient } from "@/lib/api/contract";

export type { ApiClient } from "@/lib/api/contract";
export { ApiError } from "@/lib/api/contract";

export type ApiMode = "fixture" | "http";

/** Dev and Playwright e2e (the review UI's acceptance criteria) default to "fixture"; set
 * NEXT_PUBLIC_API_MODE=http once apx/api is reachable to run the same UI against the real
 * service. Exported separately from `getApiClient` so screens (e.g. the login form's demo-account
 * list) can branch on the mode without pulling in either adapter. */
export function getApiMode(): ApiMode {
  return process.env.NEXT_PUBLIC_API_MODE === "http" ? "http" : "fixture";
}

let fixtureModule: Promise<ApiClient> | null = null;

function loadFixtureClient(): Promise<ApiClient> {
  if (!fixtureModule) {
    const pending = import("@/lib/api/fixture-adapter").then((m) => m.fixtureApiClient);
    fixtureModule = pending;
    // A failed chunk load is not cached, so the next call retries the import.
    pending.catch(() => {
      if (fixtureModule === pending) fixtureModule = null;
    });
  }
  return fixtureModule;
}

/** Starts fetching the fixture adapter's chunk now instead of on the first API call, so a
 * signed-in route's session restore and first data load do not wait for one more serial round
 * trip after hydration. A no-op in http mode; a failed prefetch is retried by the first call. */
export function warmApiClient(): void {
  if (getApiMode() === "fixture") loadFixtureClient().catch(() => {});
}

/** Forwards every call to the fixture adapter once its module has loaded. A failed load rejects
 * the call (a plain Error, not ApiError), which callers report as "couldn't reach the server". */
const lazyFixtureClient: ApiClient = {
  login: async (email, password) => (await loadFixtureClient()).login(email, password),
  logout: async (token) => (await loadFixtureClient()).logout(token),
  restoreSession: async (session) => (await loadFixtureClient()).restoreSession(session),
  listInvoices: async (params) => (await loadFixtureClient()).listInvoices(params),
  getInvoice: async (id) => (await loadFixtureClient()).getInvoice(id),
  editField: async (invoiceId, edit) => (await loadFixtureClient()).editField(invoiceId, edit),
  decide: async (invoiceId, request) => (await loadFixtureClient()).decide(invoiceId, request),
  listVendors: async () => (await loadFixtureClient()).listVendors(),
  getVendor: async (id) => (await loadFixtureClient()).getVendor(id),
  getOpsSnapshot: async () => (await loadFixtureClient()).getOpsSnapshot(),
  getInvoiceAudit: async (id) => (await loadFixtureClient()).getInvoiceAudit(id),
  askAssistant: async (history, onStep, signal) =>
    (await loadFixtureClient()).askAssistant(history, onStep, signal),
  subscribeEvents(invoiceId, onEvent) {
    let unsubscribe: (() => void) | null = null;
    let closed = false;
    loadFixtureClient().then(
      (client) => {
        if (!closed) unsubscribe = client.subscribeEvents(invoiceId, onEvent);
      },
      (err: unknown) => {
        // No live events without the adapter; surface it rather than fail silently.
        console.error("Fixture event stream unavailable:", err);
      },
    );
    return () => {
      closed = true;
      unsubscribe?.();
    };
  },
};

export function getApiClient(): ApiClient {
  return getApiMode() === "http" ? httpApiClient : lazyFixtureClient;
}
