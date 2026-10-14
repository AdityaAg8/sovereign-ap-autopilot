/** Client-side session state. The session is kept in tab-scoped sessionStorage —
 * never localStorage, so a short-lived apx/api JWT does not outlive the tab or leak to other tabs
 * — and on reload it is handed back to the API adapter (`restoreSession`), which re-attaches the
 * bearer token only if it has not expired; otherwise the user is signed out instead of every
 * request failing with a silent 401. The API client is imported on demand, not statically, so
 * routes that only render this provider (/login) do not download and evaluate the adapter and its
 * fixture data before first paint (Lighthouse performance budget). */

"use client";

import {
  createContext,
  startTransition,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { ApiClient } from "@/lib/api/client";
import type { AuthUser, Session } from "@/lib/api/types";

const STORAGE_KEY = "sovereign-ap.session";

interface AuthContextValue {
  session: Session | null;
  user: AuthUser | null;
  status: "loading" | "authenticated" | "unauthenticated";
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

let providedClient: ApiClient | null = null;

/** Hands the API client to this provider from a route bundle that already includes it (the
 * signed-in shell), so restoring the session there skips the on-demand chunk fetch below. */
export function provideApiClient(client: ApiClient): void {
  providedClient = client;
}

async function apiClient(): Promise<ApiClient> {
  if (providedClient) return providedClient;
  const { getApiClient } = await import("@/lib/api/client");
  return getApiClient();
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [status, setStatus] = useState<AuthContextValue["status"]>("loading");

  useEffect(() => {
    let cancelled = false;
    let restored: Session | null = null;
    try {
      const raw = window.sessionStorage.getItem(STORAGE_KEY);
      if (raw) restored = JSON.parse(raw) as Session;
    } catch {
      // Ignore corrupt/blocked storage; fall through to unauthenticated.
    }
    if (!restored) {
      setStatus("unauthenticated");
      return;
    }
    const candidate = restored;
    apiClient()
      .then((client) => client.restoreSession(candidate))
      .then(
        (ok) => {
          if (cancelled) return;
          if (ok) {
            // A transition, so React mounts the signed-in shell and the route's page in yielding
            // slices rather than one long task right after hydration (TBT).
            startTransition(() => {
              setSession(candidate);
              setStatus("authenticated");
            });
            return;
          }
          // The adapter rejected the session (expired/malformed): discard it for good.
          try {
            window.sessionStorage.removeItem(STORAGE_KEY);
          } catch {
            // Best-effort cleanup only.
          }
          setStatus("unauthenticated");
        },
        () => {
          // The client could not be loaded or reached: the stored session may still be valid,
          // so keep it for the next load and show the signed-out UI for now.
          if (!cancelled) setStatus("unauthenticated");
        },
      );
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const next = await (await apiClient()).login(email, password);
    setSession(next);
    setStatus("authenticated");
    try {
      window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      // Best-effort persistence only.
    }
  }, []);

  const logout = useCallback(async () => {
    if (session) await (await apiClient()).logout(session.token);
    setSession(null);
    setStatus("unauthenticated");
    try {
      window.sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      // Best-effort persistence only.
    }
  }, [session]);

  const value = useMemo<AuthContextValue>(
    () => ({ session, user: session?.user ?? null, status, login, logout }),
    [session, status, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
