/** Authenticated app shell: sidebar nav, topbar, command palette, shortcut help and the assistant panel,
 * wrapping every page under (app) (PRD §10 M5.4). Redirects to /login when no session exists. */

"use client";

import { useEffect, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { LazyAssistantPanel } from "@/components/assistant/lazy-assistant-panel";
import { AssistantProvider } from "@/components/assistant/assistant-context";
import { Sidebar } from "@/components/shell/sidebar";
import { Topbar } from "@/components/shell/topbar";
import { CommandPalette } from "@/components/shell/command-palette";
import { ShortcutsHelpOverlay } from "@/components/shell/shortcuts-help-overlay";
import { getApiClient, warmApiClient } from "@/lib/api/client";
import { provideApiClient, useAuth } from "@/lib/auth/auth-context";
import { ShortcutsProvider } from "@/lib/shortcuts/registry";
import { PageHeaderProvider, usePageHeader } from "@/lib/shell/page-header-context";

// Every signed-in screen needs the API client straight after hydration (session restore, then the
// page's first load). Hand it to the auth provider and start the adapter's chunk while the rest of
// the route's JS is still arriving, instead of fetching both serially once hydration finishes.
if (typeof window !== "undefined") {
  provideApiClient(getApiClient());
  warmApiClient();
}

function Shell({ children }: { children: ReactNode }) {
  const { title, actions } = usePageHeader();
  return (
    <div className="flex h-dvh overflow-hidden">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar title={title} actions={actions} />
        <main className="min-w-0 flex-1 overflow-y-auto">{children}</main>
      </div>
      <CommandPalette />
      <ShortcutsHelpOverlay />
      <LazyAssistantPanel />
    </div>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const { status } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (status === "unauthenticated") router.replace("/login");
  }, [status, router]);

  if (status !== "authenticated") {
    return (
      <div className="text-muted-foreground flex h-dvh items-center justify-center text-sm">
        Loading…
      </div>
    );
  }

  return (
    <ShortcutsProvider>
      <PageHeaderProvider>
        <AssistantProvider>
          <Shell>{children}</Shell>
        </AssistantProvider>
      </PageHeaderProvider>
    </ShortcutsProvider>
  );
}
