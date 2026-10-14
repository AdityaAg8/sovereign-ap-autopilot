/** Shell header: page title, ⌘K launcher, theme toggle and the signed-in user menu.
 * Rendered once by the authenticated layout; pages only supply a title and page actions. */

"use client";

import type { ReactNode } from "react";
import { MessageSquareText, SearchIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { AccountMenu } from "@/components/shell/account-menu";
import { ThemeToggle } from "@/components/shell/theme-toggle";
import { useAssistant } from "@/components/assistant/assistant-context";
import { useAuth } from "@/lib/auth/auth-context";

function AskButton() {
  const { setOpen } = useAssistant();
  return (
    <Button
      variant="outline"
      size="sm"
      className="text-muted-foreground font-normal"
      onClick={() => setOpen(true)}
      aria-label="Ask the assistant"
      aria-keyshortcuts="c"
    >
      <MessageSquareText className="size-3.5" aria-hidden />
      <span className="hidden md:inline">Ask</span>
      {/* The key cap is drawn with generated content, not text, so the visible label stays
          "Ask" and matches the accessible name (WCAG 2.5.3); aria-keyshortcuts carries the key. */}
      <kbd
        aria-hidden
        data-key="C"
        className="border-border bg-muted hidden rounded px-1 py-0.5 font-mono text-[0.6875rem] after:content-[attr(data-key)] md:inline"
      />
    </Button>
  );
}

export function Topbar({ title, actions }: { title: string; actions?: ReactNode }) {
  const { user } = useAuth();

  return (
    <header className="border-border bg-background/95 sticky top-0 z-20 flex h-14 items-center justify-between gap-4 border-b px-4 backdrop-blur-sm sm:px-6">
      <h1 className="truncate text-lg font-semibold tracking-tight">{title}</h1>
      <div className="flex items-center gap-2">
        {actions}
        <Button
          variant="outline"
          size="sm"
          className="text-muted-foreground justify-between font-normal sm:w-56"
          aria-label="Search…"
          aria-keyshortcuts="Meta+K Control+K"
          onClick={() =>
            window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true }))
          }
        >
          <span className="flex items-center gap-1.5">
            <SearchIcon className="size-3.5" aria-hidden />
            <span className="hidden sm:inline">Search…</span>
          </span>
          <kbd
            aria-hidden
            data-key="⌘K"
            className="border-border bg-muted hidden rounded px-1 py-0.5 font-mono text-[0.6875rem] after:content-[attr(data-key)] sm:inline"
          />
        </Button>
        <AskButton />
        <ThemeToggle />
        {user && <AccountMenu user={user} />}
      </div>
    </header>
  );
}
