/** Shell header: page title, the ⌘K launcher, the assistant button and the theme toggle (the
 * account menu lives in the sidebar). Rendered once by the authenticated layout; pages only supply
 * a title and page actions. */

"use client";

import type { ReactNode } from "react";
import { MessageSquareText, SearchIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/shell/theme-toggle";
import { useAssistant } from "@/components/assistant/assistant-context";

const KEYCAP =
  "border-border bg-background text-muted-foreground hidden h-5 min-w-5 items-center justify-center rounded border px-1 font-mono text-2xs after:content-[attr(data-key)]";

function AskButton() {
  const { setOpen } = useAssistant();
  return (
    <Button
      variant="outline"
      size="sm"
      className="text-muted-foreground h-8 gap-2 font-normal"
      onClick={() => setOpen(true)}
      aria-label="Ask the assistant"
      aria-keyshortcuts="c"
    >
      <MessageSquareText className="size-3.5" aria-hidden />
      <span className="hidden md:inline">Ask</span>
      {/* The key cap is drawn with generated content, not text, so the visible label stays
          "Ask" and matches the accessible name (WCAG 2.5.3); aria-keyshortcuts carries the key. */}
      <kbd aria-hidden data-key="C" className={`${KEYCAP} md:inline-flex`} />
    </Button>
  );
}

export function Topbar({ title, actions }: { title: string; actions?: ReactNode }) {
  return (
    <header className="border-border bg-background/90 sticky top-0 z-20 flex h-14 shrink-0 items-center justify-between gap-4 border-b px-4 backdrop-blur-sm sm:px-6">
      <h1 className="truncate text-base font-semibold tracking-[-0.01em]">{title}</h1>
      <div className="flex items-center gap-2">
        {actions}
        <Button
          variant="outline"
          size="sm"
          className="text-muted-foreground h-8 justify-between gap-3 font-normal sm:w-64"
          aria-label="Search…"
          aria-keyshortcuts="Meta+K Control+K"
          onClick={() =>
            window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", metaKey: true }))
          }
        >
          <span className="flex items-center gap-2">
            <SearchIcon className="size-3.5" aria-hidden />
            <span className="hidden sm:inline">Search…</span>
          </span>
          <kbd aria-hidden data-key="⌘K" className={`${KEYCAP} sm:inline-flex`} />
        </Button>
        <AskButton />
        <ThemeToggle />
      </div>
    </header>
  );
}
