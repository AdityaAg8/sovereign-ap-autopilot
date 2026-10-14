/** "?" opens a reference of every keyboard shortcut active on the current screen.
 * This entry point owns the binding and open state; the dialog body is a lazy chunk warmed on
 * the first interaction and mounted on first open, keeping it off the critical path. */

"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { useHotkeys } from "@/hooks/use-hotkeys";
import { usePrefetchOnInteraction } from "@/lib/first-interaction";
import { abandonOverlay, beginOverlayOpen, OVERLAY } from "@/lib/overlay-guard";
import { OverlayBoundary } from "@/components/shell/overlay-boundary";

const loadDialog = () => import("@/components/shell/shortcuts-help-dialog");

const ShortcutsHelpDialog = dynamic(() => loadDialog().then((m) => m.ShortcutsHelpDialog), {
  ssr: false,
});

export function ShortcutsHelpOverlay() {
  const [open, setOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  usePrefetchOnInteraction(loadDialog);

  useHotkeys([
    {
      combo: "?",
      description: "Show keyboard shortcuts",
      handler: () => {
        // Until the dialog has focus, keys go to it, not to page hotkeys (lib/overlay-guard.ts).
        if (!open) {
          beginOverlayOpen(OVERLAY.shortcuts);
          loadDialog().catch(() => abandonOverlay(OVERLAY.shortcuts));
        }
        setMounted(true);
        setOpen((v) => !v);
      },
      // "?" also closes the reference while its own dialog has focus.
      allowInPopups: true,
    },
  ]);

  return mounted ? (
    <OverlayBoundary owner={OVERLAY.shortcuts} onFail={() => setOpen(false)}>
      <ShortcutsHelpDialog open={open} onOpenChange={setOpen} />
    </OverlayBoundary>
  ) : null;
}
