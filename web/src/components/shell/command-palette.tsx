/** Global ⌘K command palette entry point (PRD §10 M5.4): owns the Mod+K binding and open state
 * from anywhere in the authenticated shell, while the palette itself (cmdk + dialog) is a lazy
 * chunk warmed on the first interaction and mounted on first open (Lighthouse budget). Opening
 * holds the keyboard (lib/overlay-guard.ts) until the palette's input has focus, so keys typed
 * while its chunk loads go into the search box instead of firing page hotkeys. */

"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { useHotkeys } from "@/hooks/use-hotkeys";
import { usePrefetchOnInteraction } from "@/lib/first-interaction";
import { abandonOverlay, beginOverlayOpen, OVERLAY } from "@/lib/overlay-guard";
import { OverlayBoundary } from "@/components/shell/overlay-boundary";

const loadDialog = () => import("@/components/shell/command-palette-dialog");

const CommandPaletteDialog = dynamic(() => loadDialog().then((m) => m.CommandPaletteDialog), {
  ssr: false,
});

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  // Latched on first open, so the palette's query and results survive closing it.
  const [mounted, setMounted] = useState(false);
  usePrefetchOnInteraction(loadDialog);

  useHotkeys([
    {
      combo: "mod+k",
      description: "Open the command palette",
      handler: () => {
        if (!open) {
          beginOverlayOpen(OVERLAY.palette);
          loadDialog().catch(() => abandonOverlay(OVERLAY.palette));
        }
        setMounted(true);
        setOpen((v) => !v);
      },
    },
  ]);

  return mounted ? (
    <OverlayBoundary owner={OVERLAY.palette} onFail={() => setOpen(false)}>
      <CommandPaletteDialog open={open} onOpenChange={setOpen} />
    </OverlayBoundary>
  ) : null;
}
