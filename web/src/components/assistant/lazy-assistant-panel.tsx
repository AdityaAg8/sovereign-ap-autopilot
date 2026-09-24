/** Mounts the ops assistant panel (PRD §10 M9.3) only once it is first opened, from a lazy chunk
 * warmed on the first interaction, so the sheet, chat UI and citation code stay off every
 * signed-in page's critical path. The `c` shortcut and the topbar button keep working because
 * they only flip the open flag in `AssistantProvider`, which is always mounted. */

"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { useAssistant } from "@/components/assistant/assistant-context";
import { usePrefetchOnInteraction } from "@/lib/first-interaction";
import { abandonOverlay, OVERLAY } from "@/lib/overlay-guard";
import { OverlayBoundary } from "@/components/shell/overlay-boundary";

const loadPanel = () => import("@/components/assistant/assistant-panel");

const AssistantPanel = dynamic(() => loadPanel().then((m) => m.AssistantPanel), { ssr: false });

export function LazyAssistantPanel() {
  const { open, setOpen } = useAssistant();
  // Latched: once opened, the panel stays mounted so a closed-then-reopened panel keeps its draft
  // and in-flight state, exactly as when it was mounted eagerly.
  const [mounted, setMounted] = useState(false);
  usePrefetchOnInteraction(loadPanel);

  useEffect(() => {
    if (!open) return;
    setMounted(true);
    // The opener holds the keyboard until the panel takes focus; a chunk that fails to load
    // must hand it straight back rather than wait out the guard's timeout.
    loadPanel().catch(() => abandonOverlay(OVERLAY.assistant));
  }, [open]);

  return mounted ? (
    <OverlayBoundary owner={OVERLAY.assistant} onFail={() => setOpen(false)}>
      <AssistantPanel />
    </OverlayBoundary>
  ) : null;
}
