/** Open/close state for the ops assistant panel (PRD §10 M9.3), shared by the topbar button, the
 * `c` shortcut and the panel itself. The conversation lives here too, so closing the panel and
 * reopening it keeps the thread. */

"use client";

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { useHotkeys } from "@/hooks/use-hotkeys";
import { beginOverlayOpen, OVERLAY } from "@/lib/overlay-guard";
import type { ChatReply, ChatTurn } from "@/lib/api/types";

export interface AssistantMessage extends ChatTurn {
  id: number;
  /** Present on assistant messages: citations, refusal and drafted proposals. */
  reply?: ChatReply;
}

interface AssistantValue {
  open: boolean;
  setOpen: (open: boolean) => void;
  messages: readonly AssistantMessage[];
  setMessages: (update: (prev: readonly AssistantMessage[]) => readonly AssistantMessage[]) => void;
}

const AssistantContext = createContext<AssistantValue | null>(null);

export function AssistantProvider({ children }: { children: ReactNode }) {
  const [open, setOpenState] = useState(false);
  const [messages, setMessages] = useState<readonly AssistantMessage[]>([]);
  // Opening holds the keyboard until the panel's question box has focus (lib/overlay-guard.ts):
  // on a first open its chunk is still loading, and keys typed meanwhile must not reach page
  // hotkeys such as Approve. The panel claims them back when it takes focus.
  const setOpen = useCallback(
    (next: boolean) => {
      if (next && !open) beginOverlayOpen(OVERLAY.assistant);
      setOpenState(next);
    },
    [open],
  );
  useHotkeys([{ combo: "c", description: "Ask the assistant", handler: () => setOpen(true) }]);
  const value = useMemo(
    () => ({ open, setOpen, messages, setMessages }),
    [open, setOpen, messages],
  );
  return <AssistantContext.Provider value={value}>{children}</AssistantContext.Provider>;
}

export function useAssistant(): AssistantValue {
  const value = useContext(AssistantContext);
  if (!value) throw new Error("useAssistant must be used inside <AssistantProvider>");
  return value;
}
