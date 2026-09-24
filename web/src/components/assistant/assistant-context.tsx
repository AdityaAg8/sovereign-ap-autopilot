/** Open/close state for the ops assistant panel (PRD §10 M9.3), shared by the topbar button, the
 * `c` shortcut and the panel itself. The conversation lives here too, so closing the panel and
 * reopening it keeps the thread. */

"use client";

import { createContext, useContext, useMemo, useState, type ReactNode } from "react";
import { useHotkeys } from "@/hooks/use-hotkeys";
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
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<readonly AssistantMessage[]>([]);
  useHotkeys([{ combo: "c", description: "Ask the assistant", handler: () => setOpen(true) }]);
  const value = useMemo(() => ({ open, setOpen, messages, setMessages }), [open, messages]);
  return <AssistantContext.Provider value={value}>{children}</AssistantContext.Provider>;
}

export function useAssistant(): AssistantValue {
  const value = useContext(AssistantContext);
  if (!value) throw new Error("useAssistant must be used inside <AssistantProvider>");
  return value;
}
