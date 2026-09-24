/** Binds keyboard shortcuts for the keyboard-first review flows (PRD §10 M5.4: j/k/enter in the
 * inbox, a/f/r in the review queue, ⌘K/? globally). Ignores keystrokes typed into inputs unless a
 * binding opts in, and normalizes "mod" to Cmd on macOS / Ctrl elsewhere. */

"use client";

import { useEffect, useRef } from "react";

export interface HotkeyBinding {
  /** e.g. "j", "shift+enter", "mod+k", "?", "escape". Case-insensitive; "mod" = ctrl or cmd. */
  combo: string;
  handler: (event: KeyboardEvent) => void;
  description: string;
  /** Fire even while a text input/textarea/contenteditable is focused. Default false. */
  allowInInputs?: boolean;
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

function normalize(combo: string): { key: string; mod: boolean; shift: boolean; alt: boolean } {
  const parts = combo.toLowerCase().split("+");
  const key = parts[parts.length - 1] ?? "";
  return {
    key,
    mod: parts.includes("mod"),
    shift: parts.includes("shift"),
    alt: parts.includes("alt"),
  };
}

export function useHotkeys(bindings: readonly HotkeyBinding[], enabled = true): void {
  const bindingsRef = useRef(bindings);
  bindingsRef.current = bindings;

  useEffect(() => {
    if (!enabled) return;
    function onKeyDown(event: KeyboardEvent) {
      const key = event.key.toLowerCase();
      const modActive = event.metaKey || event.ctrlKey;
      for (const binding of bindingsRef.current) {
        const wanted = normalize(binding.combo);
        if (wanted.key !== key) continue;
        if (wanted.mod !== modActive) continue;
        if (wanted.shift !== event.shiftKey) continue;
        if (wanted.alt !== event.altKey) continue;
        if (!binding.allowInInputs && isTypingTarget(event.target)) continue;
        event.preventDefault();
        binding.handler(event);
        return;
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [enabled]);
}
