/** Binds keyboard shortcuts for the keyboard-first review flows (j/k/enter in the
 * inbox, a/f/r in the review queue, ⌘K/? globally). Skips keys a component already consumed, keys
 * typed into inputs or pressed inside open popups (unless a binding opts in), and Enter/Space on
 * controls that own them; "mod" is Cmd on macOS / Ctrl elsewhere. Letter shortcuts also work on
 * non-Latin layouts (e.g. Arabic) via the physical key, and AltGr-typed symbols match plain combos.
 * While a lazily loaded overlay is opening (lib/overlay-guard.ts) only popup-safe bindings fire. */

"use client";

import { useEffect, useRef } from "react";
import { isOverlayPending } from "@/lib/overlay-guard";

export interface HotkeyBinding {
  /** e.g. "j", "shift+enter", "mod+k", "?", "escape". Case-insensitive; "mod" = ctrl or cmd. */
  combo: string;
  handler: (event: KeyboardEvent) => void;
  description: string;
  /** Fire even while a text input/textarea/contenteditable is focused. Default false. */
  allowInInputs?: boolean;
  /** Fire even when the key lands inside an open dialog, menu or combobox popup (e.g. a toggle
   * that should also close its own dialog). Default false. */
  allowInPopups?: boolean;
}

/** Events a useHotkeys listener already acted on. Every hook instance has its own window listener,
 * so this tells "consumed by a sibling hotkey" (still eligible, as before) apart from "consumed by
 * a component" (e.g. Radix opening a select on Enter), which is skipped. */
const handledByHotkeys = new WeakSet<Event>();

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

const POPUP_ROLES = new Set(["dialog", "alertdialog", "menu"]);

/** A listbox counts as a popup only when a combobox controls it (the ARIA combobox pattern, which
 * Radix Select and cmdk follow). A page-level listbox such as the inbox rows is not one. */
function isComboboxPopup(el: Element): boolean {
  if (el.getAttribute("role") !== "listbox" || !el.id) return false;
  const id = CSS.escape(el.id);
  return document.querySelector(`[aria-controls~="${id}"], [aria-owns~="${id}"]`) !== null;
}

/** Walks the dispatch path rather than `target.closest`, because selecting a Radix option can
 * unmount its popup before the event bubbles up to window. */
function isInsidePopup(event: KeyboardEvent): boolean {
  for (const node of event.composedPath()) {
    if (!(node instanceof Element)) continue;
    if (POPUP_ROLES.has(node.getAttribute("role") ?? "")) return true;
    if (node.getAttribute("aria-modal") === "true") return true;
    if (isComboboxPopup(node)) return true;
  }
  return false;
}

const ACTIVATING_ROLES = new Set([
  "button",
  "link",
  "combobox",
  "checkbox",
  "radio",
  "switch",
  "tab",
  "option",
  "menuitem",
  "menuitemcheckbox",
  "menuitemradio",
]);

/** Enter/Space activate buttons, links and similar controls; a page shortcut must not also fire. */
function ownsActivationKey(target: EventTarget | null, key: string): boolean {
  if (key !== "enter" && key !== " ") return false;
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === "BUTTON" || tag === "SUMMARY" || (tag === "A" && target.hasAttribute("href"))) {
    return true;
  }
  return ACTIVATING_ROLES.has(target.getAttribute("role") ?? "");
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

/** Single printable non-letter keys ("?", "/", "!") already encode Shift in the character the
 * layout produced, so a plain "?" combo ignores Shift; "shift+/" still requires it. Letters and
 * named keys match Shift exactly. */
function isShiftedSymbol(key: string): boolean {
  return key.length === 1 && key !== " " && key.toLowerCase() === key.toUpperCase();
}

const LATIN_LETTER_OR_DIGIT = /^[a-z0-9]$/;
/** One letter or digit from a non-Latin script, e.g. Arabic "ت" or "٣". Latin letters, accented
 * ones included (AZERTY "é"), keep matching only by the character the layout produced. */
const NON_LATIN_LETTER_OR_DIGIT = /^(?!\p{Script=Latin})[\p{L}\p{N}]$/u;

/** The Latin letter or digit printed on the event's physical key ("KeyJ" -> "j", "Digit1" -> "1"). */
function physicalLetterOrDigit(code: string): string | null {
  const match = /^(?:Key([A-Z])|Digit([0-9]))$/.exec(code);
  const found = match?.[1] ?? match?.[2];
  return found ? found.toLowerCase() : null;
}

interface ResolvedKey {
  /** Keys to try against combos, in priority order: the produced character first, then (only for
   * a non-Latin letter or digit) the Latin letter or digit at the same physical key. */
  keys: string[];
  mod: boolean;
  alt: boolean;
}

/** Maps a keydown to what combos compare against. Ctrl+Alt with a single printable symbol is
 * AltGr (how Windows reports it), so it counts as neither "mod" nor "alt" and matches only by the
 * typed symbol. Latin letters and digits 0-9 never take this path, so a real Ctrl+Alt+1 or
 * Ctrl/Cmd+K keeps its modifiers and cannot fire a plain "1" or "k" binding. */
function resolveKey(event: KeyboardEvent): ResolvedKey {
  const key = event.key.toLowerCase();
  const altGr =
    isShiftedSymbol(key) &&
    !LATIN_LETTER_OR_DIGIT.test(key) &&
    !event.metaKey &&
    (event.getModifierState("AltGraph") || (event.ctrlKey && event.altKey));
  if (altGr) return { keys: [key], mod: false, alt: false };
  const keys = [key];
  if (!LATIN_LETTER_OR_DIGIT.test(key) && NON_LATIN_LETTER_OR_DIGIT.test(key)) {
    const physical = physicalLetterOrDigit(event.code);
    if (physical) keys.push(physical);
  }
  return { keys, mod: event.metaKey || event.ctrlKey, alt: event.altKey };
}

export function useHotkeys(bindings: readonly HotkeyBinding[], enabled = true): void {
  const bindingsRef = useRef(bindings);
  bindingsRef.current = bindings;

  useEffect(() => {
    if (!enabled) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.defaultPrevented && !handledByHotkeys.has(event)) return;
      // Keys pressed mid-composition belong to the IME (building an Arabic or CJK word), not to us.
      if (event.isComposing || event.key === "Process") return;
      const key = event.key.toLowerCase();
      const resolved = resolveKey(event);
      for (const candidate of resolved.keys) {
        for (const binding of bindingsRef.current) {
          const wanted = normalize(binding.combo);
          if (wanted.key !== candidate) continue;
          if (wanted.mod !== resolved.mod) continue;
          if (wanted.shift !== event.shiftKey && (wanted.shift || !isShiftedSymbol(wanted.key))) {
            continue;
          }
          if (wanted.alt !== resolved.alt) continue;
          if (!binding.allowInInputs && isTypingTarget(event.target)) continue;
          if (!binding.allowInPopups && isInsidePopup(event)) continue;
          // An overlay was requested but has not taken focus yet: the key is meant for it.
          if (!binding.allowInPopups && isOverlayPending()) continue;
          if (ownsActivationKey(event.target, key)) continue;
          event.preventDefault();
          handledByHotkeys.add(event);
          binding.handler(event);
          return;
        }
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [enabled]);
}
