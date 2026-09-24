/** Holds the keyboard between an overlay being requested (⌘K palette, `c` assistant) and that
 * overlay mounting from its lazy chunk and taking focus, so fast-typed keys in that gap cannot
 * reach page hotkeys: "⌘K acme" on an invoice must never fire `a` (Approve). Keys typed in the
 * gap go to the overlay's input once it opens, and if it fails or is too slow to open, the hold
 * lasts until the typing stops, with a visible, announced status message throughout. */

"use client";

/** Owners, one per lazily loaded overlay. */
export const OVERLAY = {
  palette: "command-palette",
  assistant: "assistant",
  shortcuts: "shortcuts-help",
} as const;

export type OverlayOwner = (typeof OVERLAY)[keyof typeof OVERLAY];

const OVERLAY_NAME: Record<OverlayOwner, string> = {
  "command-palette": "the command palette",
  assistant: "the assistant",
  "shortcuts-help": "the keyboard shortcuts",
};

/** Upper bound on waiting for the overlay; past it the hold only drains the ongoing typing. */
export const OVERLAY_PENDING_TIMEOUT_MS = 4000;
/** After a failure or the timeout, the hold ends once no key has been pressed for this long
 * (or on Escape or a pointer press), so a burst of typing aimed at the overlay is used up
 * rather than landing on page hotkeys, yet the page is never locked for long. */
export const OVERLAY_DRAIN_IDLE_MS = 1000;
/** "Opening…" appears only if the overlay has not taken focus by then, so a cached chunk
 * opening within a frame or two neither flashes the message nor announces it. */
const OPENING_MESSAGE_DELAY_MS = 200;
/** How long a failure or slowness message stays up after the page gets its keys back. */
const OUTCOME_MESSAGE_MS = 5000;

export interface OverlayClaim {
  /** Printable characters typed while the overlay was loading, in order. */
  text: string;
  /** Escape was pressed while the overlay was loading: the person no longer wants it. */
  cancelled: boolean;
}

interface Hold {
  owner: OverlayOwner;
  /** Waiting for the overlay, or (after a failure or the timeout) draining the typing. */
  phase: "pending" | "draining";
  text: string;
  cancelled: boolean;
  timer: ReturnType<typeof setTimeout>;
}

let hold: Hold | null = null;

// --- status message -------------------------------------------------------------------------

let statusEl: HTMLElement | null = null;
let statusTimer: ReturnType<typeof setTimeout> | undefined;

/** One polite live region, created on first use and kept in the DOM so screen readers track it.
 * Outside React on purpose: it must still work when the overlay's render has just failed. */
function statusRegion(): HTMLElement {
  if (statusEl?.isConnected) return statusEl;
  const el = document.createElement("div");
  el.setAttribute("role", "status");
  el.setAttribute("aria-live", "polite");
  el.dataset.testid = "overlay-status";
  el.className =
    "pointer-events-none fixed inset-x-0 bottom-6 z-[60] mx-auto w-fit max-w-[calc(100%-2rem)] " +
    "rounded-md border border-border bg-popover px-3 py-2 text-sm text-popover-foreground " +
    "shadow-md empty:hidden";
  document.body.append(el);
  statusEl = el;
  return el;
}

function showStatus(text: string, delay = 0, clearAfter?: number) {
  clearTimeout(statusTimer);
  const apply = () => {
    statusRegion().textContent = text;
    if (clearAfter !== undefined) statusTimer = setTimeout(() => showStatus(""), clearAfter);
  };
  if (delay > 0) statusTimer = setTimeout(apply, delay);
  else apply();
}

// --- the hold -------------------------------------------------------------------------------

const PASS_THROUGH = new Set(["Tab", "Shift", "Control", "Alt", "AltGraph", "Meta", "CapsLock"]);

/** Runs before every other keydown listener (window, capture phase) and swallows the key: no
 * hotkey, no button activation, no default action. Tab and lone modifiers pass, so keyboard
 * users can still move focus; nothing else reaches the page while the hold lasts. */
function holdKey(event: KeyboardEvent) {
  if (!hold || PASS_THROUGH.has(event.key)) return;
  event.preventDefault();
  event.stopImmediatePropagation();
  if (hold.phase === "draining") {
    if (event.key === "Escape") release();
    else restartDrainTimer();
  }
  if (!hold || event.isComposing || event.key === "Process") return;
  if (event.key === "Escape") {
    hold.cancelled = true;
    hold.text = "";
  } else if (event.key === "Backspace") {
    hold.text = hold.text.slice(0, -1);
  } else if ([...event.key].length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
    hold.text += event.key;
  }
}

function onPointerDown() {
  if (hold?.phase === "draining") release();
}

function restartDrainTimer() {
  if (!hold) return;
  clearTimeout(hold.timer);
  hold.timer = setTimeout(release, OVERLAY_DRAIN_IDLE_MS);
}

/** The overlay failed or timed out: keep swallowing the typing already under way until it
 * stops, then give the page its keys back. */
function drain(outcome: "failed" | "slow") {
  if (!hold) return;
  const name = OVERLAY_NAME[hold.owner];
  showStatus(
    outcome === "failed"
      ? failedMessage(hold.owner)
      : `${name[0]!.toUpperCase()}${name.slice(1)} is taking too long to open.`,
  );
  if (hold.phase === "draining") return;
  hold.phase = "draining";
  restartDrainTimer();
}

function release() {
  if (!hold) return;
  const drained = hold.phase === "draining";
  clearTimeout(hold.timer);
  hold = null;
  window.removeEventListener("keydown", holdKey, true);
  window.removeEventListener("pointerdown", onPointerDown, true);
  // A failure or slowness message stays readable for a moment; "Opening…" just goes.
  if (drained) showStatus(statusRegion().textContent ?? "", 0, OUTCOME_MESSAGE_MS);
  else showStatus("");
}

/** Called synchronously by an opening handler, before the overlay has rendered. A second
 * request while one is held keeps the first owner's buffer. */
export function beginOverlayOpen(owner: OverlayOwner): void {
  if (hold) return;
  hold = {
    owner,
    phase: "pending",
    text: "",
    cancelled: false,
    timer: setTimeout(() => drain("slow"), OVERLAY_PENDING_TIMEOUT_MS),
  };
  window.addEventListener("keydown", holdKey, true);
  window.addEventListener("pointerdown", onPointerDown, { capture: true, passive: true });
  showStatus(`Opening ${OVERLAY_NAME[owner]}…`, OPENING_MESSAGE_DELAY_MS);
}

/** Called by the overlay once it is mounted and its input has focus: releases the keyboard and
 * returns what was typed meanwhile (also when it arrives late, while the typing is draining).
 * A no-op returning an empty claim when `owner` holds nothing. */
export function claimOverlay(owner: OverlayOwner): OverlayClaim {
  if (!hold || hold.owner !== owner) return { text: "", cancelled: false };
  const claim = { text: hold.text, cancelled: hold.cancelled };
  hold.phase = "pending"; // an overlay that did open leaves no failure message behind
  release();
  return claim;
}

/** Called when the overlay's chunk or render failed: says so, and returns the keyboard to the
 * page once the typing aimed at the overlay has stopped. */
export function abandonOverlay(owner: OverlayOwner): void {
  if (hold?.owner === owner) drain("failed");
  else if (!hold) showStatus(failedMessage(owner), 0, OUTCOME_MESSAGE_MS);
}

function failedMessage(owner: OverlayOwner): string {
  return `Couldn't open ${OVERLAY_NAME[owner]}. Reload the page to try again.`;
}

/** True while an overlay has been requested and has not taken focus, or its typing drains. */
export function isOverlayPending(): boolean {
  return hold !== null;
}
