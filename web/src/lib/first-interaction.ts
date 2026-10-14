/** Defers non-critical work until the person first interacts with the page (pointer, touch or
 * key). Used to warm lazily loaded overlays (command palette, shortcut help, assistant panel,
 * toasts) so their chunks never compete with first render yet are ready by the time a shortcut
 * or click needs them (Lighthouse budget). */

"use client";

import { useEffect } from "react";

const EVENTS = ["pointerdown", "pointermove", "touchstart", "keydown", "focusin"] as const;

type Listener = () => void;

let interacted = false;
const pending = new Set<Listener>();

function fire() {
  if (interacted) return;
  interacted = true;
  for (const type of EVENTS) window.removeEventListener(type, fire, true);
  const listeners = [...pending];
  pending.clear();
  for (const listener of listeners) listener();
}

/** Calls `listener` on the page's first interaction, or on the next tick if that already
 * happened. Returns a cancel function. Only events seen here count: the browser's sticky
 * `navigator.userActivation` is not used, because automation (Playwright, Lighthouse) reports it
 * as already active on a fresh page, which would load every deferred chunk at mount. */
export function onFirstInteraction(listener: Listener): () => void {
  if (interacted) {
    const handle = setTimeout(listener, 0);
    return () => clearTimeout(handle);
  }
  if (pending.size === 0) {
    for (const type of EVENTS) {
      window.addEventListener(type, fire, { capture: true, passive: true });
    }
  }
  pending.add(listener);
  return () => {
    pending.delete(listener);
  };
}

/** Prefetches a lazily imported module on the first interaction. A failed prefetch is ignored:
 * the real import on first use retries and surfaces its own error. */
export function usePrefetchOnInteraction(load: () => Promise<unknown>): void {
  useEffect(
    () =>
      onFirstInteraction(() => {
        load().catch(() => {});
      }),
    // `load` is a module-level import thunk; prefetch once per mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );
}
