/** Lets a popup control (Radix Select, Dropdown Menu) render as a look-alike static trigger first
 * and swap in the real component once its chunk has loaded on the first interaction, keeping
 * those primitives off the signed-in pages' critical path. The swap never interrupts what the
 * person is doing: it waits while the stand-in is pressed or focused, and a press or open key on
 * the stand-in opens the real popup instead of being dropped. */

"use client";

import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from "react";
import { onFirstInteraction } from "@/lib/first-interaction";

const loaded = new Map<() => Promise<unknown>, unknown>();

/** Keys that open a Radix Select / Dropdown Menu from its focused trigger. */
const OPEN_KEYS = new Set(["Enter", " ", "ArrowDown", "ArrowUp"]);

/** Loads `load`'s module on the page's first interaction (or at once if `now` is called first)
 * and returns it, or `null` until it has loaded. Modules are cached per loader across mounts. */
export function useLoadOnInteraction<M>(load: () => Promise<M>): {
  module: M | null;
  now: () => void;
} {
  const [module, setModule] = useState<M | null>(() => (loaded.get(load) as M | undefined) ?? null);
  const loadStarted = useRef(false);

  const now = useCallback(() => {
    const cached = loaded.get(load) as M | undefined;
    if (cached !== undefined) {
      setModule(() => cached);
      return;
    }
    if (loadStarted.current) return;
    loadStarted.current = true;
    load().then(
      (m) => {
        loaded.set(load, m);
        setModule(() => m);
      },
      () => {
        // Retried the next time `now` is called (e.g. the stand-in is activated again).
        loadStarted.current = false;
      },
    );
  }, [load]);

  useEffect(() => (module ? undefined : onFirstInteraction(now)), [module, now]);

  return { module, now };
}

export interface DeferredControl<M> {
  /** The loaded module once it may be swapped in, or `null` while the stand-in should render. */
  module: M | null;
  /** Spread onto the stand-in trigger. */
  standInProps: {
    onPointerDown: () => void;
    onClick: () => void;
    onKeyDown: (event: KeyboardEvent<HTMLElement>) => void;
    onFocus: () => void;
    onBlur: () => void;
  };
  /** Pass as the real component's `defaultOpen`: the stand-in was activated before the swap. */
  openRequested: boolean;
}

export function useDeferredControl<M>(load: () => Promise<M>): DeferredControl<M> {
  const { module, now } = useLoadOnInteraction(load);
  const [pressing, setPressing] = useState(false);
  const [openRequested, setOpenRequested] = useState(false);
  const [focused, setFocused] = useState(false);
  const releaseRef = useRef<(() => void) | null>(null);

  // Stop listening for the end of a press if the control unmounts mid-press.
  useEffect(() => () => releaseRef.current?.(), []);

  /** Holds the swap until one of `endEvents` reaches the window, then releases it a task later,
   * after any click or keyup handling that event triggers has run on the stand-in. */
  const hold = useCallback((endEvents: readonly string[]) => {
    releaseRef.current?.();
    setPressing(true);
    const onEnd = () => setTimeout(release, 0);
    const release = () => {
      for (const type of endEvents) window.removeEventListener(type, onEnd);
      if (releaseRef.current === release) releaseRef.current = null;
      setPressing(false);
    };
    for (const type of endEvents) window.addEventListener(type, onEnd, { once: true });
    releaseRef.current = release;
  }, []);

  const requestOpen = useCallback(() => {
    setOpenRequested(true);
    now();
  }, [now]);

  const standInProps = {
    // Hold the swap until the press ends: swapping between pointerdown and click would land the
    // click on the freshly opened popup instead of the trigger. A touch that turns into a scroll
    // ends in pointercancel rather than pointerup, so both release it.
    onPointerDown: () => hold(["pointerup", "pointercancel"]),
    onClick: requestOpen,
    // Likewise for keys: the real popup opens only once the opening key is released, so that
    // key's keyup cannot land on (and pick) the popup's highlighted option.
    onKeyDown: (event: KeyboardEvent<HTMLElement>) => {
      if (!OPEN_KEYS.has(event.key)) return;
      event.preventDefault();
      if (!event.repeat) hold(["keyup", "blur"]);
      requestOpen();
    },
    onFocus: () => setFocused(true),
    onBlur: () => setFocused(false),
  };

  // Swapping a focused stand-in would unmount the focused node, dropping focus to <body> and
  // making screen readers announce the control twice. So a focused stand-in stays until it is
  // opened (Radix then moves focus into the popup, and back to the real trigger on close) or
  // blurred; only an idle, unfocused stand-in is replaced silently.
  const swappable = !pressing && (!focused || openRequested);
  return { module: swappable ? module : null, standInProps, openRequested };
}
