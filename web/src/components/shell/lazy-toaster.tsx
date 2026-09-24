/** Mounts the toast viewport (sonner) on the first interaction instead of during first render,
 * so its chunk stays off the signed-in pages' critical path. Every toast follows a user action,
 * and sonner replays any toast created before its `Toaster` subscribes, so none is lost. */

"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { onFirstInteraction } from "@/lib/first-interaction";
import { OverlayBoundary } from "@/components/shell/overlay-boundary";

const Toaster = dynamic(() => import("@/components/ui/sonner").then((m) => m.Toaster), {
  ssr: false,
});

export function LazyToaster() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => onFirstInteraction(() => setMounted(true)), []);
  // A toast chunk that fails to load must not take the signed-in shell down with it.
  return mounted ? (
    <OverlayBoundary>
      <Toaster position="bottom-right" />
    </OverlayBoundary>
  ) : null;
}
