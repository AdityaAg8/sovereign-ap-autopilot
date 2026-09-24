/** Error boundary around each lazily loaded shell piece (command palette, shortcuts help,
 * assistant panel, toast viewport): a chunk that fails to load makes next/dynamic throw during
 * render, which without this takes down the whole signed-in shell. An overlay's failure is
 * reported through the overlay guard (visible, announced message; typed keys held off page
 * hotkeys) and nothing is rendered in its place. */

"use client";

import { Component, type ReactNode } from "react";
import { abandonOverlay, type OverlayOwner } from "@/lib/overlay-guard";

interface Props {
  /** The overlay the guard is holding keys for; omitted for pieces nobody opens (the toaster). */
  owner?: OverlayOwner;
  /** Lets the opener reset its open state, so the next request is a fresh open. */
  onFail?: () => void;
  children: ReactNode;
}

export class OverlayBoundary extends Component<Props, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }

  componentDidCatch() {
    if (this.props.owner) abandonOverlay(this.props.owner);
    this.props.onFail?.();
  }

  render() {
    return this.state.failed ? null : this.props.children;
  }
}
