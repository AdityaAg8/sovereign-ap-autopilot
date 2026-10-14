/** Body of the "?" reference of every keyboard shortcut active on the current screen, sourced
 * from the shortcuts registry so it never drifts from what's actually bound.
 * Loaded on demand by `shortcuts-help-overlay.tsx`, which owns the "?" binding. */

"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { claimOverlay, OVERLAY } from "@/lib/overlay-guard";
import { useShortcutGroups } from "@/lib/shortcuts/registry";

export function ShortcutsHelpDialog({
  open,
  onOpenChange: setOpen,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const groups = useShortcutGroups();

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent
        className="sm:max-w-md"
        // Radix moves focus in right after this; keys held while the chunk loaded are dropped
        // (there is nothing to type into), and an Escape among them closes the reference.
        onOpenAutoFocus={() => {
          if (claimOverlay(OVERLAY.shortcuts).cancelled) setOpen(false);
        }}
      >
        <DialogHeader>
          <DialogTitle>Keyboard shortcuts</DialogTitle>
          <DialogDescription>
            Everything you can do without leaving the keyboard on this screen.
          </DialogDescription>
        </DialogHeader>
        <div className="max-h-[60vh] space-y-5 overflow-y-auto pe-1">
          {groups.map((group) => (
            <div key={group.scope}>
              <h3 className="text-muted-foreground mb-2 text-xs font-semibold tracking-wide uppercase">
                {group.scope}
              </h3>
              <dl className="space-y-1.5">
                {group.entries.map((entry) => (
                  <div key={entry.combo} className="flex items-center justify-between gap-4">
                    <dt className="text-sm">{entry.description}</dt>
                    <dd>
                      <kbd className="border-border bg-muted text-muted-foreground rounded-md border px-1.5 py-0.5 font-mono text-[0.6875rem]">
                        {entry.combo}
                      </kbd>
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
}
