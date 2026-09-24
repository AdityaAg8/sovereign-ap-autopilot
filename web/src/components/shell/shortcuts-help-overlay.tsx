/** "?" opens a reference of every keyboard shortcut active on the current screen, sourced from
 * the shortcuts registry so it never drifts from what's actually bound (PRD §10 M5.4). */

"use client";

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { useHotkeys } from "@/hooks/use-hotkeys";
import { useShortcutGroups } from "@/lib/shortcuts/registry";

export function ShortcutsHelpOverlay() {
  const [open, setOpen] = useState(false);
  const groups = useShortcutGroups();

  useHotkeys([
    { combo: "?", description: "Show keyboard shortcuts", handler: () => setOpen((v) => !v) },
  ]);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-md">
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
