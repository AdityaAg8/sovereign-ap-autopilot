/** Body of the global ⌘K command palette: fuzzy navigation to invoices/vendors and quick actions
 * (PRD §10 M5.4). Loaded on demand by `command-palette.tsx`, which owns the Mod+K binding, so cmdk
 * and the dialog primitives stay off every signed-in page's critical path. */

"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { defaultFilter } from "cmdk";
import { FileText, Inbox, LayoutList, LogOut, Moon, Sun, Users } from "lucide-react";
import { useTheme } from "next-themes";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command";
import { useAuth } from "@/lib/auth/auth-context";
import { getApiClient } from "@/lib/api/client";
import type { InvoiceSummary } from "@/lib/api/types";
import { claimOverlay, OVERLAY } from "@/lib/overlay-guard";

interface Action {
  label: string;
  icon: ReactNode;
  run: () => void;
}

export function CommandPaletteDialog({
  open,
  onOpenChange: setOpen,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [query, setQuery] = useState("");
  const [invoices, setInvoices] = useState<readonly InvoiceSummary[]>([]);
  const router = useRouter();
  const { theme, setTheme } = useTheme();
  const { logout } = useAuth();
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!open) return;
    getApiClient()
      .listInvoices({ limit: 200 })
      .then(setInvoices)
      .catch(() => setInvoices([]));
  }, [open]);

  const matches = useMemo(() => {
    if (query.length < 2) return [];
    const q = query.toLowerCase();
    return invoices
      .filter(
        (i) => i.id.toLowerCase().includes(q) || (i.vendorName ?? "").toLowerCase().includes(q),
      )
      .slice(0, 8);
  }, [invoices, query]);

  function go(path: string) {
    setOpen(false);
    router.push(path);
  }

  const dark = theme === "dark";
  const groups: { heading: string; actions: Action[] }[] = [
    {
      heading: "Navigate",
      actions: [
        { label: "Inbox", icon: <Inbox />, run: () => go("/inbox") },
        { label: "Review queue", icon: <LayoutList />, run: () => go("/review") },
        { label: "Vendors", icon: <Users />, run: () => go("/vendors") },
      ],
    },
    {
      heading: "Theme",
      actions: [
        {
          label: `Toggle ${dark ? "light" : "dark"} mode`,
          icon: dark ? <Sun /> : <Moon />,
          run: () => setTheme(dark ? "light" : "dark"),
        },
      ],
    },
    {
      heading: "Session",
      actions: [
        {
          label: "Sign out",
          icon: <LogOut />,
          run: () => void logout().then(() => go("/login")),
        },
      ],
    },
  ];
  // The palette filters everything itself (`shouldFilter={false}`), actions with cmdk's own
  // fuzzy scorer: cmdk's filter drops items that mount after the query changed, which is exactly
  // how invoice matches arrive (fetched on open, or typed while the palette was still loading).
  const shown = groups
    .map((g) => ({
      ...g,
      actions: g.actions.filter((a) => !query || defaultFilter(a.label, query) > 0),
    }))
    .filter((g) => g.actions.length > 0);

  return (
    <CommandDialog
      open={open}
      onOpenChange={setOpen}
      title="Command palette"
      description="Jump to a screen or invoice"
      shouldFilter={false}
      onOpenAutoFocus={(event) => {
        // Focus the search box, then take back the keyboard the opener held: whatever was typed
        // while this chunk loaded lands in the box, as if the palette had been open already.
        event.preventDefault();
        inputRef.current?.focus();
        const { text, cancelled } = claimOverlay(OVERLAY.palette);
        if (cancelled) setOpen(false);
        else if (text) setQuery((prev) => prev + text);
      }}
    >
      <CommandInput
        ref={inputRef}
        placeholder="Search invoices, vendors, or type a command…"
        value={query}
        onValueChange={setQuery}
      />
      <CommandList>
        <CommandEmpty>No results.</CommandEmpty>
        {shown.map((group) => (
          <CommandGroup key={group.heading} heading={group.heading}>
            {group.actions.map((action) => (
              <CommandItem key={action.label} value={action.label} onSelect={action.run}>
                {action.icon} {action.label}
              </CommandItem>
            ))}
          </CommandGroup>
        ))}
        {matches.length > 0 && (
          <>
            {shown.length > 0 && <CommandSeparator />}
            <CommandGroup heading="Invoices">
              {matches.map((inv) => (
                <CommandItem key={inv.id} value={inv.id} onSelect={() => go(`/invoices/${inv.id}`)}>
                  <FileText />
                  <span className="tabular-mono">{inv.id}</span>
                  <span className="text-muted-foreground">
                    {inv.vendorName ?? "Unknown vendor"}
                  </span>
                </CommandItem>
              ))}
            </CommandGroup>
          </>
        )}
      </CommandList>
    </CommandDialog>
  );
}
