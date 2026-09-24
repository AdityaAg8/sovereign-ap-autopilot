/** Global ⌘K command palette: fuzzy navigation to invoices/vendors and quick actions (PRD §10
 * M5.4). Opens on Mod+K from anywhere in the authenticated shell. */

"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
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
import { useHotkeys } from "@/hooks/use-hotkeys";
import { useAuth } from "@/lib/auth/auth-context";
import { getApiClient } from "@/lib/api/client";
import type { InvoiceSummary } from "@/lib/api/types";

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [invoices, setInvoices] = useState<readonly InvoiceSummary[]>([]);
  const router = useRouter();
  const { theme, setTheme } = useTheme();
  const { logout } = useAuth();

  useHotkeys([
    { combo: "mod+k", description: "Open the command palette", handler: () => setOpen((v) => !v) },
  ]);

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

  return (
    <CommandDialog
      open={open}
      onOpenChange={setOpen}
      title="Command palette"
      description="Jump to a screen or invoice"
    >
      <CommandInput
        placeholder="Search invoices, vendors, or type a command…"
        value={query}
        onValueChange={setQuery}
      />
      <CommandList>
        <CommandEmpty>No results.</CommandEmpty>
        <CommandGroup heading="Navigate">
          <CommandItem onSelect={() => go("/inbox")}>
            <Inbox /> Inbox
          </CommandItem>
          <CommandItem onSelect={() => go("/review")}>
            <LayoutList /> Review queue
          </CommandItem>
          <CommandItem onSelect={() => go("/vendors")}>
            <Users /> Vendors
          </CommandItem>
        </CommandGroup>
        <CommandGroup heading="Theme">
          <CommandItem onSelect={() => setTheme(theme === "dark" ? "light" : "dark")}>
            {theme === "dark" ? <Sun /> : <Moon />} Toggle {theme === "dark" ? "light" : "dark"}{" "}
            mode
          </CommandItem>
        </CommandGroup>
        <CommandGroup heading="Session">
          <CommandItem onSelect={() => void logout().then(() => go("/login"))}>
            <LogOut /> Sign out
          </CommandItem>
        </CommandGroup>
        {matches.length > 0 && (
          <>
            <CommandSeparator />
            <CommandGroup heading="Invoices">
              {matches.map((inv) => (
                <CommandItem key={inv.id} onSelect={() => go(`/invoices/${inv.id}`)}>
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
