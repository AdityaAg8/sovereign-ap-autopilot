/** Fixed primary navigation for the authenticated shell: Inbox, Review queue, Vendors,
 * Operations (PRD §10 M5.4, M10.3). Collapses to an icon rail below md, highlights the active
 * route and binds the g-i/g-r/g-v/g-o jump shortcuts. */

"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Activity, FileStack, Inbox, LayoutList, Users } from "lucide-react";
import { cn } from "@/lib/utils";
import { useHotkeys } from "@/hooks/use-hotkeys";
import { useEffect, useRef, useState } from "react";

const CHORD_TIMEOUT_MS = 1500;

const NAV = [
  { href: "/inbox", label: "Inbox", icon: Inbox },
  { href: "/review", label: "Review queue", icon: LayoutList },
  { href: "/vendors", label: "Vendors", icon: Users },
  { href: "/operations", label: "Operations", icon: Activity },
] as const;

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [pendingG, setPendingG] = useState(false);
  const chordTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(
    () => () => {
      if (chordTimer.current) clearTimeout(chordTimer.current);
    },
    [],
  );

  function armChord() {
    setPendingG(true);
    if (chordTimer.current) clearTimeout(chordTimer.current);
    chordTimer.current = setTimeout(() => setPendingG(false), CHORD_TIMEOUT_MS);
  }

  useHotkeys([
    {
      combo: "g",
      description: "Go to… (then i/r/v/o)",
      handler: () => armChord(),
    },
    {
      combo: "i",
      description: "Go to inbox",
      handler: () => {
        if (pendingG) router.push("/inbox");
        setPendingG(false);
      },
    },
    {
      combo: "r",
      description: "Go to review queue",
      handler: () => {
        if (pendingG) router.push("/review");
        setPendingG(false);
      },
    },
    {
      combo: "v",
      description: "Go to vendors",
      handler: () => {
        if (pendingG) router.push("/vendors");
        setPendingG(false);
      },
    },
    {
      combo: "o",
      description: "Go to operations",
      handler: () => {
        if (pendingG) router.push("/operations");
        setPendingG(false);
      },
    },
  ]);

  return (
    <nav
      aria-label="Primary"
      className="border-border bg-sidebar flex h-full w-14 shrink-0 flex-col border-e md:w-60"
    >
      <div className="flex h-14 items-center justify-center gap-2 px-4 md:justify-start">
        <FileStack className="text-accent size-5" aria-hidden />
        <span className="hidden text-sm font-semibold tracking-tight md:inline">Sovereign AP</span>
      </div>
      <ul className="flex flex-col gap-0.5 px-2">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname === href || pathname.startsWith(`${href}/`);
          return (
            <li key={href}>
              <Link
                href={href}
                // Prefetch on intent, not on mount: viewport prefetch fired four RSC requests
                // (plus their chunks) on every page load, all of them on the mobile LCP path.
                prefetch={false}
                onMouseEnter={() => router.prefetch(href)}
                onFocus={() => router.prefetch(href)}
                aria-current={active ? "page" : undefined}
                title={label}
                className={cn(
                  "flex items-center justify-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors md:justify-start",
                  active
                    ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                    : "text-sidebar-foreground/80 hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground",
                )}
              >
                <Icon className="size-4" aria-hidden />
                <span className="sr-only md:not-sr-only">{label}</span>
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
