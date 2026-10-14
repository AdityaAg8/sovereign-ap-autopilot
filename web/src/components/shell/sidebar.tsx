/** Fixed primary navigation for the authenticated shell, on the dark "spine" surface: the brand,
 * grouped links (Inbox and Review queue with live counts, Vendors, Operations) and the account
 * menu. Collapses to an icon rail below md and binds the g-i/g-r/g-v/g-o jump shortcuts. */

"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Activity, Inbox, LayoutList, Users, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { useHotkeys } from "@/hooks/use-hotkeys";
import { useEffect, useRef, useState } from "react";
import { AccountMenu } from "@/components/shell/account-menu";
import { BrandMark } from "@/components/shell/brand-mark";
import { getApiClient } from "@/lib/api/client";
import { useAuth } from "@/lib/auth/auth-context";
import { formatCount } from "@/lib/format";

const CHORD_TIMEOUT_MS = 1500;
/** Same cap the inbox and review pages fetch with; a full page reads as "2,000+". */
const COUNT_LIMIT = 2000;

type CountKey = "all" | "review";

interface NavItem {
  href: string;
  label: string;
  icon: LucideIcon;
  count?: CountKey;
}

const GROUPS: readonly { id: string; label: string; items: readonly NavItem[] }[] = [
  {
    id: "invoices",
    label: "Invoices",
    items: [
      { href: "/inbox", label: "Inbox", icon: Inbox, count: "all" },
      { href: "/review", label: "Review queue", icon: LayoutList, count: "review" },
    ],
  },
  { id: "records", label: "Records", items: [{ href: "/vendors", label: "Vendors", icon: Users }] },
  {
    id: "system",
    label: "System",
    items: [{ href: "/operations", label: "Operations", icon: Activity }],
  },
];

/** Invoice totals for the nav badges, fetched once the page's own work is done (idle), and again
 * on every route change so a decision taken on one screen shows up in the queue count. */
function useNavCounts(pathname: string): Record<CountKey, string> | null {
  const [counts, setCounts] = useState<Record<CountKey, string> | null>(null);
  useEffect(() => {
    let cancelled = false;
    const run = () => {
      getApiClient()
        .listInvoices({ limit: COUNT_LIMIT })
        .then((rows) => {
          if (cancelled) return;
          const show = (n: number) => (n >= COUNT_LIMIT ? `${formatCount(n)}+` : formatCount(n));
          setCounts({
            all: show(rows.length),
            review: show(rows.filter((r) => r.status === "awaiting_review").length),
          });
        })
        .catch(() => {
          // Counts are a convenience: on failure the badges simply stay hidden.
        });
    };
    const idle =
      typeof window.requestIdleCallback === "function"
        ? window.requestIdleCallback(run, { timeout: 2000 })
        : window.setTimeout(run, 300);
    return () => {
      cancelled = true;
      if (typeof window.cancelIdleCallback === "function") window.cancelIdleCallback(idle);
      else window.clearTimeout(idle);
    };
  }, [pathname]);
  return counts;
}

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const { user } = useAuth();
  const counts = useNavCounts(pathname);
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
      data-surface="spine"
      className="bg-sidebar text-sidebar-foreground border-sidebar-border flex h-full w-14 shrink-0 flex-col border-e md:w-60"
    >
      <div className="flex h-14 shrink-0 items-center justify-center gap-2.5 px-3.5 md:justify-start">
        <BrandMark />
        <span className="hidden min-w-0 leading-tight md:block">
          <span className="text-sidebar-accent-foreground block text-sm font-semibold tracking-[-0.01em]">
            Sovereign AP
          </span>
          <span className="text-sidebar-muted block text-2xs">Accounts payable</span>
        </span>
      </div>
      <div className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto px-2 pt-4">
        {GROUPS.map((group) => (
          <div key={group.id}>
            <p
              id={`nav-${group.id}`}
              className="text-sidebar-muted mb-1 hidden px-3 text-2xs font-medium md:block"
            >
              {group.label}
            </p>
            <ul aria-labelledby={`nav-${group.id}`} className="flex flex-col gap-0.5">
              {group.items.map(({ href, label, icon: Icon, count }) => {
                const active = pathname === href || pathname.startsWith(`${href}/`);
                const value = count && counts ? counts[count] : null;
                const countId = `nav-count-${href.slice(1)}`;
                return (
                  <li key={href}>
                    <Link
                      href={href}
                      // Prefetch on intent, not on mount: viewport prefetch fired four RSC
                      // requests (plus their chunks) on every page load, all on the LCP path.
                      prefetch={false}
                      onMouseEnter={() => router.prefetch(href)}
                      onFocus={() => router.prefetch(href)}
                      aria-current={active ? "page" : undefined}
                      aria-describedby={value ? countId : undefined}
                      title={label}
                      className={cn(
                        "relative flex h-8 items-center justify-center gap-2.5 rounded-md px-3 text-sm transition-colors duration-150 md:justify-start",
                        active
                          ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                          : "hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground",
                      )}
                    >
                      {active && (
                        <span
                          aria-hidden
                          className="bg-sidebar-primary absolute inset-y-2 start-0 w-0.5 rounded-full"
                        />
                      )}
                      <Icon
                        className={cn("size-4", active ? "text-sidebar-primary" : "opacity-75")}
                        aria-hidden
                      />
                      <span className="sr-only md:not-sr-only">{label}</span>
                      {value && (
                        <>
                          <span
                            aria-hidden
                            className={cn(
                              "tabular-mono ms-auto hidden text-2xs md:inline",
                              active ? "text-sidebar-accent-foreground" : "text-sidebar-muted",
                            )}
                          >
                            {value}
                          </span>
                          <span id={countId} hidden>
                            {count === "review" ? `${value} waiting` : `${value} invoices`}
                          </span>
                        </>
                      )}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>
      {user && (
        <div className="border-sidebar-border shrink-0 border-t p-2">
          <AccountMenu user={user} />
        </div>
      )}
    </nav>
  );
}
