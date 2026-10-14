/** Sidebar account menu (signed-in user, roles, sign out). Renders its trigger right away and
 * swaps in the Radix dropdown (account-menu-dropdown.tsx) once it loads on the first
 * interaction, so Radix Menu and its positioning code stay off every signed-in page's critical
 * path; a press or focus on the trigger before then carries over to the real menu. */

"use client";

import type { ComponentProps } from "react";
import { ChevronsUpDown } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { useDeferredControl } from "@/lib/deferred-control";
import type { AuthUser } from "@/lib/api/types";

function initials(name: string): string {
  return name
    .split(" ")
    .map((p) => p[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();
}

/** The avatar button, shared by the stand-in and the real dropdown's trigger. */
export function AccountMenuTrigger({
  user,
  ...props
}: ComponentProps<typeof Button> & { user: AuthUser }) {
  return (
    <Button
      type="button"
      variant="ghost"
      className="hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground aria-expanded:bg-sidebar-accent aria-expanded:text-sidebar-accent-foreground h-10 w-full justify-center gap-2.5 px-1.5 font-normal md:justify-start"
      {...props}
    >
      <Avatar className="size-7">
        <AvatarFallback className="bg-sidebar-primary text-sidebar-primary-foreground text-2xs font-semibold">
          {initials(user.name)}
        </AvatarFallback>
      </Avatar>
      <span aria-hidden className="hidden min-w-0 flex-1 text-start leading-tight md:block">
        <span className="text-sidebar-accent-foreground block truncate text-sm">{user.name}</span>
        <span className="text-sidebar-muted block truncate text-2xs capitalize">
          {user.roles.join(", ")}
        </span>
      </span>
      <ChevronsUpDown aria-hidden className="text-sidebar-muted hidden size-3.5 md:block" />
      <span className="sr-only">Signed in as {user.name}, account menu</span>
    </Button>
  );
}

const loadDropdown = () => import("@/components/shell/account-menu-dropdown");

export function AccountMenu({ user }: { user: AuthUser }) {
  const { module, standInProps, openRequested } = useDeferredControl(loadDropdown);

  if (module) {
    return <module.AccountMenuDropdown user={user} defaultOpen={openRequested} />;
  }

  // Same attributes Radix's closed trigger sets, so styling and semantics match before the swap.
  return (
    <AccountMenuTrigger
      user={user}
      aria-haspopup="menu"
      aria-expanded={false}
      data-state="closed"
      {...standInProps}
    />
  );
}
