/** The topbar account menu's real Radix dropdown (name, roles, sign out), loaded on the first
 * interaction by `account-menu.tsx`. Opens immediately when the stand-in trigger was already
 * activated; a focused stand-in is never swapped (see lib/deferred-control.ts). */

"use client";

import { useRouter } from "next/navigation";
import { LogOut } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { AccountMenuTrigger } from "@/components/shell/account-menu";
import { useAuth } from "@/lib/auth/auth-context";
import type { AuthUser } from "@/lib/api/types";

export function AccountMenuDropdown({
  user,
  defaultOpen,
}: {
  user: AuthUser;
  defaultOpen: boolean;
}) {
  const { logout } = useAuth();
  const router = useRouter();
  return (
    <DropdownMenu defaultOpen={defaultOpen}>
      <DropdownMenuTrigger asChild>
        <AccountMenuTrigger user={user} />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel>
          <div className="flex flex-col">
            <span className="font-medium">{user.name}</span>
            <span className="text-muted-foreground text-xs font-normal capitalize">
              {user.roles.join(", ")}
            </span>
          </div>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={() => void logout().then(() => router.push("/login"))}>
          <LogOut /> Sign out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
