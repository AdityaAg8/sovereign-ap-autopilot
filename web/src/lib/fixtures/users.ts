/** Fixture users for the dev-only login flow (approver, admin). Any
 * password is accepted for a listed email so Playwright and manual dev logins stay simple; a real
 * password check happens server-side once apx/api's JWT auth lands. */
import type { AuthUser } from "@/lib/api/types";

export const FIXTURE_USERS: readonly AuthUser[] = [
  {
    id: "u-reviewer",
    name: "Fatima Al Zaabi",
    email: "reviewer@sovereign.local",
    roles: ["reviewer"],
    approvalLimitAed: null,
  },
  {
    id: "u-approver",
    name: "Omar Hassan",
    email: "approver@sovereign.local",
    roles: ["approver"],
    approvalLimitAed: 250000,
  },
  {
    id: "u-admin",
    name: "Layla Suleiman",
    email: "admin@sovereign.local",
    roles: ["admin", "approver"],
    approvalLimitAed: 1000000,
  },
];

export function findUserByEmail(email: string): AuthUser | undefined {
  return FIXTURE_USERS.find((u) => u.email.toLowerCase() === email.toLowerCase());
}
