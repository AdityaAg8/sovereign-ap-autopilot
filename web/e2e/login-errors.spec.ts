/** Login error copy: a typed 401 shows "no account matches", while an adapter that cannot load
 * (its code chunk fails, like an unreachable server) shows the retryable server message. */
import { expect, test } from "@playwright/test";

test("unknown email is reported as a missing account", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Email").fill("nobody@example.com");
  await page.getByLabel("Password").fill("x");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByText(/No account matches that email/)).toBeVisible();
});

test("a failed API load is reported as a server error, not a bad account", async ({ page }) => {
  await page.goto("/login");
  await expect(page.getByRole("button", { name: "Sign in" })).toBeEnabled();
  // Every chunk the page needs is loaded by now; fail the lazily imported API adapter.
  await page.route("**/_next/static/chunks/**", (route) => route.abort("connectionrefused"));
  await page.getByLabel("Email").fill("admin@sovereign.local");
  await page.getByLabel("Password").fill("x");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByText("Couldn't reach the server, try again.")).toBeVisible();
  await expect(page.getByText(/No account matches/)).toHaveCount(0);
});

test("after hydration, Enter in the password field signs in", async ({ page }) => {
  await page.goto("/login");
  await expect(page.getByRole("button", { name: "Sign in" })).toBeEnabled();
  await page.getByLabel("Email").fill("admin@sovereign.local");
  await page.getByLabel("Password").fill("anything");
  await page.getByLabel("Password").press("Enter");
  await expect(page).toHaveURL(/\/inbox$/);
});
