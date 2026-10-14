/** M5.4 acceptance smoke test: login, land on the inbox, open an invoice and approve it with the
 * keyboard — all against the dev fixture adapter. Runs against a production build
 * via playwright.config.ts's webServer. */
import { test, expect } from "@playwright/test";

test("login -> inbox -> open invoice -> keyboard approve", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/login$/);

  // Sign in as the admin fixture account so the approval limit never blocks the keyboard step.
  await page.getByRole("button", { name: /admin@sovereign\.local/ }).click();
  await page.getByRole("button", { name: "Sign in" }).click();

  await expect(page).toHaveURL(/\/inbox$/);
  await expect(page.getByRole("heading", { name: "Inbox", level: 1 })).toBeVisible();

  // Filter to invoices awaiting review so the one we open is guaranteed to be approvable.
  // Scope to the filter popup: inbox rows are options too, and the popup opens only once the
  // deferred Select has loaded.
  await page.getByRole("combobox", { name: "Filter by status" }).click();
  const statusPopup = page.getByRole("listbox").filter({ has: page.getByText("All statuses") });
  await statusPopup.getByRole("option", { name: /^awaiting review$/i }).click();
  await expect(statusPopup).toBeHidden();

  const firstRow = page.getByRole("option").first();
  await expect(firstRow).toBeVisible();
  const invoiceId = await firstRow.getAttribute("id");
  await firstRow.click();

  await expect(page).toHaveURL(/\/invoices\//);
  await expect(page.getByRole("button", { name: /Approve/ })).toBeEnabled();

  // Approve with the "a" shortcut (review queue keyboard shortcuts). Navigating here
  // unmounted the row that had focus, so the keystroke lands on the page, not a stale control.
  await page.keyboard.press("a");

  await expect(page).toHaveURL(/\/review$/);
  await expect(page.getByText("Invoice approved")).toBeVisible();
  expect(invoiceId).toBeTruthy();
});
