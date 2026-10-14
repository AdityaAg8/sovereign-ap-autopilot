/** Review UI acceptance smoke test: login, open a review-queue invoice, fix a field's value, then
 * approve ("review queue: approve, fix a field, reject") — against the dev fixture
 * adapter, complementing smoke.spec.ts's keyboard-only approve path with an explicit field edit. */
import { test, expect } from "@playwright/test";

test("login -> review -> fix a field -> approve", async ({ page }) => {
  await page.goto("/login");

  // The admin fixture account can always approve, regardless of the invoice's amount.
  await page.getByRole("button", { name: /admin@sovereign\.local/ }).click();
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/inbox$/);

  await page.getByRole("link", { name: "Review queue" }).click();
  await expect(page).toHaveURL(/\/review$/);
  const firstRow = page.getByRole("option").first();
  await expect(firstRow).toBeVisible();
  await firstRow.click();
  await expect(page).toHaveURL(/\/invoices\//);

  // Every fixture invoice has an "Invoice number" field regardless of its confidence, so editing
  // it is deterministic across the randomly-generated fixture set.
  await page.getByRole("button", { name: "Edit Invoice number", exact: true }).click();
  const input = page.getByRole("textbox", { name: "Edit Invoice number" });
  await expect(input).toBeFocused();
  await input.fill("INV-2026-9999");
  await input.press("Enter");
  await expect(page.getByText("Invoice number updated")).toBeVisible();
  await expect(page.getByText("INV-2026-9999")).toBeVisible();

  const approve = page.getByRole("button", { name: /Approve/ });
  await expect(approve).toBeEnabled();
  await approve.click();

  await expect(page).toHaveURL(/\/review$/);
  await expect(page.getByText("Invoice approved")).toBeVisible();
});
