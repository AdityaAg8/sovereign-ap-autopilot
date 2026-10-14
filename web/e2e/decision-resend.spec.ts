/** A decision the API recorded but could not signal to the workflow (503) stays
 * resendable after a reload. The page remembers it per invoice in sessionStorage and offers
 * "Resend decision" while the invoice still sits where that decision left it. The 503 itself is
 * API-side (covered by tests/integration/test_review_signal_retry.py); here the remembered state is
 * seeded directly on the fixture adapter. */
import { test, expect, type Page } from "@playwright/test";

async function openFirstWithStatus(page: Page, status: RegExp): Promise<string> {
  await page.goto("/login");
  await page.getByRole("button", { name: /admin@sovereign\.local/ }).click();
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/inbox$/);
  await page.getByRole("combobox", { name: "Filter by status" }).click();
  const popup = page.getByRole("listbox").filter({ has: page.getByText("All statuses") });
  await popup.getByRole("option", { name: status }).click();
  await expect(popup).toBeHidden();
  await page.getByRole("listbox", { name: "Invoices" }).getByRole("option").first().click();
  await expect(page).toHaveURL(/\/invoices\//);
  return decodeURIComponent(new URL(page.url()).pathname.split("/").pop() ?? "");
}

test("an undelivered approval is offered for resend after a reload, and resent", async ({
  page,
}) => {
  const id = await openFirstWithStatus(page, /^approved$/i);
  await expect(page.getByRole("button", { name: "Resend decision" })).toHaveCount(0);

  await page.evaluate(
    (key) => window.sessionStorage.setItem(key, "approve"),
    `apx.undelivered-decision:${id}`,
  );
  await page.reload();

  const notice = page.getByRole("status").filter({ hasText: "workflow has not received it" });
  await expect(notice).toContainText("Your approval is recorded");
  await notice.getByRole("button", { name: "Resend decision" }).click();
  await expect(page).toHaveURL(/\/review$/);
  const remembered = await page.evaluate(
    (key) => window.sessionStorage.getItem(key),
    `apx.undelivered-decision:${id}`,
  );
  expect(remembered).toBeNull();
});

test("a remembered decision the invoice has moved past is not offered", async ({ page }) => {
  const id = await openFirstWithStatus(page, /^awaiting review$/i);
  await page.evaluate(
    (key) => window.sessionStorage.setItem(key, "reject"),
    `apx.undelivered-decision:${id}`,
  );
  await page.reload();
  await expect(page.getByRole("button", { name: /^Reject/ })).toBeEnabled();
  await expect(page.getByRole("button", { name: "Resend decision" })).toHaveCount(0);
});
