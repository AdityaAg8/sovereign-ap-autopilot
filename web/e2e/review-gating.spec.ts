/** REVIEW.md C-01: the review action bar offers Approve and Reject only while an invoice is
 * awaiting review. An invoice still in the pipeline (here fraud-checked) shows them disabled, so a
 * reviewer can never record a decision the workflow would not hear. Runs on the fixture adapter. */
import { test, expect, type Page } from "@playwright/test";

async function openFirstWithStatus(page: Page, status: RegExp) {
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
}

test("an in-flight invoice offers no review decision", async ({ page }) => {
  await openFirstWithStatus(page, /^fraud checked$/i);
  await expect(page.getByRole("button", { name: /^Reject/ })).toBeDisabled();
  await expect(page.getByRole("button", { name: /^Approve/ })).toBeDisabled();
});

test("an invoice awaiting review offers approve and reject", async ({ page }) => {
  await openFirstWithStatus(page, /^awaiting review$/i);
  await expect(page.getByRole("button", { name: /^Reject/ })).toBeEnabled();
  await expect(page.getByRole("button", { name: /^Approve/ })).toBeEnabled();
});
