/** Acceptance: every fraud flag on an invoice renders its plain-language explanation, with
 * severity spelled out as a word (not colour alone). Blocked fixture invoices always carry at
 * least one flag, so the first blocked invoice is a stable subject. */
import { test, expect } from "@playwright/test";

test("blocked invoice shows a plain-language explanation for every flag", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /reviewer@sovereign\.local/ }).click();
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/inbox$/);

  await page.getByRole("combobox", { name: "Filter by status" }).click();
  await page.getByRole("option", { name: /blocked/i }).click();
  const firstRow = page.getByRole("option").first();
  await expect(firstRow).toBeVisible();
  await firstRow.click();
  await expect(page).toHaveURL(/\/invoices\//);

  const flagsTab = page.getByRole("tab", { name: /^Flags \((\d+)\)$/ });
  await expect(flagsTab).toBeVisible();
  const count = Number((await flagsTab.textContent())?.match(/\((\d+)\)/)?.[1]);
  expect(count).toBeGreaterThan(0);
  await flagsTab.click();

  const panel = page.getByRole("tabpanel", { name: /Flags/ });
  const items = panel.getByRole("listitem");
  await expect(items).toHaveCount(count);
  for (let index = 0; index < count; index++) {
    const item = items.nth(index);
    // Rule name, a severity word and a full-sentence explanation.
    await expect(item).toContainText(/· (High|Medium|Low)/);
    const explanation = item.locator("p").nth(1);
    await expect(explanation).toHaveText(/\w+ .+\.$/);
  }
});
