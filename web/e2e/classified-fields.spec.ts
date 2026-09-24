/** ADR-0012 Amendment 1 review flow against the fixture adapter: approving an invoice whose
 * document type or line VAT treatment is unresolved is refused (422), the refusal is announced
 * inline and focus lands on the first named field; fixing them with the keyboard alone lets the
 * approval through. */
import { test, expect, type Page } from "@playwright/test";

const INVOICE = "/invoices/inv-adr12-disagreement";

async function signInAndOpen(page: Page) {
  await page.goto("/login");
  await page.getByRole("button", { name: /admin@sovereign\.local/ }).click();
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/inbox$/);
  await page.goto(INVOICE);
  await expect(page.getByRole("heading", { name: "Line 1" })).toBeVisible();
}

const focusedField = (page: Page) =>
  page.evaluate(() => (document.activeElement as HTMLElement | null)?.dataset.field ?? null);

/** Tabs forward until the row for `name` has focus (keyboard only, no clicks). */
async function tabToRow(page: Page, name: string) {
  for (let i = 0; i < 60; i++) {
    if ((await focusedField(page)) === name) return;
    await page.keyboard.press("Tab");
  }
  throw new Error(`never reached the ${name} row by tabbing`);
}

/** Opens the row's editor with Enter, applies `edit`, and commits with Enter. */
async function editRow(page: Page, name: string, edit: () => Promise<void>) {
  await tabToRow(page, name);
  await page.keyboard.press("Enter");
  await edit();
  await page.keyboard.press("Enter");
  await expect(page.locator(`[data-field="${name}"]`)).toBeFocused();
}

test("approve with classified fields missing: refusal is announced, focus moves to the field", async ({
  page,
}) => {
  await signInAndOpen(page);
  await expect(page.locator('[data-field="invoice_type_code"]')).toContainText("Missing");

  await page.keyboard.press("a");

  const alert = page.locator("#approve-error");
  await expect(alert).toContainText("Approval refused");
  await expect(alert).toContainText("Document type");
  await expect(alert).toContainText("Line 2 VAT rate (%)");
  // The message sits inside an always-mounted assertive live region.
  await expect(alert.locator("xpath=..")).toHaveAttribute("aria-live", "assertive");
  await expect(page.locator('[data-field="invoice_type_code"]')).toBeFocused();
  await expect(page).toHaveURL(new RegExp(`${INVOICE}$`));
});

test("fix the type and VAT categories with the keyboard only, then approve", async ({ page }) => {
  await signInAndOpen(page);

  // Start from the refusal so focus is inside the field list, as a reviewer would be.
  await page.keyboard.press("a");
  await expect(page.locator('[data-field="invoice_type_code"]')).toBeFocused();

  await editRow(page, "invoice_type_code", async () => {
    await expect(page.getByRole("combobox", { name: "Edit Document type" })).toBeFocused();
    await page.keyboard.press("ArrowDown"); // 380 · Invoice
  });
  await expect(page.locator('[data-field="invoice_type_code"]')).toContainText("380 · Invoice");

  await editRow(page, "line_items[1].vat_category", async () => {
    await page.keyboard.press("ArrowDown"); // S · Standard
  });
  await expect(page.getByText("Standard rate: 5")).toBeVisible();
  await editRow(page, "line_items[1].vat_rate", async () => {
    await page.keyboard.type("5");
  });

  await editRow(page, "line_items[2].vat_category", async () => {
    for (let i = 0; i < 4; i++) await page.keyboard.press("ArrowDown"); // AE · Reverse charge
  });
  await expect(page.getByText("Reverse charge: 0 or 5")).toBeVisible();
  await editRow(page, "line_items[2].vat_rate", async () => {
    await page.keyboard.type("0");
  });

  await expect(page.locator('[data-field="line_items[2].vat_category"]')).toContainText(
    "AE · Reverse charge",
  );
  // Every corrected field left the refusal list, so the inline error is gone.
  await expect(page.locator("#approve-error")).toHaveCount(0);

  await page.keyboard.press("a");
  await expect(page).toHaveURL(/\/review$/);
  await expect(page.getByText("Invoice approved")).toBeVisible();
});
