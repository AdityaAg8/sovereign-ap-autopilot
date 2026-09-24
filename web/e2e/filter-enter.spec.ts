/** Regression spec for the inbox Enter shortcut: picking a status-filter option with Enter must only
 * apply the filter, not also fire the page-level "open highlighted invoice" hotkey, while Enter on
 * the highlighted row still opens it (PRD §10 M5.4 keyboard-first flows). Also covers "?" typed
 * with a real Shift press, which the hotkey matcher must accept. */
import { test, expect, type Page } from "@playwright/test";

async function signIn(page: Page) {
  await page.goto("/login");
  await page.getByRole("button", { name: /admin@sovereign\.local/ }).click();
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/inbox$/);
  await expect(page.getByRole("option").first()).toBeVisible();
}

test("Enter on a status-filter option applies the filter and stays on the inbox", async ({
  page,
}) => {
  await signIn(page);
  const trigger = page.getByRole("combobox", { name: "Filter by status" });
  const rows = page.getByRole("listbox", { name: "Invoices" }).getByRole("option");
  const before = await rows.allInnerTexts();
  expect(before.some((text) => !/awaiting review/i.test(text))).toBe(true);
  // A stray "open highlighted invoice" would request an /invoices/ route; record any such request.
  const invoiceRequests: string[] = [];
  page.on("request", (req) => {
    if (new URL(req.url()).pathname.startsWith("/invoices/")) invoiceRequests.push(req.url());
  });

  // Keyboard only: focus the trigger, open it with Enter, ArrowDown to the option, then Enter.
  // Park the cursor first: the popup opens where Sign in was clicked, and Chromium's hover on
  // newly laid-out content would otherwise steal the highlight from the keyboard.
  await page.mouse.move(0, 0);
  await trigger.focus();
  await page.keyboard.press("Enter");
  const popup = page.getByRole("listbox").filter({ has: page.getByText("All statuses") });
  const options = popup.getByRole("option");
  await expect(options.first()).toBeVisible();
  const labels = (await options.allInnerTexts()).map((text) => text.trim().toLowerCase());
  const targetIndex = labels.indexOf("awaiting review");
  expect(labels[0]).toBe("all statuses");
  expect(targetIndex).toBeGreaterThan(0);
  // Radix highlights the current value once open; wait for it, then assert every single step.
  await expect(options.nth(0)).toHaveAttribute("data-highlighted", "");
  for (let i = 1; i <= targetIndex; i++) {
    await page.keyboard.press("ArrowDown");
    await expect(options.nth(i)).toHaveAttribute("data-highlighted", "");
  }
  await page.keyboard.press("Enter");

  await expect(popup).toBeHidden();
  await expect(trigger).toHaveText(/awaiting review/i);
  // Settled: the filtered rows rendered in the same commit the stray hotkey would have navigated.
  await expect
    .poll(async () => {
      const texts = await rows.allInnerTexts();
      return texts.length > 0 && texts.every((text) => /awaiting review/i.test(text));
    })
    .toBe(true);
  await expect(page).toHaveURL(/\/inbox$/);
  expect(invoiceRequests).toEqual([]);
});

test("Enter on the highlighted inbox row still opens that invoice", async ({ page }) => {
  await signIn(page);
  const list = page.getByRole("listbox", { name: "Invoices" });
  const highlighted = list.locator('[role="option"][aria-selected="true"]');
  await list.focus();
  const first = await highlighted.getAttribute("id");
  await page.keyboard.press("j");
  await expect(highlighted).not.toHaveAttribute("id", first ?? "");
  const id = await highlighted.getAttribute("id");
  expect(id).toBeTruthy();
  await page.keyboard.press("Enter");

  await expect(page).toHaveURL(new RegExp(`/invoices/${id!.replace(/^inbox-row-/, "")}$`));
});

test("Shift+/ on the inbox opens the keyboard shortcuts dialog", async ({ page }) => {
  await signIn(page);
  await page.keyboard.press("Shift+Slash");
  const dialog = page.getByRole("dialog", { name: "Keyboard shortcuts" });
  await expect(dialog).toBeVisible();
  await page.keyboard.press("Shift+Slash");
  await expect(dialog).toBeHidden();
});
