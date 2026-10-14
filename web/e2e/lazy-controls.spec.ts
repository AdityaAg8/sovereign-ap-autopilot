/** The overlays and popup controls that now load on demand (command palette, shortcut
 * help, assistant panel, inbox filter pickers, account menu) still work keyboard-first on a
 * fresh page load, before and after their chunks arrive, and swapping them in shifts nothing
 * (CLS 0). Runs against the production build via playwright.config.ts's webServer. */
import { test, expect, type Page } from "@playwright/test";

/** Signs in, then reloads /inbox so the test starts from a cold page whose lazy chunks have not
 * been requested yet (the session survives the reload in sessionStorage). */
async function freshInbox(page: Page) {
  await page.goto("/login");
  await page.getByRole("button", { name: /admin@sovereign\.local/ }).click();
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/inbox$/);
  await page.reload();
  await expect(page.getByRole("heading", { name: "Inbox", level: 1 })).toBeVisible();
}

async function layoutShift(page: Page): Promise<number> {
  return page.evaluate(
    () =>
      new Promise<number>((resolve) => {
        let total = 0;
        new PerformanceObserver((list) => {
          for (const entry of list.getEntries() as (PerformanceEntry & {
            value: number;
            hadRecentInput: boolean;
          })[]) {
            if (!entry.hadRecentInput) total += entry.value;
          }
        }).observe({ type: "layout-shift", buffered: true });
        setTimeout(() => resolve(total), 300);
      }),
  );
}

test("global shortcuts open the lazily loaded overlays on a fresh page", async ({ page }) => {
  await freshInbox(page);

  await page.keyboard.press("?");
  await expect(page.getByRole("dialog", { name: "Keyboard shortcuts" })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog", { name: "Keyboard shortcuts" })).toBeHidden();

  await page.keyboard.press("ControlOrMeta+k");
  const palette = page.getByPlaceholder("Search invoices, vendors, or type a command…");
  await expect(palette).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(palette).toBeHidden();

  await page.keyboard.press("c");
  await expect(page.getByRole("dialog", { name: "Assistant" })).toBeVisible();
});

test("the status filter opens from the keyboard and the upgrade shifts nothing", async ({
  page,
}) => {
  await freshInbox(page);
  const status = page.getByRole("combobox", { name: "Filter by status" });
  await expect(status).toHaveText("All statuses");

  // Keyboard only: focus the picker (first interaction, so its Radix chunk starts loading) and
  // open it straight away, whichever of the stand-in or the real trigger has focus by then.
  await status.focus();
  await page.keyboard.press("Enter");
  // Anchored: inbox rows are options too, and their names include their status.
  const option = page.getByRole("option", { name: /^awaiting review$/i });
  await expect(option).toBeVisible();
  // Space, not Enter: Enter on an open option also reaches the inbox's window-level Enter hotkey
  // (open the highlighted invoice). That pre-dates the lazy loading and is tracked separately.
  await option.press(" ");

  await expect(status).toHaveText(/awaiting review/i);
  await expect(status).toBeFocused();
  const rows = page.getByRole("option");
  await expect(rows.first()).toBeVisible();
  await expect(rows.filter({ hasNotText: /awaiting review/i })).toHaveCount(0);
  expect(await layoutShift(page)).toBe(0);
});

test("the account menu opens on the first click and signs out", async ({ page }) => {
  await freshInbox(page);
  await page.getByRole("button", { name: /account menu/ }).click();
  await page.getByRole("menuitem", { name: "Sign out" }).click();
  await expect(page).toHaveURL(/\/login$/);
});

// Radix's real Select trigger carries a `dir` attribute; the stand-in does not.
const REAL_TRIGGER_ATTR = "dir";

/** Opens /inbox with a stored fixture session and no prior input of any kind (no sign-in clicks,
 * no pointer over the page), so the first interaction is whatever the test does first. */
async function untouchedInbox(page: Page) {
  await page.addInitScript(() => {
    const user = {
      id: "u-admin",
      name: "Layla Suleiman",
      email: "admin@sovereign.local",
      roles: ["admin", "approver"],
      approvalLimitAed: 1000000,
    };
    const session = { user, token: `fixture-token.u-admin.${Date.now()}` };
    window.sessionStorage.setItem("sovereign-ap.session", JSON.stringify(session));
  });
  await page.goto("/inbox");
  await expect(page.getByRole("heading", { name: "Inbox", level: 1 })).toBeVisible();
}

test("a focused stand-in is not swapped out from under the keyboard", async ({ page }) => {
  await untouchedInbox(page);
  const status = page.getByRole("combobox", { name: "Filter by status" });
  await expect(status).not.toHaveAttribute(REAL_TRIGGER_ATTR);
  // Focusing is the first interaction: the real select's chunk loads while the stand-in is
  // focused, and the stand-in must keep focus rather than be swapped out from under it.
  await status.focus();
  await page.waitForLoadState("networkidle");
  await page.waitForTimeout(300);
  await expect(status).toBeFocused();
  await expect(status).not.toHaveAttribute(REAL_TRIGGER_ATTR);

  // Tabbing away lets the swap happen; tabbing back lands on the real trigger.
  await page.keyboard.press("Tab");
  await expect(status).toHaveAttribute(REAL_TRIGGER_ATTR, "ltr");
  await page.keyboard.press("Shift+Tab");
  await expect(status).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("option", { name: /^awaiting review$/i })).toBeVisible();
});

test.describe("touch", () => {
  test.use({ hasTouch: true });

  test("a touch that turns into a scroll does not freeze the stand-in", async ({ page }) => {
    await freshInbox(page);
    const status = page.getByRole("combobox", { name: "Filter by status" });
    // A touch press the browser takes over for scrolling: pointerdown, then pointercancel (no
    // pointerup, no click, no focus).
    await status.dispatchEvent("pointerdown", { pointerType: "touch", isPrimary: true });
    await page.evaluate(() =>
      window.dispatchEvent(new PointerEvent("pointercancel", { pointerType: "touch" })),
    );
    await expect(status).toHaveAttribute(REAL_TRIGGER_ATTR, "ltr");
    await status.tap();
    await expect(page.getByRole("option", { name: /^awaiting review$/i })).toBeVisible();
  });
});

test("the invoice document renders in the viewer and pages with its field boxes", async ({
  page,
}) => {
  await freshInbox(page);
  await page.getByRole("option").first().click();
  await expect(page).toHaveURL(/\/invoices\//);
  await expect(page.getByRole("img", { name: "Invoice document, page 1" })).toBeVisible();
  await expect(page.getByText("Couldn't render this document")).toHaveCount(0);
  await page.getByRole("button", { name: "Next page" }).click();
  await expect(page.getByRole("img", { name: "Invoice document, page 2" })).toBeVisible();
  await expect(page.getByText("Couldn't render this document")).toHaveCount(0);
  expect(await layoutShift(page)).toBe(0);
});
