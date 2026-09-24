/** Money-path regression for the lazily loaded overlays: between ⌘K / `c` / `?` and the overlay
 * having focus (its chunk may still be loading), fast-typed keys must never reach page hotkeys,
 * so "⌘K acme" on an invoice cannot fire `a` (Approve). Chunk requests are throttled to make the
 * race deterministic; the held keys end up in the overlay's input, and a chunk that fails or
 * outlasts the hold's cap leaves the shell alive and the typing swallowed. Also covers the inbox's
 * internal-id fallback and quick navigation between two invoices (review of PR #48). */
import { test, expect, type BrowserContext, type Page } from "@playwright/test";

const CHUNK_DELAY_MS = 1500;

/** A stored fixture session for the admin (whose approval limit never blocks Approve), so a page
 * can open on an invoice with no sign-in interaction warming the lazy chunks first. */
async function seedAdminSession(context: BrowserContext) {
  await context.addInitScript(() => {
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
}

/** IDs (and supplier numbers) of the first `n` invoices awaiting review, read from the inbox DOM
 * in a throwaway page: no clicks, and the page the test drives stays untouched. The first such
 * invoice is approvable (smoke.spec.ts approves it). */
async function awaitingReview(
  context: BrowserContext,
  n: number,
): Promise<{ id: string; number: string }[]> {
  const scout = await context.newPage();
  await scout.goto("/inbox");
  const rows = scout
    .getByRole("listbox", { name: "Invoices" })
    .getByRole("option")
    .filter({ hasText: /awaiting review/i });
  await expect(rows.first()).toBeVisible();
  const found: { id: string; number: string }[] = [];
  for (let i = 0; i < n; i++) {
    const row = rows.nth(i);
    const id = (await row.getAttribute("id"))!.replace(/^inbox-row-/, "");
    const number = (await row.getByTestId("inbox-invoice-number").textContent())!.trim();
    found.push({ id, number });
  }
  await scout.close();
  return found;
}

/** Opens the invoice on a cold page, waits for it to settle, then delays every further chunk
 * request so an overlay's lazy chunk is guaranteed to arrive well after the keys typed next. */
async function coldInvoiceWithSlowChunks(page: Page, id: string, delayMs = CHUNK_DELAY_MS) {
  await page.goto(`/invoices/${id}`);
  await expect(page.getByRole("button", { name: /Approve/ })).toBeEnabled();
  await page.waitForLoadState("networkidle");
  await page.route(CHUNKS, async (route) => {
    await new Promise((resolve) => setTimeout(resolve, delayMs));
    await route.continue().catch(() => {});
  });
}

const CHUNKS = "**/_next/static/chunks/**";

/** Opens the invoice on a cold page, then makes every further chunk request fail. */
async function coldInvoiceWithFailingChunks(page: Page, id: string) {
  await page.goto(`/invoices/${id}`);
  await expect(page.getByRole("button", { name: /Approve/ })).toBeEnabled();
  await page.waitForLoadState("networkidle");
  await page.route(CHUNKS, (route) => route.abort("connectionrefused"));
}

async function expectNotApproved(page: Page, id: string) {
  await expect(page).toHaveURL(new RegExp(`/invoices/${id}$`));
  await expect(page.getByText("Invoice approved")).toHaveCount(0);
  await expect(page.getByText("Awaiting review", { exact: true }).first()).toBeVisible();
  // An open modal overlay hides the page from the accessibility tree, hence `includeHidden`.
  await expect(page.getByRole("button", { name: /Approve/, includeHidden: true })).toBeEnabled();
}

for (const mod of ["Meta", "Control"] as const) {
  test(`${mod}+K then typing at once fills the palette and never approves`, async ({
    page,
    context,
  }) => {
    await seedAdminSession(context);
    const [target] = await awaitingReview(context, 1);
    await coldInvoiceWithSlowChunks(page, target!.id);

    await page.keyboard.press(`${mod}+k`);
    await page.keyboard.type("acme");
    await page.keyboard.press("a");
    await page.keyboard.press("r");
    await page.keyboard.press("f");

    const palette = page.getByPlaceholder("Search invoices, vendors, or type a command…");
    await expect(palette).toBeFocused({ timeout: CHUNK_DELAY_MS * 4 });
    await expect(palette).toHaveValue("acmearf");
    await expectNotApproved(page, target!.id);

    // The hold ends once the palette has focus: closed again, the page's `a` approves as usual.
    await page.keyboard.press("Escape");
    await expect(palette).toBeHidden();
    await page.keyboard.press("a");
    await expect(page).toHaveURL(/\/review$/);
    await expect(page.getByText("Invoice approved")).toBeVisible();
  });
}

test("`c` then typing at once fills the assistant and never approves", async ({
  page,
  context,
}) => {
  await seedAdminSession(context);
  const [target] = await awaitingReview(context, 1);
  await coldInvoiceWithSlowChunks(page, target!.id);

  await page.keyboard.press("c");
  await page.keyboard.type("are these duplicates? a");

  const input = page.getByRole("textbox", { name: "Ask the assistant" });
  await expect(input).toBeFocused({ timeout: CHUNK_DELAY_MS * 4 });
  await expect(input).toHaveValue("are these duplicates? a");
  await expectNotApproved(page, target!.id);

  await page.keyboard.press("Escape");
  await expect(input).toBeHidden();
  await page.keyboard.press("a");
  await expect(page).toHaveURL(/\/review$/);
});

test("`?` then a page key at once does not approve, and Escape still closes it", async ({
  page,
  context,
}) => {
  await seedAdminSession(context);
  const [target] = await awaitingReview(context, 1);
  await coldInvoiceWithSlowChunks(page, target!.id);

  await page.keyboard.press("?");
  await page.keyboard.press("a");
  await page.keyboard.press("Escape");
  // Escape while loading cancels the request: the reference closes itself as soon as it mounts.
  await page.waitForTimeout(CHUNK_DELAY_MS * 2);
  await expect(page.getByRole("dialog", { name: "Keyboard shortcuts" })).toBeHidden();
  await expectNotApproved(page, target!.id);
});

test("an invoice with no supplier number shows its marked internal id in the inbox", async ({
  page,
  context,
}) => {
  await seedAdminSession(context);
  const [target] = await awaitingReview(context, 1);
  await page.goto(`/invoices/${target!.id}`);
  const edit = page.getByRole("textbox", { name: "Edit Invoice number" });
  await page.getByRole("button", { name: "Edit Invoice number", exact: true }).click();
  await edit.fill("");
  await edit.press("Enter");
  await expect(page.getByText("Invoice number updated")).toBeVisible();

  await page.getByRole("link", { name: "Inbox" }).click();
  await expect(page).toHaveURL(/\/inbox$/);
  const row = page.locator(`[id="inbox-row-${target!.id}"]`);
  const cell = row.getByTestId("inbox-invoice-number");
  await expect(cell).toHaveText(`Internal ID ${target!.id}`);
  await expect(cell).toHaveAttribute("title", /internal ID/);
  const marked = cell.getByTestId("inbox-internal-id");
  await expect(marked).toHaveClass(/text-muted-foreground/);
  // The id is still searchable, and no row falls back to an unlabelled dash.
  await page.getByRole("textbox", { name: "Search invoices" }).fill(target!.id);
  await expect(page.getByRole("listbox", { name: "Invoices" }).getByRole("option")).toHaveCount(1);
  await expect(page.getByTestId("inbox-invoice-number").filter({ hasText: /^—$/ })).toHaveCount(0);
});

test("moving quickly between two invoices shows and approves only the current one", async ({
  page,
  context,
}) => {
  await seedAdminSession(context);
  // The invoice navigated to (and approved) is the known-approvable first one.
  const [second, first] = await awaitingReview(context, 2);
  await page.goto(`/invoices/${first!.id}`);
  await expect(page.getByTestId("field-value-invoice_number")).toHaveText(first!.number);

  // Straight on to the second invoice via the palette, without waiting for anything between.
  await page.keyboard.press("ControlOrMeta+k");
  const palette = page.getByPlaceholder("Search invoices, vendors, or type a command…");
  await expect(palette).toBeFocused();
  await palette.fill(second!.id);
  await expect(page.getByRole("option", { name: new RegExp(second!.id) })).toBeVisible();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(new RegExp(`/invoices/${second!.id}$`));
  await expect(page.getByTestId("field-value-invoice_number")).toHaveText(second!.number);
  await page.waitForTimeout(500);
  await expect(page.getByTestId("field-value-invoice_number")).toHaveText(second!.number);

  await page.keyboard.press("a");
  await expect(page).toHaveURL(/\/review$/);
  await page.getByRole("link", { name: "Inbox" }).click();
  await page.getByRole("textbox", { name: "Search invoices" }).fill(second!.id);
  await expect(page.locator(`[id="inbox-row-${second!.id}"]`)).toContainText(/approved/i);
  await page.getByRole("textbox", { name: "Search invoices" }).fill(first!.id);
  await expect(page.locator(`[id="inbox-row-${first!.id}"]`)).toContainText(/awaiting review/i);
});

for (const ending of ["Escape", "idle"] as const) {
  test(`a palette that fails to load keeps the typing off page hotkeys until ${ending}`, async ({
    page,
    context,
  }) => {
    await seedAdminSession(context);
    const [target] = await awaitingReview(context, 1);
    await coldInvoiceWithFailingChunks(page, target!.id);

    await page.keyboard.press("ControlOrMeta+k");
    await page.keyboard.type("acme");
    await page.keyboard.press("a");

    await expect(page.getByRole("status").filter({ hasText: /command palette/ })).toHaveText(
      "Couldn't open the command palette. Reload the page to try again.",
    );
    // Still typing after the failure: the hold keeps swallowing keys while they keep coming.
    for (let i = 0; i < 4; i++) {
      await page.keyboard.press("a");
      await page.waitForTimeout(300);
    }
    await expectNotApproved(page, target!.id);
    // The shell survived the failed chunk: navigation and the invoice are still there.
    await expect(page.getByRole("navigation", { name: "Primary" })).toBeVisible();
    await expect(page.getByRole("dialog")).toHaveCount(0);

    if (ending === "Escape") await page.keyboard.press("Escape");
    else await page.waitForTimeout(1200);
    await page.unroute(CHUNKS);
    await page.keyboard.press("a");
    await expect(page).toHaveURL(/\/review$/);
  });
}

test("a palette slower than the hold's cap drains the typing, then still opens", async ({
  page,
  context,
}) => {
  await seedAdminSession(context);
  const [target] = await awaitingReview(context, 1);
  await coldInvoiceWithSlowChunks(page, target!.id, 6500);
  const status = page.getByRole("status").filter({ hasText: /command palette/ });

  await page.keyboard.press("ControlOrMeta+k");
  await expect(status).toHaveText("Opening the command palette…");
  await page.keyboard.type("ac");
  await page.waitForTimeout(4300);
  await expect(status).toHaveText("The command palette is taking too long to open.");
  // Past the cap, typing that is still under way is swallowed rather than approving.
  await page.keyboard.type("me");
  await page.keyboard.press("a");
  await expectNotApproved(page, target!.id);

  const palette = page.getByPlaceholder("Search invoices, vendors, or type a command…");
  await expect(palette).toBeFocused({ timeout: 10_000 });
  await expectNotApproved(page, target!.id);
});
