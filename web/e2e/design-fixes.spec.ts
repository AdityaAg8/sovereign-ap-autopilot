/** Review UI design-fix regressions: the inbox's Received cell stays on one line and it shows the
 * supplier's invoice number, every amount goes through one grouped formatter (also under an
 * Arabic browser locale), row focus rings are drawn inside the row, the dark-mode Approve button
 * meets WCAG AA contrast and differs from its disabled state, and audit events carry human labels
 * for apx's real action vocabulary (web/DESIGN.md "Ledger"). */
import { test, expect, type Locator, type Page } from "@playwright/test";
import { auditActionLabel } from "../src/lib/audit-actions";
import { fixtureAudit } from "../src/lib/fixtures/assistant";
import { fixtureInvoices } from "../src/lib/fixtures/invoices";
import { formatAed, formatCount, formatGroupedAmount, parseAmount } from "../src/lib/format";

const GROUPED = String.raw`\d{1,3}(?:,\d{3})*\.\d{2}`;
const AED = new RegExp(`^AED ${GROUPED}$`);

async function signIn(page: Page, who: "admin" | "reviewer" = "admin") {
  await page.goto("/login");
  await page.getByRole("button", { name: new RegExp(`${who}@sovereign\\.local`) }).click();
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/inbox$/);
  await expect(page.getByRole("option").first()).toBeVisible();
}

async function lineCount(cell: Locator): Promise<number> {
  return cell.evaluate((el) => {
    const lh = parseFloat(getComputedStyle(el).lineHeight);
    return Math.round(el.getBoundingClientRect().height / lh);
  });
}

async function openFirstWithStatus(page: Page, status: RegExp) {
  await page.getByRole("combobox", { name: "Filter by status" }).click();
  const popup = page.getByRole("listbox").filter({ has: page.getByText("All statuses") });
  await popup.getByRole("option", { name: status }).click();
  await expect(popup).toBeHidden();
  await page.getByRole("listbox", { name: "Invoices" }).getByRole("option").first().click();
  await expect(page).toHaveURL(/\/invoices\//);
}

for (const width of [1280, 1440, 1920]) {
  test(`inbox at ${width}px: Received on one line, invoice number column`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await signIn(page);
    await expect(page.getByTestId("inbox-header")).toContainText(/Invoice no\./i);
    const received = page.getByTestId("inbox-received");
    const count = await received.count();
    expect(count).toBeGreaterThan(5);
    for (let i = 0; i < count; i++) {
      const cell = received.nth(i);
      await expect(cell).toHaveText(/^\d{2} [A-Z][a-z]{2} \d{2}:\d{2}$/);
      expect(await lineCount(cell)).toBe(1);
      expect(await cell.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
    }
    const numbers = page.getByTestId("inbox-invoice-number");
    await expect(numbers.first()).toBeVisible();
    await expect(numbers.first()).toHaveText(/^[A-Z]+-[\d-]+$/);
    // Columns never overlap: each header label ends before the next one starts.
    const lefts = await page
      .getByTestId("inbox-header")
      .locator(":scope > *")
      .evaluateAll((els) =>
        els
          .filter((el) => getComputedStyle(el).display !== "none")
          .map((el) => {
            const r = el.getBoundingClientRect();
            return [r.left, r.right] as const;
          }),
      );
    for (let i = 1; i < lefts.length; i++)
      expect(lefts[i]![0]).toBeGreaterThanOrEqual(lefts[i - 1]![1]);
  });
}

test("searching by the supplier's invoice number finds the invoice", async ({ page }) => {
  await signIn(page);
  const number = (await page.getByTestId("inbox-invoice-number").first().textContent())!.trim();
  await page.getByRole("textbox", { name: "Search invoices" }).fill(number);
  const rows = page.getByRole("listbox", { name: "Invoices" }).getByRole("option");
  await expect(rows.first()).toContainText(number);
});

test("phone inbox keeps the number, received time and amount readable", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await signIn(page);
  const rows = page.getByRole("listbox", { name: "Invoices" }).getByRole("option");
  const row = rows.first();
  await expect(row.getByTestId("inbox-sub")).toContainText(/^[A-Z]+-[\d-]+/);
  await expect(row.getByTestId("inbox-amount")).toHaveText(AED);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);

  // The received time must stay whole however long the number is: every sub line, including
  // the longest fixture number, shows its time untruncated inside the row.
  const subs = page.getByTestId("inbox-sub");
  const measured = await subs.evaluateAll((els) =>
    els.map((el) => {
      const time = el.querySelector('[data-testid="inbox-sub-received"]')!;
      return {
        length: el.querySelector(".tabular-mono")!.textContent!.length,
        whole:
          time.scrollWidth <= time.clientWidth &&
          time.getBoundingClientRect().right <= el.getBoundingClientRect().right + 0.5,
      };
    }),
  );
  const longest = Math.max(...measured.map((m) => m.length));
  expect(longest).toBeGreaterThanOrEqual("INV-2026-0000".length);
  expect(measured.filter((m) => m.length === longest).every((m) => m.whole)).toBe(true);
  expect(measured.every((m) => m.whole)).toBe(true);

  // A supplier number far longer than any fixture's: only the number truncates.
  const longNumber = "INV-2026-LONG-SUPPLIER-REFERENCE-000000000042";
  await row.click();
  await expect(page).toHaveURL(/\/invoices\//);
  await page.getByRole("button", { name: "Edit Invoice number", exact: true }).click();
  await page.getByRole("textbox", { name: "Edit Invoice number" }).fill(longNumber);
  await page.getByRole("textbox", { name: "Edit Invoice number" }).press("Enter");
  await expect(page.getByText("Invoice number updated")).toBeVisible();
  await page.getByRole("link", { name: "Inbox" }).click();
  await expect(page).toHaveURL(/\/inbox$/);
  await page.getByRole("textbox", { name: "Search invoices" }).fill(longNumber);
  const sub = rows.first().getByTestId("inbox-sub");
  await expect(sub).toContainText(longNumber);
  await expect(sub.getByTestId("inbox-sub-received")).toHaveText(
    /^\d{2} [A-Z][a-z]{2} \d{2}:\d{2}$/,
  );
  const time = sub.getByTestId("inbox-sub-received");
  expect(await time.evaluate((el) => el.scrollWidth <= el.clientWidth)).toBe(true);
  const [timeBox, subBox] = [await time.boundingBox(), await sub.boundingBox()];
  expect(timeBox!.x + timeBox!.width).toBeLessThanOrEqual(subBox!.x + subBox!.width + 0.5);
  const number = sub.locator(".tabular-mono").first();
  expect(await number.evaluate((el) => el.scrollWidth > el.clientWidth)).toBe(true);
});

test("one amount formatter: grouped, two decimals, Latin digits", () => {
  expect(formatAed("1234567.8")).toBe("AED 1,234,567.80");
  expect(formatAed(0)).toBe("AED 0.00");
  expect(formatAed(null)).toBe("—");
  expect(formatAed("  ")).toBe("—");
  // Unparseable or ambiguous text is shown as-is, never turned into a different number.
  expect(formatAed("not a number")).toBe("not a number");
  expect(formatGroupedAmount("182345.67")).toBe("182,345.67");
  // apx.common.normalize.normalize_amount's comma-vs-period rule, mirrored exactly: with both
  // present, whichever comes last is the decimal point; a lone comma groups thousands.
  expect(parseAmount("1.234,56")).toBe(1234.56);
  expect(formatAed("1.234,56")).toBe("AED 1,234.56");
  expect(parseAmount("1.234.567,89")).toBe(1234567.89);
  expect(parseAmount("1,234")).toBe(1234);
  // Cases from tests/unit/test_normalize.py.
  expect(parseAmount("AED 1,234.56")).toBe(1234.56);
  expect(parseAmount("(1,234.56)")).toBe(-1234.56);
  expect(parseAmount("١٬٢٣٤٫٥٦")).toBe(1234.56);
  expect(parseAmount("١٬٢٣٤٫٥٦ د.إ")).toBe(1234.56);
  expect(parseAmount("not an amount")).toBeNull();
  expect(parseAmount("   AED   ")).toBeNull();
  // More than 2 decimals would be silently rounded on display: shown raw instead.
  expect(parseAmount("1,234.567")).toBeNull();
  expect(formatGroupedAmount("1,234.567")).toBe("1,234.567");
  expect(formatAed(0.125)).toBe("0.125");
  expect(parseAmount("−12.50")).toBe(-12.5);
  expect(parseAmount("1.2.3")).toBeNull();
  // Arabic-Indic and Eastern Arabic-Indic digits with Arabic separators normalize to one form.
  expect(parseAmount("١٬٢٣٤٬٥٦٧٫٨٩")).toBe(1234567.89);
  expect(formatGroupedAmount("۱۲۳۴٫۵")).toBe("1,234.50");
  expect(formatGroupedAmount("1,234.5")).toBe("1,234.50");
  expect(formatGroupedAmount("12a")).toBe("12a");
  expect(formatCount(12500)).toBe("12,500");
});

test.describe("under an Arabic browser locale", () => {
  test.use({ locale: "ar-AE" });

  test("every amount on inbox, invoice and operations uses the same grouped format", async ({
    page,
  }) => {
    await signIn(page);
    const amounts = page.getByTestId("inbox-amount");
    expect(await amounts.count()).toBeGreaterThan(5);
    for (const text of await amounts.allTextContents()) expect(text).toMatch(AED);

    await openFirstWithStatus(page, /^awaiting review$/i);
    await expect(page.getByTestId("invoice-total")).toHaveText(AED);
    for (const name of ["subtotal", "vat_amount", "total"]) {
      const value = page.getByTestId(`field-value-${name}`);
      await expect(value).toHaveText(new RegExp(`^${GROUPED}$`));
      await expect(value).toHaveAttribute("dir", "ltr");
    }

    await page.getByRole("link", { name: "Operations" }).click();
    const ops = page.getByTestId("ops-amount");
    await expect(ops.first()).toBeVisible();
    for (const text of await ops.allTextContents()) expect(text.trim()).toMatch(AED);
  });
});

test("keyboard focus rings sit inside inbox and field rows", async ({ page }) => {
  await signIn(page);
  await page.getByRole("textbox", { name: "Search invoices" }).focus();
  const list = page.getByRole("listbox", { name: "Invoices" });
  for (let i = 0; i < 6 && !(await list.evaluate((el) => el === document.activeElement)); i++) {
    await page.keyboard.press("Tab");
  }
  await expect(list).toBeFocused();
  const active = list.locator('[role="option"][aria-selected="true"]');
  const ring = await active.evaluate((el) => {
    const s = getComputedStyle(el);
    return { style: s.outlineStyle, width: s.outlineWidth, offset: s.outlineOffset };
  });
  expect(ring).toEqual({ style: "solid", width: "2px", offset: "-2px" });

  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/invoices\//);
  const field = page.locator("[data-field]").first();
  await field.focus();
  await page.keyboard.press("Tab");
  await page.keyboard.press("Shift+Tab");
  await expect(field).toBeFocused();
  const fieldRing = await field.evaluate((el) => {
    const s = getComputedStyle(el);
    return { style: s.outlineStyle, width: s.outlineWidth, offset: s.outlineOffset };
  });
  expect(fieldRing).toEqual({ style: "solid", width: "2px", offset: "-2px" });
});

/** WCAG contrast of an element's text against what is actually painted behind it: every
 * ancestor background (translucent tints included) is composited on a canvas, bottom-most first,
 * then the text color is drawn at the element's own opacity times its ancestors' — so a dimmed
 * `<kbd>` hint or a `bg-destructive/20` button is measured as the eye sees it. Canvas also
 * resolves oklch() computed colors to sRGB. */
async function contrast(target: Locator): Promise<number> {
  return target.evaluate((el) => {
    const ctx = document.createElement("canvas").getContext("2d")!;
    const chain: Element[] = [];
    for (let node: Element | null = el; node; node = node.parentElement) chain.push(node);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, 1, 1);
    for (const node of [...chain].reverse()) {
      ctx.fillStyle = getComputedStyle(node).backgroundColor;
      ctx.fillRect(0, 0, 1, 1);
    }
    const pixel = () => Array.from(ctx.getImageData(0, 0, 1, 1).data).slice(0, 3);
    const background = pixel();
    ctx.globalAlpha = chain.reduce((a, node) => a * Number(getComputedStyle(node).opacity), 1);
    ctx.fillStyle = getComputedStyle(el).color;
    ctx.fillRect(0, 0, 1, 1);
    const text = pixel();
    const lum = (rgb: number[]) => {
      const [r, g, b] = rgb.map((v) => {
        const c = v / 255;
        return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
      });
      return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!;
    };
    const [a, b] = [lum(text), lum(background)].sort((x, y) => y - x);
    return (a! + 0.05) / (b! + 0.05);
  });
}

for (const scheme of ["dark", "light"] as const) {
  test(`${scheme} mode: Approve text meets AA and looks unlike its disabled state`, async ({
    browser,
  }) => {
    const ctx = await browser.newContext({ colorScheme: scheme });
    const page = await ctx.newPage();
    await signIn(page, "admin");
    await openFirstWithStatus(page, /^awaiting review$/i);
    const approve = page.getByRole("button", { name: /Approve/ });
    await expect(approve).toBeEnabled();
    // The button's opacity transition fades it in from the disabled opacity while the invoice
    // loads, so both readings are polled until the transition settles.
    await expect.poll(() => approve.evaluate((el) => getComputedStyle(el).opacity)).toBe("1");
    await expect.poll(() => contrast(approve)).toBeGreaterThanOrEqual(4.5);
    // Every action label and its shortcut hint meets AA 4.5:1 (hints are text too).
    for (const name of [/^Approve/, /^Fix/, /^Reject/]) {
      const button = page.getByRole("button", { name });
      await expect.poll(() => contrast(button)).toBeGreaterThanOrEqual(4.5);
      await expect.poll(() => contrast(button.locator("kbd"))).toBeGreaterThanOrEqual(4.5);
    }

    // A reviewer cannot approve: the same button is disabled and visibly dimmed.
    const reviewer = await browser.newContext({ colorScheme: scheme });
    const page2 = await reviewer.newPage();
    await signIn(page2, "reviewer");
    await openFirstWithStatus(page2, /^awaiting review$/i);
    const disabled = page2.getByRole("button", { name: /Approve/ });
    await expect(disabled).toBeDisabled();
    await expect.poll(() => disabled.evaluate((el) => getComputedStyle(el).opacity)).toBe("0.5");
    await ctx.close();
    await reviewer.close();
  });
}

test("audit labels cover apx's action vocabulary and never crash on unknown actions", () => {
  const steps = [
    "scan",
    "parse",
    "parse_cache_miss",
    "extract",
    "trust_gate",
    "build_invoice",
    "validate",
    "apply_review",
    "fraud",
    "route",
    "payee_changed",
    "build_einvoice",
    "send_to_asp",
    "archive",
    "failed",
  ];
  const labels = new Set<string>();
  for (const step of steps) {
    const label = auditActionLabel(`activity:${step}`);
    expect(label).not.toMatch(/[:_]|^Workflow step/);
    labels.add(label);
  }
  expect(labels.size).toBe(steps.length);
  expect(auditActionLabel("authorize:approve_payment")).toBe("Permission checked: approve payment");
  expect(auditActionLabel("authorize:new_cap")).toBe("Permission checked: new cap");
  expect(auditActionLabel("decision.approve")).toBe("Approved");
  expect(auditActionLabel("decision.reject")).toBe("Rejected");
  expect(auditActionLabel("decision.send_to_review")).toBe("Sent back to review");
  expect(auditActionLabel("proposal.submit")).toBe("Proposal drafted");
  expect(auditActionLabel("proposal.accept")).toBe("Proposal accepted");
  expect(auditActionLabel("payment_state:approve")).toBe("Payment approval recorded");
  expect(auditActionLabel("payment_state:bank_change")).toBe("Bank details change recorded");
  expect(auditActionLabel("payment_state:verify_bank")).toBe("Bank details verified");
  expect(auditActionLabel("field_edit")).toBe("Field corrected");
  expect(auditActionLabel("mcp.refusal")).toBe("Assistant request refused");
  expect(auditActionLabel("mcp.read")).toBe("Assistant read records");
  expect(auditActionLabel("payment_state:execute_proposal")).toBe("Proposal execution recorded");
  expect(auditActionLabel("agent.review_note")).toBe("Assistant review note");
  expect(auditActionLabel("agent.review_note_refused")).toBe(
    "Review note withheld (not verifiable)",
  );
  expect(auditActionLabel("chat.unverifiable")).toBe(
    "Assistant answer withheld (citations not verifiable)",
  );
  expect(auditActionLabel("activity:failed")).toBe("Workflow failed");
  expect(auditActionLabel("activity:payee_changed")).toBe("Payee changed since routing");
  expect(auditActionLabel("activity:new_step")).toBe("Workflow step: new step");
  expect(auditActionLabel("frobnicate.widget_x")).toBe("Frobnicate widget x");
  expect(auditActionLabel("payment_state:")).toBe("Payment state");
  expect(auditActionLabel("")).toBe("Audit event");
});

test("every action the fixture audit trail writes has a curated label", () => {
  const actions = new Set(
    fixtureInvoices().flatMap((inv) => fixtureAudit(inv).map((e) => e.action)),
  );
  expect(actions.size).toBeGreaterThan(10);
  for (const action of actions) {
    expect(action).toMatch(/^(activity|authorize|decision|payment_state|agent)[.:][a-z_]+$/);
    const label = auditActionLabel(action);
    expect(label, action).not.toMatch(/^(Workflow step|Decision|Payment state):|[:_]/);
  }
});

test("an invoice's audit tab shows human labels, not raw action names", async ({ page }) => {
  await signIn(page);
  await openFirstWithStatus(page, /^sent$/i);
  await page.getByRole("tab", { name: "Audit" }).click();
  const labels = page.getByTestId("audit-label");
  await expect(labels.first()).toBeVisible();
  const texts = await labels.allTextContents();
  for (const expected of [
    "Scanned for hidden instructions",
    "Fields extracted",
    "Validated against PINT AE",
    "Fraud checks run",
    "Routed",
    "E-invoice built",
    "Sent to the ASP",
  ]) {
    expect(texts).toContain(expected);
  }
  for (const text of texts) expect(text).not.toMatch(/activity:|fraud\.flag|policy\.route/);
});
