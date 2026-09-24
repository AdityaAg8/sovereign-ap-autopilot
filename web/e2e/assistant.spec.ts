/** M9.3 acceptance: asking the assistant "why was X blocked" returns an answer that cites audit
 * ids, and a citation opens that invoice's Audit tab on the cited event. Also checks the
 * assistant declines a payment request instead of acting (PRD §10 M9.3, ADR-0010). */
import { test, expect, type Page } from "@playwright/test";

async function signIn(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: /reviewer@sovereign\.local/ }).click();
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/inbox$/);
}

test("why was X blocked is answered with audit citations that open the audit tab", async ({
  page,
}) => {
  await signIn(page);
  await page.locator("body").click();
  await page.keyboard.press("c");
  const panel = page.getByRole("dialog", { name: "Assistant" });
  await expect(panel).toBeVisible();

  const suggestion = panel.getByRole("button", { name: /^Why was .+ blocked\?$/ });
  await expect(suggestion).toBeVisible();
  await suggestion.click();

  const answer = panel.locator('[data-role="assistant"]').last();
  await expect(answer).toContainText("was blocked");
  const citations = answer.getByRole("link", { name: /^Open audit \d+$/ });
  expect(await citations.count()).toBeGreaterThan(0);
  const cited = (await citations.first().getAttribute("aria-label"))!.match(/\d+/)![0];

  await citations.first().click();
  // Following a citation closes the panel so the cited event isn't hidden behind it.
  await expect(panel).toBeHidden();
  await expect(page).toHaveURL(new RegExp(String.raw`/invoices/[^?]+\?tab=audit&event=${cited}$`));
  await expect(page.getByRole("tab", { name: "Audit" })).toHaveAttribute("aria-selected", "true");
  const event = page.getByTestId(`audit-event-${cited}`);
  await expect(event).toBeVisible();
  await expect(event).toHaveAttribute("aria-current", "true");
  await expect(event).toContainText("Routed");
});

test("the assistant declines to pay and says who can", async ({ page }) => {
  await signIn(page);
  await page.getByRole("button", { name: "Ask the assistant" }).click();
  const panel = page.getByRole("dialog", { name: "Assistant" });
  await panel.getByLabel("Ask the assistant").fill("Pay GT-3840 now");
  await panel.getByLabel("Ask the assistant").press("Enter");
  const answer = panel.locator('[data-role="assistant"]').last();
  await expect(answer).toContainText("Declined");
  await expect(answer).toContainText("can't do that");
});
