/** Acceptance: the operations view shows every KPI, the pipeline rail with live counts,
 * the GPU model strip with states spelled out in words, and refreshes on its own. The fixture
 * adapter advances synthetic in-flight invoices every tick, so "updated" must change. */
import { test, expect } from "@playwright/test";

test("operations view shows KPIs, pipeline, models and refreshes live", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /reviewer@sovereign\.local/ }).click();
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/inbox$/);

  await page.getByRole("link", { name: "Operations" }).click();
  await expect(page).toHaveURL(/\/operations$/);

  for (const id of ["stp", "auto-accept-error", "latency", "attacks", "queue"]) {
    await expect(page.getByTestId(`kpi-${id}`)).toBeVisible();
  }
  await expect(page.getByRole("heading", { name: "Pipeline" })).toBeVisible();
  const stages = page
    .getByRole("heading", { name: "Pipeline" })
    .locator("xpath=../..")
    .getByRole("list")
    .first();
  await expect(stages.getByRole("listitem")).toHaveCount(7);
  await expect(page.getByTestId("stage-received")).toContainText(/\d+/);

  const models = page.getByRole("heading", { name: "Models on the GPU" }).locator("xpath=../..");
  await expect(models.getByTestId("model-extractor-a")).toContainText(/Loaded|Loading|Asleep/);

  await expect(page.getByRole("heading", { name: "Latest activity" })).toBeVisible();

  const stamp = page.locator("time").first();
  const first = await stamp.getAttribute("datetime");
  await expect.poll(async () => stamp.getAttribute("datetime"), { timeout: 8000 }).not.toBe(first);
});

test("g then o jumps to operations from the keyboard", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /reviewer@sovereign\.local/ }).click();
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/inbox$/);
  await page.locator("body").click();
  await page.keyboard.press("g");
  await page.keyboard.press("o");
  await expect(page).toHaveURL(/\/operations$/);
});
