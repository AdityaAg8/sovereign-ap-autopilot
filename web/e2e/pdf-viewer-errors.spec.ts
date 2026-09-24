/** M5.4a: the invoice viewer ends in its explicit error state, not an endless "Loading document…",
 * when pdf.js's parser worker script cannot be loaded (pdf-render-worker.ts listens for the nested
 * worker's `error`). Runs against the production build via playwright.config.ts's webServer. */
import { test, expect } from "@playwright/test";

test("a parser worker that fails to load shows the document error state", async ({ page }) => {
  await page.goto("/login");
  await page.getByRole("button", { name: /admin@sovereign\.local/ }).click();
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/inbox$/);

  await page
    .context()
    .route("**/pdfjs/pdf.worker.entry.mjs", (route) =>
      route.fulfill({ status: 404, contentType: "text/plain", body: "Not found" }),
    );
  await page.goto("/invoices/inv-adr12-disagreement");

  await expect(page.getByText("Couldn't render this document.", { exact: false })).toBeVisible({
    timeout: 10_000,
  });
  await expect(page.getByText("Loading document…")).toBeHidden();
});
