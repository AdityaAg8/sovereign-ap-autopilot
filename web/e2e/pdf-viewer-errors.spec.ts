/** M5.4a: the invoice viewer ends in its explicit error state, not an endless "Loading document…",
 * when pdf.js's parser worker script cannot be loaded (pdf-render-worker.ts listens for the nested
 * worker's `error`). Runs against the production build via playwright.config.ts's webServer. */
import { test, expect } from "@playwright/test";

test("a parser worker that fails to load shows the document error state", async ({ page }) => {
  await page.goto("/login");
  await page.getByRole("button", { name: /admin@sovereign\.local/ }).click();
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/inbox$/);

  // Chromium does not route a nested worker's own top-level script fetch through Playwright (the
  // render worker's `new Worker(pdf.worker.entry.mjs)` never reaches `route`), but it does route
  // that worker's module imports. Failing the vendored pdf.js module the entry imports makes the
  // parser worker's module graph fail to load, which is the same `error` event a 404 entry fires.
  await page
    .context()
    .route("**/pdfjs/pdf.worker.*.mjs", (route) =>
      route.fulfill({ status: 404, contentType: "text/plain", body: "Not found" }),
    );
  await page.goto("/invoices/inv-adr12-disagreement");

  await expect(page.getByText("Couldn't render this document.", { exact: false })).toBeVisible({
    timeout: 10_000,
  });
  await expect(page.getByText("Loading document…")).toBeHidden();
});
