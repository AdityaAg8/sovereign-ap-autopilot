/** Before React hydrates, the login form must not submit natively: a GET submission would put the
 * email and password in the URL (history, logs, referrers). JavaScript disabled is exactly the
 * pre-hydration page, so these tests pin the server-rendered form's behaviour. */
import { expect, test } from "@playwright/test";

test.use({ javaScriptEnabled: false });

test("the server-rendered sign-in button is disabled and the form posts", async ({ page }) => {
  await page.goto("/login");
  await expect(page.getByRole("button", { name: "Sign in" })).toBeDisabled();
  await expect(page.locator("form")).toHaveAttribute("method", "post");
});

test("pressing Enter before hydration never puts credentials in the URL", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Email").fill("someone@example.com");
  await page.getByLabel("Password").fill("not-in-the-url");
  const submits: string[] = [];
  page.on("request", (request) => {
    if (request.isNavigationRequest() && request.url() !== page.url()) submits.push(request.url());
    if (request.method() === "POST") submits.push(`POST ${request.url()}`);
  });
  await page.getByLabel("Password").press("Enter");
  await page.waitForTimeout(500);
  expect(submits).toEqual([]);
  expect(page.url()).not.toContain("not-in-the-url");
});
