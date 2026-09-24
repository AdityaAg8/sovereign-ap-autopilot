/** Dev-only Lighthouse runner (not shipped, not part of any build): signs in through the real UI
 * as the fixture admin account, then audits each page in that same tab — the session lives in
 * tab-scoped sessionStorage, so a Lighthouse CLI run in a fresh tab would only see /login.
 * Prints the median of N runs per page for performance and accessibility, mobile and desktop.
 *
 *   node scripts/prime-auth-session.mjs [baseURL] [runs] [path ...]
 *   e.g. node scripts/prime-auth-session.mjs http://127.0.0.1:3002 5 /login /inbox /invoices/inv-adr12-disagreement
 *
 * lighthouse and puppeteer-core are installed on first use into $LH_TOOLS (default: a folder in
 * the OS temp dir) rather than the app's dependencies. Chromium is Playwright's. */
import { chromium } from "@playwright/test";
import { execSync } from "node:child_process";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const baseURL = process.argv[2] ?? "http://127.0.0.1:4173";
const runs = Number(process.argv[3] ?? 5);
const paths = process.argv.slice(4).length
  ? process.argv.slice(4)
  : ["/login", "/inbox", "/invoices/inv-adr12-disagreement"];

const toolsDir = process.env.LH_TOOLS ?? join(tmpdir(), "sovereign-ap-lh-tools");
if (!existsSync(join(toolsDir, "node_modules", "lighthouse"))) {
  execSync(`npm i --prefix "${toolsDir}" --no-audit --no-fund lighthouse puppeteer-core`, {
    stdio: "inherit",
  });
}
const toolsRequire = createRequire(join(toolsDir, "package.json"));
const load = (name) => import(pathToFileURL(toolsRequire.resolve(name)).href);
const { startFlow, desktopConfig } = await load("lighthouse");
const puppeteer = (await load("puppeteer-core")).default;

const browser = await puppeteer.launch({
  executablePath: chromium.executablePath(),
  headless: true,
});

async function signedInPage() {
  const page = await browser.newPage();
  await page.goto(`${baseURL}/login`);
  await (
    await page.waitForSelector("xpath/.//button[contains(., 'admin@sovereign.local')]")
  ).click();
  await (await page.waitForSelector("xpath/.//button[normalize-space(.)='Sign in']")).click();
  await page.waitForFunction(() => location.pathname === "/inbox");
  // Let /inbox finish loading, or its leftover work lands in the audited page's TBT.
  await page.waitForNetworkIdle({ idleTime: 500 });
  return page;
}

const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

console.log("page".padEnd(40), "form".padEnd(8), "perf", "a11y", " failing a11y audits");
for (const path of paths) {
  for (const form of ["mobile", "desktop"]) {
    const perf = [];
    const a11y = [];
    const failing = new Set();
    for (let i = 0; i < runs; i++) {
      // /login is audited signed out; every other page in a freshly signed-in tab.
      const page = path === "/login" ? await browser.newPage() : await signedInPage();
      const base = form === "desktop" ? desktopConfig : { extends: "lighthouse:default" };
      const flow = await startFlow(page, {
        config: {
          ...base,
          settings: {
            ...base.settings,
            onlyCategories: ["performance", "accessibility"],
            disableStorageReset: true,
          },
        },
      });
      // disableStorageReset keeps the sessionStorage session but also skips Lighthouse's cache
      // clear; clear the HTTP cache ourselves so every run is a cold load.
      const cdp = await page.createCDPSession();
      await cdp.send("Network.clearBrowserCache");
      await cdp.detach();
      await flow.navigate(`${baseURL}${path}`);
      const lhr = (await flow.createFlowResult()).steps[0].lhr;
      if (new URL(lhr.finalDisplayedUrl).pathname !== path)
        throw new Error(`${path} redirected to ${lhr.finalDisplayedUrl}`);
      perf.push(Math.round(lhr.categories.performance.score * 100));
      a11y.push(Math.round(lhr.categories.accessibility.score * 100));
      for (const ref of lhr.categories.accessibility.auditRefs) {
        const audit = lhr.audits[ref.id];
        if (audit.score !== null && audit.score < 1) failing.add(ref.id);
      }
      await page.close();
    }
    console.log(
      path.padEnd(40),
      form.padEnd(8),
      String(median(perf)).padStart(4),
      String(median(a11y)).padStart(4),
      ` ${[...failing].join(", ") || "-"}`,
    );
  }
}
await browser.close();
