/** Keyboard-layout regression spec for the review UI's hotkeys (PRD §10 M5.4 keyboard-first flows,
 * EN/AR product): on an Arabic layout letter shortcuts match by physical key, AltGr-typed symbols
 * match plain symbol combos, and Ctrl never turns a letter shortcut into a plain one. */
import { test, expect, type Page } from "@playwright/test";

interface SyntheticKey {
  key: string;
  code: string;
  ctrlKey?: boolean;
  altKey?: boolean;
  shiftKey?: boolean;
  isComposing?: boolean;
}

const ARABIC_J = { key: "ت", code: "KeyJ" } as const;
const ARABIC_K = { key: "ن", code: "KeyK" } as const;

function rows(page: Page) {
  return page.getByRole("listbox", { name: "Invoices" }).getByRole("option");
}

async function signIn(page: Page) {
  await page.goto("/login");
  await page.getByRole("button", { name: /admin@sovereign\.local/ }).click();
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/inbox$/);
  await expect(rows(page).first()).toBeVisible();
  // Rows take the highlight on hover, and the pointer is left where Sign in was clicked; park it
  // off the list so only the keys under test move the highlight.
  await page.mouse.move(0, 0);
}

/** Index of the single highlighted row among the rendered rows (the list starts scrolled to top). */
async function highlighted(page: Page): Promise<number> {
  const selected = await rows(page).evaluateAll((els) =>
    els.map((el) => el.getAttribute("aria-selected") === "true"),
  );
  expect(selected.filter(Boolean)).toHaveLength(1);
  return selected.indexOf(true);
}

async function expectHighlighted(page: Page, index: number) {
  await expect(rows(page).nth(index)).toHaveAttribute("aria-selected", "true");
}

/** Dispatches a keydown the way a browser with the given layout reports it, on the focused element
 * so it bubbles to the hook's window listener. Returns whether a hotkey consumed it. */
async function pressSynthetic(page: Page, init: SyntheticKey): Promise<boolean> {
  return page.evaluate((keyInit) => {
    const target = document.activeElement ?? document.body;
    const event = new KeyboardEvent("keydown", { ...keyInit, bubbles: true, cancelable: true });
    target.dispatchEvent(event);
    return event.defaultPrevented;
  }, init);
}

test("Arabic-layout j/k move the inbox highlight and Enter opens it", async ({ page }) => {
  await signIn(page);
  const start = await highlighted(page);

  expect(await pressSynthetic(page, ARABIC_J)).toBe(true);
  await expectHighlighted(page, start + 1);
  expect(await pressSynthetic(page, ARABIC_J)).toBe(true);
  await expectHighlighted(page, start + 2);
  expect(await pressSynthetic(page, ARABIC_K)).toBe(true);
  await expectHighlighted(page, start + 1);

  const id = await rows(page)
    .nth(start + 1)
    .getAttribute("id");
  expect(id).toBeTruthy();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(new RegExp(`/invoices/${id!.replace(/^inbox-row-/, "")}$`));
});

test("Ctrl with an Arabic-layout j does not move the highlight", async ({ page }) => {
  await signIn(page);
  const start = await highlighted(page);

  expect(await pressSynthetic(page, { ...ARABIC_J, ctrlKey: true })).toBe(false);
  // A plain j afterwards lands one row down, so the Ctrl press cannot have moved it as well.
  expect(await pressSynthetic(page, ARABIC_J)).toBe(true);
  await expectHighlighted(page, start + 1);
  expect(await highlighted(page)).toBe(start + 1);
});

test("Ctrl with an Arabic-layout k opens the command palette via its physical key", async ({
  page,
}) => {
  await signIn(page);
  expect(await pressSynthetic(page, { ...ARABIC_K, ctrlKey: true })).toBe(true);
  await expect(page.getByRole("dialog", { name: "Command palette" })).toBeVisible();
});

test("AltGr-typed ? (reported as Ctrl+Alt) opens the shortcuts dialog", async ({ page }) => {
  await signIn(page);
  const altGrQuestion = { key: "?", code: "Minus", ctrlKey: true, altKey: true };
  expect(await pressSynthetic(page, altGrQuestion)).toBe(true);
  await expect(page.getByRole("dialog", { name: "Keyboard shortcuts" })).toBeVisible();
});

test("Latin-layout j/k and Ctrl+K still work", async ({ page }) => {
  await signIn(page);
  const start = await highlighted(page);

  expect(await pressSynthetic(page, { key: "j", code: "KeyJ" })).toBe(true);
  await expectHighlighted(page, start + 1);
  expect(await pressSynthetic(page, { key: "k", code: "KeyK" })).toBe(true);
  await expectHighlighted(page, start);
  // Dvorak: the character decides, so a "j" typed on the physical C key still moves down.
  expect(await pressSynthetic(page, { key: "j", code: "KeyC" })).toBe(true);
  await expectHighlighted(page, start + 1);
  expect(await pressSynthetic(page, { key: "k", code: "KeyK", ctrlKey: true })).toBe(true);
  await expect(page.getByRole("dialog", { name: "Command palette" })).toBeVisible();
});

test("AZERTY '&' on the Digit1 key is not remapped to a digit shortcut", async ({ page }) => {
  await signIn(page);
  const start = await highlighted(page);
  // No inbox binding takes "&", and the physical-key fallback is only for non-Latin scripts, so
  // nothing consumes it; a following j then lands exactly one row down.
  expect(await pressSynthetic(page, { key: "&", code: "Digit1" })).toBe(false);
  expect(await pressSynthetic(page, { key: "j", code: "KeyJ" })).toBe(true);
  await expectHighlighted(page, start + 1);
  expect(await highlighted(page)).toBe(start + 1);
});

test("A Latin remap fires the typed letter's shortcut, not the physical key's", async ({
  page,
}) => {
  await signIn(page);
  const start = await highlighted(page);
  // Same shape as AZERTY "a" on the Q key, but j and k are both real inbox bindings, so this
  // tells "matched by character" (down) apart from "matched by physical key" (up).
  expect(await pressSynthetic(page, { key: "j", code: "KeyK" })).toBe(true);
  await expectHighlighted(page, start + 1);
  expect(await pressSynthetic(page, { key: "k", code: "KeyJ" })).toBe(true);
  await expectHighlighted(page, start);
});

test("A j pressed mid IME composition does not move the highlight", async ({ page }) => {
  await signIn(page);
  const start = await highlighted(page);
  const composing = { key: "j", code: "KeyJ", isComposing: true };
  expect(await pressSynthetic(page, composing)).toBe(false);
  expect(await pressSynthetic(page, { key: "j", code: "KeyJ" })).toBe(true);
  await expectHighlighted(page, start + 1);
  expect(await highlighted(page)).toBe(start + 1);
});
