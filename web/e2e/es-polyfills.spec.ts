/** public/pdfjs/es-polyfills.mjs (the ES2026 built-ins pdf.js 6 needs) follows the
 * proposals where it matters to callers: argument validation, exact summation and upsert
 * semantics. The polyfill is evaluated fresh in a page whose native versions (if any) were
 * removed, so the polyfill itself is under test, not the browser's built-in. */
import { test, expect } from "@playwright/test";

test("es-polyfills.mjs installs spec-conformant Math.sumPrecise and Map upserts", async ({
  page,
}) => {
  await page.goto("/login");
  const result = await page.evaluate(async () => {
    const math = Math as unknown as Record<string, unknown>;
    delete math.sumPrecise;
    for (const proto of [Map.prototype, WeakMap.prototype] as unknown as Record<
      string,
      unknown
    >[]) {
      delete proto.getOrInsert;
      delete proto.getOrInsertComputed;
    }
    await import(/* webpackIgnore: true */ `/pdfjs/es-polyfills.mjs?fresh=${Date.now()}`);
    const sum = (Math as unknown as { sumPrecise: (items: unknown) => number }).sumPrecise;
    const throwsType = (arg: unknown) => {
      try {
        sum(arg);
        return false;
      } catch (error) {
        return error instanceof TypeError;
      }
    };
    const map = new Map<string, number>([["a", 1]]) as Map<string, number> & {
      getOrInsert: (k: string, v: number) => number;
      getOrInsertComputed: (k: string, f: (k: string) => number) => number;
    };
    return {
      nonObjects: ["", "12", 0, BigInt(1), true, Symbol.iterator, undefined, null].map(throwsType),
      nonNumbers: throwsType([1, "2"]),
      exact: sum([0.1, 0.2, 0.3]),
      cancel: sum([1e308, 1e308, -1e308]),
      set: sum(new Set([1, 2, 3])),
      emptyIsMinusZero: Object.is(sum([]), -0),
      nan: Number.isNaN(sum([Infinity, -Infinity])),
      upsert: [
        map.getOrInsert("a", 9),
        map.getOrInsert("b", 2),
        map.getOrInsertComputed("c", () => 3),
      ],
    };
  });
  expect(result.nonObjects).toEqual(Array(8).fill(true));
  expect(result.nonNumbers).toBe(true);
  expect(result.exact).toBe(0.6);
  expect(result.cancel).toBe(1e308);
  expect(result.set).toBe(6);
  expect(result.emptyIsMinusZero).toBe(true);
  expect(result.nan).toBe(true);
  expect(result.upsert).toEqual([1, 2, 3]);
});
