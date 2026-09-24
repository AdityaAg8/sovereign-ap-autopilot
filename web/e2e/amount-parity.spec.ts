/** Parity table between the web's `parseAmount` (lib/format.ts) and the backend's
 * apx.common.normalize.normalize_amount, mirroring tests/unit/test_normalize.py. Each row states
 * the backend's result and the web's; where they differ the difference is intentional and
 * documented: the display refuses (shows raw text) what it cannot show exactly. */
import { test, expect } from "@playwright/test";
import { formatAed, parseAmount } from "../src/lib/format";

interface ParityRow {
  raw: string;
  /** normalize_amount's Decimal as a string, or "error" for NormalizationError. */
  backend: string;
  /** parseAmount's result. */
  web: number | null;
  /** Why the two differ, when they do. */
  difference?: string;
}

const AR_1234_56 = "١٬٢٣٤٫٥٦"; // "1,234.56" in Arabic

const TABLE: readonly ParityRow[] = [
  // test_normalize_amount_western_thousands_and_decimal
  { raw: "AED 1,234.56", backend: "1234.56", web: 1234.56 },
  // test_normalize_amount_arabic_separators_and_digits
  { raw: AR_1234_56, backend: "1234.56", web: 1234.56 },
  // test_normalize_amount_round_trips_formatted_decimal: Arabic digits with the Arabic AED marker
  { raw: `${AR_1234_56} د.إ`, backend: "1234.56", web: 1234.56 },
  // ... and accounting parentheses for negatives, with and without thousands separators
  { raw: "(1,234.56)", backend: "-1234.56", web: -1234.56 },
  { raw: "AED (0.05)", backend: "-0.05", web: -0.05 },
  { raw: "9999999.99", backend: "9999999.99", web: 9999999.99 },
  { raw: "AED 9,999,999.99", backend: "9999999.99", web: 9999999.99 },
  // test_normalize_amount_bare_comma_is_thousands_separator
  { raw: "1,234", backend: "1234", web: 1234 },
  // Both separators present: whichever comes last is the decimal point.
  { raw: "1.234,56", backend: "1234.56", web: 1234.56 },
  // NBSP left by PDF/OCR extraction (normalize._ARABIC_NBSP).
  { raw: "AED 1,234.56", backend: "1234.56", web: 1234.56 },
  { raw: "−12.50", backend: "-12.50", web: -12.5 },
  // test_normalize_amount_rejects_unparseable_text / _rejects_empty_text
  { raw: "not an amount", backend: "error", web: null },
  { raw: "   AED   ", backend: "error", web: null },
  { raw: "1.2.3", backend: "error", web: null },
  // Intentional differences: the backend keeps an exact Decimal; the display shows 2 decimals
  // from a float, so rather than round or drift it refuses and the raw text is shown.
  {
    raw: "1,234.567",
    backend: "1234.567",
    web: null,
    difference: "more than 2 decimals would be rounded on display",
  },
  {
    raw: "9007199254740993",
    backend: "9007199254740993",
    web: null,
    difference: "integer part past Number.MAX_SAFE_INTEGER loses precision as a float",
  },
  {
    raw: "90071992547409.93",
    backend: "90071992547409.93",
    web: null,
    difference: "a float cannot hold these cents exactly",
  },
];

test("parseAmount agrees with normalize_amount, bar the documented differences", () => {
  for (const row of TABLE) {
    const web = parseAmount(row.raw);
    expect(web, row.raw).toBe(row.web);
    if (row.difference) {
      expect(row.backend, row.raw).not.toBe("error");
      expect(web, row.raw).toBeNull();
    } else if (row.backend === "error") {
      expect(web, row.raw).toBeNull();
    } else {
      expect(web, row.raw).toBe(Number(row.backend));
    }
  }
});

test("an amount the display cannot hold exactly is shown raw, never as a nearby number", () => {
  expect(formatAed("9007199254740993")).toBe("9007199254740993");
  expect(formatAed("9007199254740991")).toBe("AED 9,007,199,254,740,991.00");
  expect(formatAed(2 ** 53)).toBe(String(2 ** 53));
  expect(formatAed("0012.50")).toBe("AED 12.50");
});
