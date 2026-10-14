/** Seed vendor fixtures (EN/AR names, 15-digit TRNs, AE IBANs) used by the dev-only fixture
 * adapter, standing in for the factory ERP vendors until apx/api serves real ones. */
import type { Vendor } from "@/lib/api/types";

/** Masks an illustrative fixture IBAN the same way `apx.api.service._mask_iban` does, so the
 * fixture adapter's vendor view matches the http adapter's shape (never a full IBAN, PRD §10
 * M5.4) instead of quietly being more permissive than the real backend. */
function maskIban(iban: string): string {
  return "*".repeat(Math.max(iban.length - 4, 0)) + iban.slice(-4);
}

/** The full (unmasked) illustrative IBAN behind each fixture vendor's single bank account. Kept
 * separate from `Vendor`/`BankDetail` -- which, matching the real API, never carry a full IBAN
 * -- because `fixtures/invoices.ts` needs a full value to simulate an invoice *document's own*
 * printed `supplier_iban` field (a distinct concept from the vendor master's bank account; see
 * `fixtureDocumentIban` below). */
const RAW_IBAN_BY_VENDOR_ID: Record<string, string> = {
  "v-emirates-steel": "AE070331234567890123456",
  "v-noor-alsahra": "AE260331234567890111222",
  "v-gulf-office": "AE940331234567890333444",
  "v-al-fardan": "AE120331234567890555666",
  "v-desert-rose": "AE380331234567890777888",
  "v-al-maha-tech": "AE500331234567890999000",
};

/** The full illustrative IBAN a fixture invoice's own `supplier_iban` field should show, as
 * extracted from the (simulated) document -- not the vendor master's masked bank account. */
export function fixtureDocumentIban(vendorId: string): string {
  return RAW_IBAN_BY_VENDOR_ID[vendorId] ?? "";
}

export const FIXTURE_VENDORS: readonly Vendor[] = [
  {
    id: "v-emirates-steel",
    nameEn: "Emirates Steel Trading LLC",
    nameAr: "شركة الإمارات للتجارة في الصلب ذ.م.م",
    trn: "100234567800003",
    verified: true,
    bankDetails: [
      {
        id: "bd-1",
        ibanMasked: maskIban(RAW_IBAN_BY_VENDOR_ID["v-emirates-steel"]!),
        bankName: "Emirates NBD",
        verified: true,
        verifiedAt: "2025-03-11T09:00:00.000Z",
        isCurrent: true,
        changedAt: "2025-03-11T09:00:00.000Z",
      },
    ],
    invoiceCount: 142,
    totalAed: "1284500.00",
    isNew: false,
  },
  {
    id: "v-noor-alsahra",
    nameEn: "Noor Al Sahra General Trading",
    nameAr: "نور الصحراء للتجارة العامة",
    trn: "100987654300003",
    verified: true,
    bankDetails: [
      {
        id: "bd-2",
        ibanMasked: maskIban(RAW_IBAN_BY_VENDOR_ID["v-noor-alsahra"]!),
        bankName: "First Abu Dhabi Bank",
        verified: true,
        verifiedAt: "2024-11-02T09:00:00.000Z",
        isCurrent: true,
        changedAt: "2024-11-02T09:00:00.000Z",
      },
    ],
    invoiceCount: 58,
    totalAed: "312900.50",
    isNew: false,
  },
  {
    id: "v-gulf-office",
    nameEn: "Gulf Office Supplies FZE",
    nameAr: "الخليج لتوريدات المكاتب",
    trn: "100112233400003",
    verified: true,
    bankDetails: [
      {
        id: "bd-3",
        ibanMasked: maskIban(RAW_IBAN_BY_VENDOR_ID["v-gulf-office"]!),
        bankName: "Mashreq Bank",
        verified: true,
        verifiedAt: "2023-06-20T09:00:00.000Z",
        isCurrent: true,
        changedAt: "2023-06-20T09:00:00.000Z",
      },
    ],
    invoiceCount: 231,
    totalAed: "89340.00",
    isNew: false,
  },
  {
    id: "v-al-fardan",
    nameEn: "Al Fardan Logistics",
    nameAr: "الفردان للخدمات اللوجستية",
    trn: "100556677800003",
    verified: false,
    bankDetails: [
      {
        id: "bd-4",
        ibanMasked: maskIban(RAW_IBAN_BY_VENDOR_ID["v-al-fardan"]!),
        bankName: "Abu Dhabi Commercial Bank",
        verified: false,
        verifiedAt: null,
        isCurrent: true,
        changedAt: "2026-08-30T09:00:00.000Z",
      },
    ],
    invoiceCount: 4,
    totalAed: "412000.00",
    isNew: true,
  },
  {
    id: "v-desert-rose",
    nameEn: "Desert Rose Facilities Management",
    nameAr: "وردة الصحراء لإدارة المرافق",
    trn: "100778899000003",
    verified: true,
    bankDetails: [
      {
        id: "bd-5",
        ibanMasked: maskIban(RAW_IBAN_BY_VENDOR_ID["v-desert-rose"]!),
        bankName: "Dubai Islamic Bank",
        verified: true,
        verifiedAt: "2022-01-15T09:00:00.000Z",
        isCurrent: true,
        changedAt: "2022-01-15T09:00:00.000Z",
      },
    ],
    invoiceCount: 96,
    totalAed: "745200.75",
    isNew: false,
  },
  {
    id: "v-al-maha-tech",
    nameEn: "Al Maha Technical Services",
    nameAr: "المها للخدمات الفنية",
    trn: "100443322100003",
    verified: true,
    bankDetails: [
      {
        id: "bd-6",
        ibanMasked: maskIban(RAW_IBAN_BY_VENDOR_ID["v-al-maha-tech"]!),
        bankName: "RAKBANK",
        verified: true,
        verifiedAt: "2025-09-01T09:00:00.000Z",
        isCurrent: true,
        changedAt: "2025-09-01T09:00:00.000Z",
      },
    ],
    invoiceCount: 27,
    totalAed: "156780.00",
    isNew: false,
  },
];

export function findVendor(id: string): Vendor | undefined {
  return FIXTURE_VENDORS.find((v) => v.id === id);
}

// Fails fast in dev if a fixture TRN drifts from the UAE FTA format this app renders everywhere
// (15 digits, leading "1"; this fixture set also keeps the trailing "03" convention every seed
// vendor above uses) — a design-review finding on this UI, not an apx.validate rule (that one
// only checks the 15-digit length; apx/validate/ids.py).
if (process.env.NODE_ENV !== "production") {
  const TRN_RE = /^1\d{12}03$/;
  for (const vendor of FIXTURE_VENDORS) {
    if (!vendor.trn || !TRN_RE.test(vendor.trn)) {
      throw new Error(`Fixture vendor "${vendor.id}" has an invalid TRN: ${String(vendor.trn)}`);
    }
  }
}
