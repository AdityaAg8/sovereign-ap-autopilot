"""Deterministic identifier generation for the synthetic ERP (M2.1): sequential entity ids and
document numbers, UAE TRNs and AE IBANs built from a seeded `random.Random`, so the same seed
always yields the same ids and every generated id passes `apx.validate.ids` (PRD §10 M2.1)."""

from __future__ import annotations

import random

from apx.validate.ids import make_ae_iban

# 3-digit bank codes as used inside AE IBANs, paired with a representative UAE bank name for
# readability in generated data. Curated for plausibility, not sourced from an official registry.
UAE_BANK_CODES: tuple[tuple[str, str], ...] = (
    ("010", "Emirates NBD"),
    ("014", "Abu Dhabi Commercial Bank"),
    ("020", "Dubai Islamic Bank"),
    ("030", "Mashreq Bank"),
    ("033", "Abu Dhabi Islamic Bank"),
    ("044", "HSBC Bank Middle East"),
    ("057", "RAKBANK"),
    ("060", "Commercial Bank of Dubai"),
    ("076", "Standard Chartered Bank"),
    ("081", "First Abu Dhabi Bank"),
)

TRN_PREFIX = "1"  # PINT AE rule ibr-132-ae: every TRN must start with "1".
TRN_SUFFIX = "03"  # PINT AE rule ibr-132-ae: every TRN must end with "03" (e.g. 100xxxxxxxxx003).
_TRN_RANDOM_DIGITS = 15 - len(TRN_PREFIX) - len(TRN_SUFFIX)


def sequential_id(prefix: str, index: int, width: int = 4) -> str:
    """Format a stable, human-readable id such as `VEND-0007` from a 1-based index."""
    return f"{prefix}-{index:0{width}d}"


def document_number(prefix: str, year: int, index: int, width: int = 6) -> str:
    """Format a stable document number such as `PO-2026-000042`."""
    return f"{prefix}-{year}-{index:0{width}d}"


def generate_trn(rng: random.Random, seen: set[str]) -> str:
    """Draw a 15-digit UAE TRN from `rng`, shaped `TRN_PREFIX` + random middle + `TRN_SUFFIX` so it
    always satisfies PINT AE rule ibr-132-ae (starts with "1", ends with "03", e.g.
    "100xxxxxxxxx003"), retrying on the (astronomically unlikely) collision against `seen` so every
    vendor in one ERP snapshot gets a distinct TRN."""
    while True:
        middle = "".join(str(rng.randrange(10)) for _ in range(_TRN_RANDOM_DIGITS))
        trn = f"{TRN_PREFIX}{middle}{TRN_SUFFIX}"
        if trn not in seen:
            seen.add(trn)
            return trn


def generate_ae_iban(rng: random.Random, seen: set[str]) -> str:
    """Draw a valid AE IBAN (random bank code + 16-digit account number) from `rng`, retrying
    on collision against `seen` so every bank account in one ERP snapshot is distinct."""
    while True:
        bank_code, _bank_name = rng.choice(UAE_BANK_CODES)
        account_number = "".join(str(rng.randrange(10)) for _ in range(16))
        iban = make_ae_iban(bank_code, account_number)
        if iban not in seen:
            seen.add(iban)
            return iban
