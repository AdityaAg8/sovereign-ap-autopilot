"""UAE TRN and IBAN identity validators. Reused by the synthetic ERP factory (M2.1) to build
IDs that already pass validation and by `apx.validate.interfaces` (M4.1 extends this module
with the full mandatory-field checklist, PRD §6.3, §10 M4.1)."""

from __future__ import annotations

import re

_TRN_RE = re.compile(r"^1\d{12}03$")
_IBAN_BBAN_RE = re.compile(r"^[A-Z0-9]+$")

AE_COUNTRY_CODE = "AE"
AE_IBAN_LENGTH = 23
AE_BBAN_LENGTH = AE_IBAN_LENGTH - 4  # country code (2) + check digits (2)


def is_valid_trn(trn: str) -> bool:
    """UAE TRN: exactly 15 digits, starting with "1" and ending with "03" (PINT AE rule
    ibr-132-ae, e.g. "100xxxxxxxxx003"). There is no published check-digit scheme, so beyond that
    shape this is a format check only (PRD §6.3)."""
    return bool(_TRN_RE.fullmatch(trn))


def _iban_numeral_string(iban: str) -> str:
    """Rearrange an IBAN candidate per ISO 7064 MOD 97-10: move the first four characters
    (country code + check digits) to the end, then map letters A-Z to the digits 10-35."""
    rearranged = iban[4:] + iban[:4]
    digits: list[str] = []
    for ch in rearranged:
        if ch.isdigit():
            digits.append(ch)
        elif "A" <= ch <= "Z":
            digits.append(str(ord(ch) - ord("A") + 10))
        else:
            raise ValueError(f"invalid IBAN character: {ch!r}")
    return "".join(digits)


def iban_mod97(iban: str) -> int:
    """ISO 7064 MOD 97-10 remainder of `iban`. A structurally valid IBAN has remainder 1."""
    return int(_iban_numeral_string(iban)) % 97


def is_valid_iban(iban: str) -> bool:
    """General ISO 13616 IBAN check: 2 letters + 2 check digits + BBAN (up to 30 alphanumeric
    characters), MOD 97-10 remainder of 1. Independent of country-specific length."""
    if not (4 <= len(iban) <= 34):
        return False
    if not (iban[:2].isalpha() and iban[:2].isupper()):
        return False
    if not iban[2:4].isdigit():
        return False
    if not _IBAN_BBAN_RE.fullmatch(iban[4:]):
        return False
    try:
        return iban_mod97(iban) == 1
    except ValueError:
        return False


def is_valid_ae_iban(iban: str) -> bool:
    """AE IBANs are exactly 23 characters: AE + 2 check digits + 3-digit bank code +
    16-digit account number (PRD §3; §10 M4.1 extends the mandatory-field list)."""
    return iban.startswith(AE_COUNTRY_CODE) and len(iban) == AE_IBAN_LENGTH and is_valid_iban(iban)


def compute_iban_check_digits(country_code: str, bban: str) -> str:
    """Compute the 2-digit ISO 13616 check digits for `country_code` + `bban`, so a generator
    can assemble an IBAN that satisfies `is_valid_iban` (used by `factory.ids`)."""
    if len(country_code) != 2 or not country_code.isalpha():
        raise ValueError("country_code must be a 2-letter ISO code")
    provisional = f"{country_code}00{bban}"
    remainder = iban_mod97(provisional)
    return f"{98 - remainder:02d}"


def make_ae_iban(bank_code: str, account_number: str) -> str:
    """Assemble a valid 23-character AE IBAN from a 3-digit UAE bank code and a 16-digit
    account number, computing the ISO 13616 check digits."""
    if not (len(bank_code) == 3 and bank_code.isdigit()):
        raise ValueError("bank_code must be exactly 3 digits")
    if not (len(account_number) == 16 and account_number.isdigit()):
        raise ValueError("account_number must be exactly 16 digits")
    bban = f"{bank_code}{account_number}"  # len == AE_BBAN_LENGTH by the checks above
    check_digits = compute_iban_check_digits(AE_COUNTRY_CODE, bban)
    return f"{AE_COUNTRY_CODE}{check_digits}{bban}"
