"""IBAN checks at vendor-master write time: the strict ISO 13616 electronic form from
`apx.policy.interfaces.electronic_iban` (ADR-0014 §2) plus the ISO 7064 mod-97-10 checksum that
ADR-0014 deferred to `change_bank_details`, so a mistyped IBAN never enters the bank master."""

from __future__ import annotations

from apx.bank.errors import InvalidIbanError
from apx.policy.interfaces import electronic_iban


def mod97_valid(iban: str) -> bool:
    """True if the electronic-format `iban` passes ISO 7064 mod-97-10 (remainder 1 after moving
    the first four characters to the end and mapping A-Z to 10-35)."""
    rearranged = iban[4:] + iban[:4]
    remainder = 0
    for char in rearranged:
        digits = str(int(char, 36))  # "0"-"9" stay digits, "A"-"Z" become 10-35
        for digit in digits:
            remainder = (remainder * 10 + int(digit)) % 97
    return remainder == 1


def validated_iban(raw: str) -> str:
    """The electronic form of `raw` if it is well formed and passes mod-97; otherwise raises
    `InvalidIbanError`. The message never echoes `raw` (it may be attacker-supplied text)."""
    normalized = electronic_iban(raw)
    if normalized is None:
        raise InvalidIbanError(
            "IBAN is not in ISO 13616 form (letters, digits and single spaces only)"
        )
    if not mod97_valid(normalized):
        raise InvalidIbanError("IBAN fails the ISO 7064 mod-97 checksum")
    return normalized
