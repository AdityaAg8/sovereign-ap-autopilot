"""`document_text` must keep Extractor A's parse text inside its token budget whatever the text's
density. The cut is proportional in characters, but `estimate_tokens` counts a digit as a whole
token and a letter as a third, so dense ends around a sparse middle overflow."""

from __future__ import annotations

from apx.extract.prompt import PromptSettings, document_text, estimate_tokens
from tests.unit.test_extractor_a import _document


def test_trimmed_text_stays_inside_the_budget_when_the_ends_are_digit_dense() -> None:
    table = [["1234567890"] * 8 for _ in range(150)]
    prose = [["abcdefghijklmnopqrstuvwxyz"] * 8 for _ in range(750)]
    document = _document(table + prose + table)
    budget = PromptSettings().max_text_tokens
    assert estimate_tokens(document_text(document, budget)) <= budget
