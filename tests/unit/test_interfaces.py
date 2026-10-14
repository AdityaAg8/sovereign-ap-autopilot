"""Imports every apx interface module and checks the type-level trust boundary: extraction
requests cannot carry tools and raw model fields cannot carry boxes."""

import importlib
import pkgutil

import pytest
from pydantic import ValidationError

import apx
from apx.common.untrusted import UNTRUSTED_MAX_CHARS, UntrustedText
from apx.extract.interfaces import RawField
from apx.llm.interfaces import StructuredRequest


def _all_modules() -> list[str]:
    return sorted(m.name for m in pkgutil.walk_packages(apx.__path__, prefix="apx."))


def test_every_apx_module_imports_with_header() -> None:
    names = _all_modules()
    assert len(names) >= 32
    for name in names:
        module = importlib.import_module(name)
        assert module.__doc__, f"{name} is missing its PRD §8 header docstring"


def test_structured_request_has_no_tools_field() -> None:
    assert "tools" not in StructuredRequest.model_fields
    with pytest.raises(ValidationError):
        StructuredRequest.model_validate(
            {"model": "m", "messages": [], "json_schema": {}, "tools": []}
        )


def test_structured_request_forbids_thinking_and_high_temperature() -> None:
    base = {"model": "m", "messages": [], "json_schema": {}}
    with pytest.raises(ValidationError):
        StructuredRequest.model_validate({**base, "enable_thinking": True})
    with pytest.raises(ValidationError):
        StructuredRequest.model_validate({**base, "temperature": 0.5})


def test_raw_field_cannot_carry_boxes() -> None:
    with pytest.raises(ValidationError):
        RawField.model_validate({"name": "total", "value": "1", "bbox": [0, 0, 1, 1]})


def test_untrusted_text_is_length_capped() -> None:
    with pytest.raises(ValidationError):
        UntrustedText(text="x" * (UNTRUSTED_MAX_CHARS + 1), source="doc:test")
