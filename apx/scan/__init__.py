"""Detects hidden and invisible text, zero-width and bidi control characters, and
suspicious PDF metadata, attachments or JavaScript before a document is trusted (PRD §3, §6)."""

from __future__ import annotations

from apx.scan.detector import PdfHiddenContentDetector

__all__ = ["PdfHiddenContentDetector"]
