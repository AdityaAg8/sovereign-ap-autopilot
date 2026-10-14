"""Pure, typed scorers for the evaluation harness: DocILE KILE/LIR, FATURA
and UAE field F1, ExtractBench value/grounding F1, and a latency/VRAM collector."""

from evals.scorers.docile import DocileTaskScore, score_kile, score_lir
from evals.scorers.extractbench import ExtractBenchExample, ExtractBenchScore, score_extractbench
from evals.scorers.fatura import score_fatura
from evals.scorers.field_f1 import FieldF1Result, FieldRecord, field_f1, normalize_value
from evals.scorers.perf import LatencyStats, PerfCollector
from evals.scorers.uae import UaeDocument, UaeFieldF1Result, score_uae

__all__ = [
    "DocileTaskScore",
    "ExtractBenchExample",
    "ExtractBenchScore",
    "FieldF1Result",
    "FieldRecord",
    "LatencyStats",
    "PerfCollector",
    "UaeDocument",
    "UaeFieldF1Result",
    "field_f1",
    "normalize_value",
    "score_extractbench",
    "score_fatura",
    "score_kile",
    "score_lir",
    "score_uae",
]
