# Real-data eval: cord test, Extractor B r1

n = 100, cord test, Extractor B `r1` (extractor-b-r1), in-distribution

- Scored on the 7 mapped CORD keys only (ADR-0019 §2).
- Generated at 2026-10-03T13:43:24.272415+00:00; normaliser `04d4eb8964f09d79`; git 073e6c3d8d49e808d74f61032b8f972fcc229ce0
- Mapped gold share: 76.02% of gold field instances

| Metric | Value | 95% CI | n |
|---|---|---|---|
| field_f1 | 90.52% | 87.29% to 93.17% (bootstrap) | 100 |
| nted_accuracy | 88.69% | 83.97% to 92.66% (bootstrap) | 100 |
| valid_json_rate | 100.00% | 96.30% to 100.00% (wilson) | 100 |

Valid-JSON gate (PRD ≥ 99%): PASS. Failures: 0 of 100; every failure counts in every denominator.
Latency: p50 2.6 s, p95 5.5 s per doc.

| Published baseline | Value | Comparable | Note | Source |
|---|---|---|---|---|
| Donut field F1 | 84.10% | no | fine-tuned on CORD train, full label set: not like-for-like | Kim et al., OCR-free Document Understanding Transformer (ECCV 2022), arXiv:2111.15664v5, Table 2, p. 10 |
| Donut TED accuracy | 90.90% | no | fine-tuned on CORD train, full label set: not like-for-like | Kim et al., OCR-free Document Understanding Transformer (ECCV 2022), arXiv:2111.15664v5, Table 2, p. 10 |

Identity: model=extractor-b-r1, gguf_sha256=11ce0660b74abe95f64fee509396b1ec8b78888ef7d43c2bf42c85e2f3eea246, prompt_version=6a9d2adb7b011729, serving_args_sha256=0b433ec65c6548bf, parser_identity=a8832889e65c7fd5
