# Scan-variant restore: data/uae (ADR-0012 amendment 3)

Run 2026-09-25 on the host CPU: `python -m factory.restore_scans --data-dir data/uae --seed 2026 --workers 10`.
It took 278 s for 1,500 documents (3,096 pages). A second run skipped all 1,500 in 0.6 s, which shows it is idempotent.

## Method

- Plan: the `factory.build` defaults are seed 2026, 5,000 documents split 3,500/750/750, 25% Arabic-first, 30% scans, 20 vendors, 8 users and scale 2.0.
  All 5,000 documents on disk matched the plan by `doc_seed`, with no split, layout, tax case, Arabic-first or hook mismatch.
- Clean pages: the tool re-rendered the same HTML with `capture_pages=True`.
  Every re-rendered PDF was byte-identical to the one on disk (sha256) and every labels JSON was identical, so no pypdfium2 rasterization fallback was needed.
- Scans: `factory.scan.scan_page(png, seed=_derive_seed(doc_seed, "scan", page))`, the factory's own seeds.
  PNGs are written atomically before `has_scan` is set to true.

## Counts

| split | docs | scan docs | rate | scan pages | uae_stale_m2 scan docs |
|---|---:|---:|---:|---:|---:|
| train | 3,500 | 1,042 | 29.8% | 2,140 | 1,041 / 3,497 |
| cal | 750 | 218 | 29.1% | 449 | 218 / 749 |
| test | 750 | 240 | 32.0% | 507 | 239 / 749 |
| all | 5,000 | 1,500 | 30.0% | 3,096 | 1,498 / 4,995 |

All 1,498 stale scan documents appear in the new set, matched by `doc_seed`, so both builds come from the same plan.

## Checks

- PDFs: all 5,000 still hash to their file names. No `labels.json` file was modified.
  `data/manifest.json` and `evals/splits.json` were not touched.
- Every scan PNG decodes and is 793 x 1122 px. That is the clean screenshot raster: one A4 page in CSS pixels, truncated.
  `factory.scan.map_bbox_to_pixels` maps the labels' page fractions onto this raster.
- Samples are in `reports/factory_samples/scan_*.png`. Each shows the test-split page 0 scan with its label boxes drawn in red, and the boxes sit on the text.

## Consumers

The following consumers never read a scan PNG. All of them read `<sha>.pdf`, so the restored scans reach none of them yet:
- `evals/extraction_a.py` (Extractor A's before/after run)
- `ml/extractor_b/data.py` (B's training)
- `scripts/e2e_pipeline.py`: its "scanned" pick filters on `has_scan` and now selects real scan documents, but it still sends the PDF.
- the parse gate

These use `has_scan` only as a filter:
- `scripts/demo.py` and `scripts/measure_serving_budget.py` exclude `has_scan` documents.
  Their pools shrink by 30%, but enough documents are left.

The following caches file lists:
- `data/ml/extractor_b/manifest.jsonl` and `pages/` (B's fingerprinted training manifest) list PDF pages only. We did not regenerate them.
- `data/manifest.json` lists PDFs only.
- Neither cache changes because of this restore.
