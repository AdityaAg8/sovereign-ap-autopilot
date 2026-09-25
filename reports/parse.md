# Parse service report (PRD §10 M3.1)

Generated: 2026-09-25T12:07:22.072324+00:00

Born-digital PDFs are parsed host-side by `apx.parse.digital` (pdfplumber): word/line boxes come straight from the PDF's own text layer, no model call (I1). This is the path the IoU gate below runs against. Scans go through the PaddleOCR-VL service (`apx.parse.client`) and are reported separately as informational only, per the PRD.

## Digital-path IoU gate

- Documents scored: 750
- Labeled fields scored: 96444
- Gate: IoU >= 0.5 on >= 98% of labeled fields

| Split | Fields | Pass rate (IoU>=0.5) |
|---|---|---|
| Overall | 96444 | 0.9824 |
| EN | 85147 | 0.9802 |
| AR | 11297 | 0.9987 |

## Digital-path latency (pdfplumber, host-side, no GPU)

- p50: 187.36 ms
- p95: 380.53 ms

## Scan path (PaddleOCR-VL service) -- informational, not gated

Not measured this run: the `parse` compose service was not reachable, or no `*.page0.scan.png` samples were found under data/uae/test.
