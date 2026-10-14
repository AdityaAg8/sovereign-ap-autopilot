# ADR-0020: v1.0 scope: what ships, what is deferred, and why

- Status: accepted
- Date: 2026-10-04
- Related: ADR-0009 §1.8 and §2-§3 (B failure path, teacher, arbiter), ADR-0013 (VRAM budget),
  ADR-0016 (bake-off), ADR-0018 (reduced GPU plan), ADR-0019 (real-data track, amendment 3)

## Context

Four planned items need either large downloads or several GPU days, and the v1.0 budget was
about a day of GPU time, after another project releases the card. The
measured state on 2026-10-04:
- Extractor B R1 passes P1 (accuracy), P3, P4 and P5, and fails only P2 (latency: B p50 19.03 s
  against a 16.94 s limit). Its accuracy gap on long invoices has a found cause (ADR-0019
  amendment 3) that a same-config retrain (R1b) addresses.
- With no arbiter configured, a field A and B disagree on keeps A's value
  (`apx/extract/pipeline.py`). The offline champion-only gate (0.54% error among auto-accepted
  fields, limit 1%) is fitted on A-only features and is not active at runtime, so every field
  goes to a human ([KI-08](../../KNOWN_ISSUES.md)).

## Decision

1. **Teacher round: skipped for v1.0.** ADR-0009 §1.8 calls the teacher only when round-1 B
   misses its gate on accuracy. B misses only P2, and teacher labels improve accuracy, not
   decode time, so a teacher round cannot close the gap that remains. It would also need a
   27-35B model download (over 1 GB) and a multi-day pilot.
2. **Arbiter: deferred to after v1.0.** The pipeline already has the seam
   (`Arbiter`, `has_arbiter` in the extract audit). Without it, disputes keep A's value. The candidate models
   (Gemma 4 12B QAT, Qwen3.8-27B) need downloads and a disagreement-accuracy eval on cal.
3. **Extractor A bake-off and MTP latency arm: deferred.** The bake-off (ADR-0016) was not
   run, so Qwen3.5-9B stays Extractor A by default. Speculative
   decoding for A or B is re-opened only by a measured probe on the served build, and any
   serving change goes through an ADR-0013 amendment.
4. **Live retrain loop: shipped as code, exercised offline.** The retrain and promotion
   pipeline (ADR-0015, `ml/promotion`, `ml/extractor_b`) has been run end to end twice on real
   data (R1, R1b) with the same gates. A retrain triggered by live review corrections is not
   run for v1.0, because the deployed stack has no corrections volume yet.
5. **v1.0.0 needs**, in addition to the release checklist: a clean security audit of the code
   changed after the third audit pass, the R1b gate result recorded honestly (promoted only if every gate passes), and a
   live `make e2e-pipeline` run on the rebuilt stack.

## Consequences

- KNOWN_ISSUES.md lists the arbiter, the teacher and the live retrain loop as deferred
  features, with this ADR as the reason.
- Nothing about the gates, thresholds or eval sets changes here.
