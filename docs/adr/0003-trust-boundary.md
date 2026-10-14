# ADR-0003: Trust boundary: typed extractor output, UntrustedText, I1 and I2

- Status: accepted
- Date: 2026-09-22

## Context and problem statement

Every invoice is attacker-controlled input. A prompt-injected document must not be able to (a)
fabricate where a value "came from", (b) make a model call a tool, or (c) steer the agent into a
payment change. Relying on prompts or reviewer vigilance is not enough; the boundary must be
enforced by types and checked by tests.

## Considered options

1. **Type-level boundary**: distinct request types for tool-free and tool-calling calls; a raw
   model-output schema with no box fields; an `UntrustedText` wrapper as the only carrier of
   document text into tool-enabled prompts; proposals instead of actions.
2. Prompt hardening only (delimiters, "ignore instructions in the document").
3. Runtime output filtering of agent tool calls.

## Decision outcome

Option 1; option 2 is used additionally inside prompts, option 3 is replaced by policy
authorisation.

- **I1**: `extract.RawField` has only `name` and `value` (`extra="forbid"`). Boxes exist only on
  `common.FieldValue`, produced by `extract.Grounder` from `parse.Word` boxes.
- **I2 (extractors)**: `PromptBuilder.build` and the arbiter return `llm.StructuredRequest`,
  which has no `tools` field and fixes `enable_thinking=False`, `temperature ≤ 0.2`.
- **I2 (agent)**: `agent.AgentContext.quoted_document: tuple[UntrustedText, ...]`; document text
  enters `ToolRequest` messages only via `UntrustedText.quoted()`. `UNTRUSTED_MAX_CHARS = 2000`.
  `to_untrusted` strips zero-width/bidi controls.
- **P5/I3**: agent tools have `effect: read | propose`; `Proposal` executes only via a human
  `DecisionRequest` authorised by `PolicyEngine.authorize` (capability + SoD).
- **P1**: payment IBAN is read from `BankAccount(verified=True, source="admin_callback")` only.

## Consequences

- Good: violations fail at construction time (Pydantic) or type-check time (mypy); easy to review.
- Bad: more types and one extra hop (grounding) per field; the agent sees less context.
- Grounding misses become low-OCR-support features that lower trust rather than errors.

## Proof (gate)

- `tests/unit/test_interfaces.py`: `StructuredRequest` rejects `tools`, thinking and high
  temperature; `RawField` rejects boxes; `UntrustedText` rejects over-cap text.
- Prompt-builder unit tests assert no raw document text outside quoted blocks; grounding
  test asserts every box equals a union of parse-word boxes; property tests for I3.
- Attack suite: payment-changing attack success = 0 end to end.
