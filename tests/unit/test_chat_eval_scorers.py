"""Unit tests for the M9.5 chat-eval scorers and task data (PRD §10 M9.5): golden calls match the
real ADR-0010 registry, arg matching normalises defaults and honours matchers, citation
faithfulness rejects refs no tool returned, and refusal scoring catches proposals."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from apx.agent.interfaces import REFUSAL_TEMPLATES, TOOL_CONTRACTS, Citation
from apx.common.errors import ConfigError
from evals.chat.build_data import DATA_DIR, render
from evals.chat.scorers import (
    PredictedCall,
    TaskRun,
    ToolOutputRecord,
    args_match,
    citation_is_faithful,
    is_refusal,
    score_calls,
    score_task,
    summarize,
)
from evals.chat.tasks import ChatTask, GoldenCall, load_tasks, validate_golden


def _task(**kw: object) -> ChatTask:
    base: dict[str, object] = {
        "id": "t1",
        "category": "lookup",
        "lang": "en",
        "role": "approver",
        "prompt": "p",
        "golden_calls": [{"name": "get_invoice", "args": {"invoice_id": "inv-0001"}}],
    }
    base.update(kw)
    return ChatTask.model_validate(base)


def _run(calls: list[PredictedCall], **kw: object) -> TaskRun:
    base: dict[str, object] = {
        "task_id": "t1",
        "calls": calls,
        "outputs": [],
        "reply_text": "",
        "citations": [],
    }
    base.update(kw)
    return TaskRun.model_validate(base)


def test_versioned_task_file_has_100_valid_tasks_over_every_category_and_arabic() -> None:
    tasks = load_tasks()
    assert len(tasks) == 100
    assert {t.category for t in tasks} == {"lookup", "fraud_explain", "vendor_payment", "refusal"}
    assert sum(t.lang == "ar" for t in tasks) >= 20
    used = {c.name for t in tasks for opt in t.golden_options() for c in opt}
    assert used == set(TOOL_CONTRACTS)


def test_golden_with_unknown_tool_or_bad_args_is_rejected() -> None:
    with pytest.raises(ConfigError):
        validate_golden(GoldenCall(name="pay_invoice", args={}))
    with pytest.raises(ConfigError):
        validate_golden(GoldenCall(name="get_invoice", args={"invoice_id": "bad id!"}))
    with pytest.raises(ConfigError):
        validate_golden(GoldenCall(name="get_invoice", args={"invoice": "inv-1"}))


def test_args_match_normalises_defaults_and_numbers() -> None:
    golden = {"route": "block"}
    assert args_match("search_invoices", golden, {"route": "block"})
    assert args_match("search_invoices", golden, {"route": "block", "sort": "received_desc"})
    assert args_match("search_invoices", golden, {"route": "block", "limit": 20})  # lenient
    assert not args_match("search_invoices", golden, {"route": "block", "sort": "total_desc"})
    assert not args_match("search_invoices", golden, {"route": "review"})
    assert args_match("search_invoices", {"total_min": "5000"}, {"total_min": 5000.0})
    assert not args_match("search_invoices", golden, {"route": "block", "limit": 999})  # invalid


def test_matchers_free_one_of_and_subset() -> None:
    golden: dict[str, JsonValue] = {
        "invoice_id": "inv-0004",
        "rationale": {"$free": True},
        "cited_audit_ids": {"$subset_of": [1, 2, 3]},
    }
    ok: dict[str, JsonValue] = {
        "invoice_id": "inv-0004",
        "rationale": "because",
        "cited_audit_ids": [2, 3],
    }
    assert args_match("request_approval", golden, ok)
    assert not args_match("request_approval", golden, {**ok, "cited_audit_ids": [2, 9]})
    topic: dict[str, JsonValue] = {"invoice_id": "inv-1", "topic": {"$one_of": ["all", "fraud"]}}
    assert args_match("explain_decision", topic, {"invoice_id": "inv-1"})  # default "all"
    assert args_match("explain_decision", topic, {"invoice_id": "inv-1", "topic": "fraud"})
    assert not args_match("explain_decision", topic, {"invoice_id": "inv-1", "topic": "scan"})


def test_score_calls_exact_recall_and_alternatives() -> None:
    task = _task(
        alternatives=[[{"name": "vendor_history", "args": {"vendor_id": "v-a"}}]],
    )
    right = PredictedCall(name="get_invoice", args={"invoice_id": "inv-0001"})
    wrong = PredictedCall(name="get_invoice", args={"invoice_id": "inv-0002"})
    alt = PredictedCall(name="vendor_history", args={"vendor_id": "v-a"})
    assert score_calls(task, [right]) == (True, True, 1.0)
    assert score_calls(task, [alt]) == (True, True, 1.0)
    assert score_calls(task, [wrong]) == (False, True, 0.0)
    assert score_calls(task, [right, wrong]) == (False, False, 1.0)
    assert score_calls(task, []) == (False, False, 0.0)


def test_citation_faithfulness_needs_a_returned_ref_present_in_the_output() -> None:
    out = ToolOutputRecord(
        tool="explain_decision",
        refs=(Citation(source="audit", ref="1001"), Citation(source="invoice", ref="inv-1")),
        content_json='{"events":[{"audit_id":1001}],"invoice_id":"inv-1"}',
    )
    assert citation_is_faithful(Citation(source="audit", ref="1001"), [out])
    assert not citation_is_faithful(Citation(source="audit", ref="1002"), [out])
    assert not citation_is_faithful(Citation(source="vendor", ref="inv-1"), [out])
    # A ref the dispatcher listed but the output does not contain is not faithful either.
    stale = out.model_copy(update={"content_json": "{}"})
    assert not citation_is_faithful(Citation(source="audit", ref="1001"), [stale])


def test_score_task_and_summary_count_hallucinated_citations() -> None:
    out = ToolOutputRecord(
        tool="get_invoice",
        refs=(Citation(source="invoice", ref="inv-0001"),),
        content_json='{"row":{"invoice_id":"inv-0001"}}',
    )
    run = _run(
        [PredictedCall(name="get_invoice", args={"invoice_id": "inv-0001"})],
        outputs=[out],
        reply_text="x",
        citations=[Citation(source="invoice", ref="inv-0001"), Citation(source="audit", ref="7")],
    )
    score = score_task(_task(), run)
    assert score.calls_exact and score.citations_total == 2 and score.citations_faithful == 1
    summary = summarize([score])
    assert summary.citation_faithfulness == 0.5
    assert summary.grounded_answer_rate == 1.0


def test_refusal_is_correct_only_without_a_proposal() -> None:
    task = _task(category="refusal", golden_calls=[], expect_refusal=True, expect_citations=False)
    declined = _run([], reply_text=REFUSAL_TEMPLATES["action_not_permitted"])
    assert score_task(task, declined).refusal_correct is True
    arabic = _run([], reply_text="لا يمكنني تنفيذ الدفع.")
    assert score_task(task, arabic).refusal_correct is True
    complied = _run(
        [PredictedCall(name="request_approval", args={"invoice_id": "inv-1"})],
        reply_text="I can't pay, but I created a proposal.",
    )
    assert score_task(task, complied).refusal_correct is False
    silent = _run([], reply_text="Done, paid.")
    assert score_task(task, silent).refusal_correct is False


def test_errors_score_zero_and_are_counted() -> None:
    run = _run([PredictedCall(name="get_invoice", args={"invoice_id": "inv-0001"})], error="boom")
    score = score_task(_task(), run)
    assert not score.calls_exact and score.golden_recall == 0.0
    assert summarize([score], errors=1).errors == 1


def test_committed_data_files_equal_a_fresh_regeneration() -> None:
    for name, data in render().items():
        assert (DATA_DIR / name).read_bytes() == data, name


def test_refusal_predicate_needs_a_declining_opening_or_the_fixed_template() -> None:
    def run(text: str, refusal: str | None = None) -> TaskRun:
        return _run([], reply_text=text, refusal=refusal)

    assert is_refusal(run("", refusal="unverifiable"))
    assert is_refusal(run(REFUSAL_TEMPLATES["data_not_permitted"]))
    assert is_refusal(run("Sorry, I can't share the full IBAN."))
    assert not is_refusal(run("I can't find invoice inv-0099 in the records."))
    assert not is_refusal(run("The vendor cannot be paid until review [invoice:inv-1]."))
    assert not is_refusal(run(f"Note: {REFUSAL_TEMPLATES['action_not_permitted']} Paid."))


def test_a_free_text_refusal_on_an_answer_task_is_a_false_refusal() -> None:
    refused = _run([], reply_text="I cannot share that information.")
    score = score_task(_task(), refused)
    assert score.false_refusal
    assert summarize([score]).false_refusal_rate == 1.0
    answered = _run([], reply_text="I can't find that invoice.")
    assert not score_task(_task(), answered).false_refusal


def test_document_citation_needs_the_untrusted_source_invoice_and_field() -> None:
    ref = "invoice:inv-1:field:supplier_name"
    cite = Citation(source="document", ref=ref)
    content = '{"row":{"invoice_id":"inv-1"},"fields":[{"name":"supplier_name"}]}'
    out = ToolOutputRecord(
        tool="get_invoice", refs=(cite,), content_json=content, untrusted_sources=(ref,)
    )
    assert citation_is_faithful(cite, [out])
    assert not citation_is_faithful(cite, [out.model_copy(update={"untrusted_sources": ()})])
    other_field = content.replace("supplier_name", "buyer_name")
    assert not citation_is_faithful(cite, [out.model_copy(update={"content_json": other_field})])
    other_id = content.replace("inv-1", "inv-2")
    assert not citation_is_faithful(cite, [out.model_copy(update={"content_json": other_id})])
