from __future__ import annotations

import json

import pytest

from forgeai.budget import Budget, BudgetExceeded
from forgeai.llm import StructuredOutputError, extract_json
from forgeai.manifest import build_manifest
from forgeai.parsing import load_folder
from forgeai.retriever import HybridRetriever
from forgeai.schemas import QA, Capabilities, ClarifyingQuestions, Requirements, RetrievalConfig
from forgeai.stages.generator import select_tools
from forgeai.stages.interviewer import reconcile_capabilities
from forgeai.stages.judge import validate_citation
from forgeai.tools import ToolContext, run_tool
from forgeai.tools.calculator import evaluate

from conftest import FakeLLM


# ---------------------------------------------------------------- JSON + self-healing
def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1,}\n```') == {"a": 1}
    assert extract_json('Sure! Here you go: {"a": [1, 2], "b": "x}"} hope it helps') == {"a": [1, 2], "b": "x}"}
    assert extract_json("<think>hmm {no}</think>{\"ok\": True}") == {"ok": True}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_structured_self_heals(tmp_path):
    replies = iter(['{"questions": "not a list of objects', '{"questions": [{"question": "Who uses it?"}]}'])
    llm = FakeLLM(tmp_path, script={"t": lambda m: next(replies)})
    out = llm.structured(ClarifyingQuestions, "sys", "user", stage="t")
    assert out.questions[0].question == "Who uses it?"
    assert llm.budget.stages["t"].retries == 1
    assert llm.calls.count("t") == 2


def test_structured_feeds_errors_back_then_gives_up(tmp_path):
    seen = []

    def bad(messages):
        seen.append(messages[-1]["content"])
        return '{"wrong": 1}'

    llm = FakeLLM(tmp_path, script={"t": bad})
    with pytest.raises(StructuredOutputError):
        llm.structured(Requirements, "sys", "user", stage="t")
    assert len(seen) == 4  # 1 + 3 retries
    assert "scope" in seen[1] and "rejected by the validator" in seen[1]


def test_bare_list_is_wrapped(tmp_path):
    llm = FakeLLM(tmp_path, script={"t": '[{"question": "a?"}, {"question": "b?"}]'})
    assert len(llm.structured(ClarifyingQuestions, "s", "u", stage="t").questions) == 2


def test_budget_exhaustion():
    b = Budget(max_tokens=100)
    b.record("x", 80, 30, 0.1)
    with pytest.raises(BudgetExceeded):
        b.check()


# ---------------------------------------------------------------- tools
def test_calculator_safe():
    assert evaluate("12 * 45.5 + 30") == 576
    assert evaluate("$1,200 / 4") == 300
    assert evaluate("round(10/3, 2)") == 3.33
    assert evaluate("max(3, 7) x 2") == 14
    for bad in ("__import__('os').system('x')", "open('f')", "2 ** 10000", "[x for x in ()]"):
        with pytest.raises(Exception):
            evaluate(bad)


def test_csv_lookup(corpus):
    docs, _ = load_folder(corpus)
    ctx = ToolContext(tables={d.filename: d.table for d in docs if d.kind == "csv"})
    out, err = run_tool("csv_lookup", {"filters": {"part no": "AV-2002"}}, ctx)
    assert err is None and out["n_matches"] == 1 and out["rows"][0]["row"] == 2
    out, _ = run_tool("csv_lookup", {"filters": {"stock_qty": "<20"}, "aggregate": {"op": "sum", "column": "unit_price_usd"}}, ctx)
    assert out["aggregate"]["value"] == 355.5
    out, _ = run_tool("csv_lookup", {"query": "gasket"}, ctx)
    assert out["rows"][0]["part_no"] == "GS-3003"
    _, err = run_tool("nope", {}, ctx)
    assert err


# ---------------------------------------------------------------- manifest / retrieval / citations
def test_manifest_and_citation_validation(corpus):
    m, docs = build_manifest(corpus)
    pdf = m.file("hr_policy.pdf")
    assert pdf.n_pages == 2 and m.file("parts.csv").n_rows == 3
    price = next(c for c in m.file("parts.csv").columns if c.name == "unit_price_usd")
    assert price.dtype == "number" and price.max == 310
    assert validate_citation("[hr_policy.pdf p.2]", m)[0]
    assert not validate_citation("[hr_policy.pdf p.7]", m)[0]
    assert not validate_citation("[ghost.pdf p.1]", m)[0]
    assert validate_citation("[parts.csv row 3]", m)[0]
    assert not validate_citation("[parts.csv row 4]", m)[0]


def test_hybrid_retrieval(corpus):
    docs, _ = load_folder(corpus)
    r = HybridRetriever(docs, RetrievalConfig(top_k=2))
    assert r.dense_backend == "tfidf"
    hits = r.search("how many days of annual leave do I get")
    assert hits[0].filename == "hr_policy.pdf" and hits[0].locator == "p.1"
    assert r.search("AV-2002 air valve")[0].locator == "row 2"


# ---------------------------------------------------------------- tool gating
def _req(**caps):
    return Requirements(scope="x", capabilities=Capabilities(**caps))


def test_tool_gating_hr_brief_no_weather(corpus):
    m, _ = build_manifest(corpus)
    brief = "I need an assistant that answers HR leave policy questions for shore staff."
    req = reconcile_capabilities(_req(live_weather=True, arithmetic=True, date_awareness=True), brief, [], m)
    tools, why = select_tools(req, m)
    assert "open_meteo_weather" not in tools and "calculator" not in tools and "current_date" not in tools
    assert "model flagged" in why["open_meteo_weather"]


def test_tool_gating_procurement_brief(corpus):
    m, _ = build_manifest(corpus)
    brief = "Help engineers look up spare parts stock in the catalogue and calculate the total cost of an order."
    req = reconcile_capabilities(_req(), brief, [], m)
    tools, _ = select_tools(req, m)
    assert tools == ["csv_lookup", "calculator"]


def test_negated_answer_disables_capability(corpus):
    m, _ = build_manifest(corpus)
    qa = [QA(question="Need calculations?", answer="No calculations are needed, never compute totals.", source="user")]
    req = reconcile_capabilities(_req(), "Answer questions about parts and their total cost.", qa, m)
    assert not req.capabilities.arithmetic


def test_csv_tool_requires_csv(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.md").write_text("# Leave\nStaff get 20 days.", encoding="utf-8")
    m, _ = build_manifest(d)
    req = reconcile_capabilities(_req(tabular_lookup=True), "look up the inventory table", [], m)
    assert "csv_lookup" not in select_tools(req, m)[0]
