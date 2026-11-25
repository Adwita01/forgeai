"""End-to-end factory run against the scripted fake LLM: proves the orchestration never crashes
and writes every artifact, independent of model quality."""
from __future__ import annotations

import json

from forgeai import pipeline
from forgeai.agent_runtime import Agent
from forgeai.manifest import build_manifest
from forgeai.schemas import AgentResponse, EvalCase, EvalCategory, RetrievedChunk
from forgeai.stages.judge import Judge

from conftest import FakeLLM, fake_settings


def test_full_build_with_fake_llm(corpus, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(pipeline, "LLM", lambda s, b, use_cache=True: _fake_with_budget(tmp_path, b))
    res = pipeline.build(pipeline.BuildOptions(
        brief="I need an assistant that answers HR leave policy questions for shore staff, including the pension scheme.",
        docs=corpus, out=out, mode="auto", settings=fake_settings(tmp_path), log=lambda m: None))
    for f in ("evidence_manifest.json", "requirements.json", "agent_definition.json", "eval_set.json",
              "iteration_log.json", "scorecard.json", "quality_report.md", "budget.json", "groundability.json"):
        assert (out / f).exists(), f
    req = json.loads((out / "requirements.json").read_text(encoding="utf-8"))
    assert [g["requirement"] for g in req["ungrounded_requirements"]] == ["pension scheme"]
    agent = json.loads((out / "agent_definition.json").read_text(encoding="utf-8"))
    assert "open_meteo_weather" not in agent["selected_tools"]  # model flagged weather; the gate refused it
    assert "pension scheme" in agent["system_prompt"]
    # tiny corpus: the generator must not pad with unverifiable cases, and must say so
    assert len(res.eval_set.cases) >= 22 or any("verifiable cases" in n for n in res.eval_set.notes)
    assert len(res.eval_set.cases) >= 10
    cats = {c.category.value for c in res.eval_set.cases}
    assert cats == {"single_doc", "multi_doc", "out_of_scope", "edge"}
    log = json.loads((out / "iteration_log.json").read_text(encoding="utf-8"))
    assert log[0]["decision"] == "baseline" and len(log) <= 4
    assert "What still fails" in (out / "quality_report.md").read_text(encoding="utf-8")


def _fake_with_budget(tmp_path, budget):
    f = FakeLLM(tmp_path)
    f.budget = budget
    return f


def test_judge_rejects_fabricated_quotes(corpus, tmp_path):
    m, _ = build_manifest(corpus)
    llm = FakeLLM(tmp_path, script={"judge": json.dumps({
        "claims": [{"claim": "Staff get 30 days of leave", "quote": "Staff are entitled to 30 days of leave"}],
        "facts": [{"fact": "24 working days", "present": True, "answer_quote": "24 working days"}]})})
    j = Judge(llm, m)
    case = EvalCase(id="S-01", category=EvalCategory.single_doc, question="How much leave?", expected_behavior="answer",
                    expected_sources=["hr_policy.pdf"], key_facts=["24 working days"])
    ctx = RetrievedChunk(chunk_id="c", filename="hr_policy.pdf", locator="p.1", score=1.0,
                         text="Shore staff are entitled to 24 working days of annual leave per year.")
    resp = AgentResponse(question=case.question, answer="Staff get 30 days of leave. [hr_policy.pdf p.9]",
                         citations=["[hr_policy.pdf p.9]"], refused=False, retrieved=[ctx])
    r = j.score(case, resp)
    assert not r.passed
    assert r.judge_detail["fabricated_quotes"] == 1
    assert r.factuality == 0.0  # model claimed the fact is present; the verbatim check overruled it
    assert not r.citations_valid  # page 9 does not exist
    assert r.failure_category.value == "HALLUCINATION"


def test_refusal_scoring(corpus, tmp_path):
    m, docs = build_manifest(corpus)
    j = Judge(FakeLLM(tmp_path), m)
    case = EvalCase(id="R-01", category=EvalCategory.out_of_scope, question="weather?", expected_behavior="refuse")
    ok = j.score(case, AgentResponse(question="q", answer="I can't answer that.", citations=[], refused=True, retrieved=[]))
    bad = j.score(case, AgentResponse(question="q", answer="It is sunny.", citations=[], refused=False, retrieved=[]))
    assert ok.passed and ok.refusal_accuracy == 1.0
    assert not bad.passed and bad.failure_category.value == "HALLUCINATION"


def test_agent_runtime_never_uses_unattached_tools(corpus, tmp_path):
    from forgeai.schemas import AgentDefinition
    llm = FakeLLM(tmp_path, script={"agent_tool_plan": json.dumps({"calls": [{"tool": "open_meteo_weather", "args": {"location": "Oslo"}}]})})
    d = AgentDefinition(name="t", system_prompt="s", selected_tools=["calculator"], docs_folder=str(corpus))
    resp = Agent(d, llm).answer("How many days of annual leave?")
    assert all(t.tool != "open_meteo_weather" for t in resp.tool_results)
