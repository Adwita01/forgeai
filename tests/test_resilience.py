"""Budget cuts and dead model servers must end a build cleanly, never hang or crash without artifacts."""
from __future__ import annotations

import json

import pytest
import requests

from forgeai import llm as llm_mod
from forgeai import pipeline
from forgeai.budget import Budget
from forgeai.llm import LLM, BackendUnavailable

from conftest import FakeLLM, fake_settings

BRIEF = "I need an assistant that answers HR leave policy questions for shore staff, including the pension scheme."


def _patched(monkeypatch, tmp_path, script=None):
    def make(s, b, use_cache=True):
        f = FakeLLM(tmp_path, script=script)
        f.budget = b
        return f
    monkeypatch.setattr(pipeline, "LLM", make)


def test_budget_cut_mid_eval_writes_partial_report(corpus, tmp_path, monkeypatch):
    # 1st run (unlimited) measures how many tokens the stages before evaluation need
    _patched(monkeypatch, tmp_path)
    full = pipeline.build(pipeline.BuildOptions(brief=BRIEF, docs=corpus, out=tmp_path / "full", settings=fake_settings(tmp_path), log=lambda m: None))
    pre_eval = sum(v["prompt_tokens"] + v["completion_tokens"] for k, v in full.budget["stages"].items()
                   if k not in ("agent_answer", "agent_tool_plan", "judge", "optimizer"))
    # 2nd run: budget runs out a few cases into the baseline evaluation
    s = fake_settings(tmp_path).with_overrides(max_tokens=pre_eval + 150 * 6)
    out = tmp_path / "cut"
    res = pipeline.build(pipeline.BuildOptions(brief=BRIEF, docs=corpus, out=out, settings=s, log=lambda m: None))
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "budget_exhausted"
    final = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))["final"]
    assert not final["complete"] and 0 < final["n_cases"] < final["n_planned"]
    assert len(res.history) == 1  # no optimisation on a partial baseline
    report = (out / "quality_report.md").read_text(encoding="utf-8")
    assert "cut short by its budget" in report and "scored (of" in report
    for f in ("requirements.json", "agent_definition.json", "eval_set.json", "iteration_log.json"):
        assert (out / f).exists(), f


def test_dead_backend_aborts_with_status(corpus, tmp_path, monkeypatch):
    def boom(messages):
        raise BackendUnavailable("[groundability] model server stalled")
    _patched(monkeypatch, tmp_path, script={"groundability": boom})
    out = tmp_path / "dead"
    with pytest.raises(BackendUnavailable):
        pipeline.build(pipeline.BuildOptions(brief=BRIEF, docs=corpus, out=out, settings=fake_settings(tmp_path), log=lambda m: None))
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "aborted" and status["stage"] == "groundability"
    assert "clarifications.json" in status["artifacts_written"] and "rerun the same command" in status["resume"]


def _real_llm(tmp_path, monkeypatch, exc, alive):
    llm = LLM(fake_settings(tmp_path), Budget(), use_cache=False)
    calls = {"n": 0}

    def fail(*a, **k):
        calls["n"] += 1
        raise exc

    monkeypatch.setattr(llm, "_call", fail)
    monkeypatch.setattr(llm, "_alive", lambda: alive)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    return llm, calls


def test_two_timeouts_mean_stalled_server(tmp_path, monkeypatch):
    llm, calls = _real_llm(tmp_path, monkeypatch, requests.Timeout("read timed out"), alive=True)
    with pytest.raises(BackendUnavailable, match="stalled"):
        llm.chat([{"role": "user", "content": "hi"}], stage="judge")
    assert calls["n"] == 2  # one retry, not 3 x request_timeout


def test_unreachable_server_fails_fast(tmp_path, monkeypatch):
    llm, calls = _real_llm(tmp_path, monkeypatch, requests.ConnectionError("refused"), alive=False)
    with pytest.raises(BackendUnavailable, match="not reachable"):
        llm.chat([{"role": "user", "content": "hi"}], stage="judge")
    assert calls["n"] == 2


def test_budget_does_not_block_cached_calls(tmp_path, monkeypatch):
    llm = LLM(fake_settings(tmp_path), Budget(max_tokens=1), use_cache=True)
    monkeypatch.setattr(llm, "_call", lambda *a, **k: ("cached-text", 5, 5))
    msgs = [{"role": "user", "content": "hi"}]
    assert llm.chat(msgs, stage="x") == "cached-text"   # spends the budget
    assert llm.budget.exhausted()
    assert llm.chat(msgs, stage="x") == "cached-text"   # replay still works (resume after a cut)
