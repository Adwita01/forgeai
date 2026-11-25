"""Evaluate any saved agent definition against any eval set, and compare two agents.

Used by `forge eval` (incl. regression checks after documents change) and `forge compare`
(factory-built agent vs a hand-written baseline).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from .agent_runtime import Agent
from .llm import LLM
from .manifest import build_manifest
from .schemas import AgentDefinition, CaseResult, EvalSet, Scorecard
from .stages.judge import Judge, run_eval


def load_definition(path: str | Path) -> tuple[AgentDefinition, Path]:
    p = Path(path)
    if p.is_dir():
        p = p / "agent_definition.json"
    return AgentDefinition.model_validate_json(p.read_text(encoding="utf-8")), p.parent


def resolve_docs(defn: AgentDefinition, base_dir: Path) -> Path:
    folder = Path(defn.docs_folder)
    return folder if folder.is_absolute() else (base_dir / folder).resolve()


def evaluate_definition(def_path: str | Path, eval_set_path: str | Path, llm: LLM,
                        progress: Optional[Callable[[int, int, CaseResult], None]] = None,
                        docs_override: Optional[Path] = None) -> tuple[AgentDefinition, list[CaseResult], Scorecard]:
    defn, base = load_definition(def_path)
    docs_dir = docs_override or resolve_docs(defn, base)
    manifest, docs = build_manifest(docs_dir)
    eval_set = EvalSet.model_validate_json(Path(eval_set_path).read_text(encoding="utf-8"))
    agent = Agent(defn, llm, docs=docs)
    judge = Judge(llm, manifest, require_citations=defn.output_contract.require_citations)
    results, sc = run_eval(agent, judge, eval_set, iteration=0, progress=progress)
    return defn, results, sc


def compare_markdown(name_a: str, sc_a: Scorecard, res_a: list[CaseResult], name_b: str, sc_b: Scorecard, res_b: list[CaseResult],
                     eval_set_path: str) -> str:
    L = [f"# Agent comparison: {name_a} vs {name_b}\n",
         f"Both agents were run on the same eval set (`{eval_set_path}`), the same model and the same judge.\n",
         f"| Metric | {name_a} | {name_b} | Δ |", "|---|---|---|---|"]
    for key in ("pass_rate", "overall", "groundedness", "factuality", "refusal_accuracy", "citation_validity"):
        a, b = getattr(sc_a, key), getattr(sc_b, key)
        L.append(f"| {key} | {a:.3f} | {b:.3f} | {a - b:+.3f} |")
    L.append("\n## By category\n")
    L.append(f"| Category | {name_a} pass | {name_b} pass |\n|---|---|---|")
    for cat in sorted(set(sc_a.by_category) | set(sc_b.by_category)):
        pa = sc_a.by_category.get(cat, {}).get("pass_rate", 0)
        pb = sc_b.by_category.get(cat, {}).get("pass_rate", 0)
        L.append(f"| {cat} | {pa:.0%} | {pb:.0%} |")
    by_b = {r.case_id: r for r in res_b}
    wins_a = [r for r in res_a if r.passed and not by_b[r.case_id].passed]
    wins_b = [by_b[r.case_id] for r in res_a if not r.passed and by_b[r.case_id].passed]
    both_fail = [r for r in res_a if not r.passed and not by_b[r.case_id].passed]
    L.append(f"\n## Where {name_a} wins ({len(wins_a)})\n")
    for r in wins_a:
        b = by_b[r.case_id]
        L.append(f"- `{r.case_id}` *{r.question}* - {name_b} failed: {b.failure_category.value if b.failure_category else '?'} ({b.failure_reason})")
    L.append(f"\n## Where {name_b} wins ({len(wins_b)})\n")
    for r in wins_b:
        a = next(x for x in res_a if x.case_id == r.case_id)
        L.append(f"- `{r.case_id}` *{r.question}* - {name_a} failed: {a.failure_category.value if a.failure_category else '?'} ({a.failure_reason})")
    L.append(f"\n## Both fail ({len(both_fail)})\n")
    for r in both_fail:
        L.append(f"- `{r.case_id}` *{r.question}* - {r.failure_category.value if r.failure_category else '?'}")
    return "\n".join(L) + "\n"


def save_results(out: Path, results: list[CaseResult], sc: Scorecard, stem: str = "eval") -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}_results.json").write_text(json.dumps([r.model_dump(mode="json") for r in results], indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"{stem}_scorecard.json").write_text(sc.model_dump_json(indent=2), encoding="utf-8")
