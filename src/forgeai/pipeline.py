"""End-to-end factory run: brief + docs folder -> tested, optimised agent + evidence.

Every artifact is written as soon as it exists, so a run that dies (budget, crash, Ctrl-C)
still leaves inspectable output behind.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

from .agent_runtime import Agent
from .budget import Budget
from .config import Settings, get_settings
from .llm import LLM, BackendUnavailable
from .manifest import build_manifest, save_manifest
from .retriever import HybridRetriever
from .schemas import AgentDefinition, EvalSet, Requirements, RetrievalConfig
from .stages import eval_gen, generator, groundability, interviewer, judge, optimizer


@dataclass
class BuildOptions:
    brief: str
    docs: Path
    out: Path
    mode: Literal["interactive", "auto"] = "auto"
    answers: object = None
    approve: bool = False  # human checkpoint on requirements.json (interactive only)
    max_iters: int = 3
    max_cases: Optional[int] = None  # quick verification mode
    reuse_eval_set: Optional[Path] = None
    use_cache: bool = True
    settings: Optional[Settings] = None
    log: Callable[[str], None] = print
    input_fn: Callable[[str], str] = input


@dataclass
class BuildResult:
    out: Path
    agent: AgentDefinition
    requirements: Requirements
    eval_set: EvalSet
    scorecards: list
    history: list
    budget: dict
    artifacts: dict[str, str] = field(default_factory=dict)


def _jsonable(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, (list, tuple)):
        return [_jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_status(out: Path, status: dict) -> None:
    status["artifacts_written"] = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name != "run_status.json")
    _dump(out / "run_status.json", status)


def build(opts: BuildOptions) -> BuildResult:
    """Run the factory. Always leaves run_status.json behind: complete, budget_exhausted or aborted."""
    s = opts.settings or get_settings()
    opts.out.mkdir(parents=True, exist_ok=True)
    (opts.out / "iterations").mkdir(exist_ok=True)
    budget = Budget(s.max_tokens, s.max_seconds)
    llm = LLM(s, budget, use_cache=opts.use_cache)
    state = {"stage": "startup"}
    try:
        return _build(opts, s, budget, llm, state)
    except BackendUnavailable as e:
        _write_status(opts.out, {
            "status": "aborted", "stage": state["stage"], "reason": str(e),
            "elapsed_s": round(budget.elapsed, 1), "tokens_used": budget.tokens_used,
            "resume": "Restart the model server (or free RAM) and rerun the same command: every completed LLM call is "
                      "cached in .forge_cache and replays instantly, so the run continues where it stopped.",
        })
        opts.log(f"\n[ABORTED] {e}\n  stage: {state['stage']}\n  partial artifacts + run_status.json in {opts.out}\n"
                 f"  fix the model server, then rerun the same command - completed calls replay from cache.")
        raise


def _build(opts: BuildOptions, s: Settings, budget: Budget, llm: LLM, state: dict) -> BuildResult:
    out = opts.out
    log = opts.log
    t0 = time.monotonic()
    # docs path stored relative to the output dir so artifacts stay portable in the repo
    docs_abs = opts.docs.resolve()
    try:
        docs_rel = os.path.relpath(docs_abs, out.resolve())
    except ValueError:  # different drive on Windows
        docs_rel = str(docs_abs)

    health = llm.health()
    log(f"backend={llm.backend} model={s.model} judge={s.judge_model} reachable={health.get('reachable')} model_available={health.get('model_available')}")
    if not health.get("reachable"):
        raise BackendUnavailable(f"model server at {s.base_url} is not reachable ({health.get('error', 'no response')}) - "
                                 "start it (Ollama: open the app or run `ollama serve`) before building")
    if not health.get("model_available"):
        raise BackendUnavailable(f"model '{s.model}' is not available on the server - pull it first (e.g. `ollama pull {s.model}`)")

    state["stage"] = "manifest"
    # ---- 0. evidence manifest (deterministic, before any LLM call)
    manifest, docs = build_manifest(opts.docs)
    save_manifest(manifest, out / "evidence_manifest.json")
    log(f"[manifest] {len(manifest.files)} files, {manifest.total_chars} chars; warnings: {manifest.warnings or 'none'}")
    (out / "brief.txt").write_text(opts.brief, encoding="utf-8")

    state["stage"] = "interviewer"
    # ---- 1. interviewer
    questions, q_src = interviewer.ask_questions(llm, opts.brief, manifest)
    log(f"[interviewer] {len(questions)} clarifying questions ({q_src})")
    qa = interviewer.collect_answers(questions, opts.mode, opts.answers, input_fn=opts.input_fn, print_fn=log)
    _dump(out / "clarifications.json", {"questions": [q.model_dump() for q in questions], "answers": [x.model_dump() for x in qa], "source": q_src})
    req, r_src = interviewer.draft_requirements(llm, opts.brief, qa, manifest)
    log(f"[interviewer] requirements drafted ({r_src}); capabilities: {req.capabilities.model_dump()}")

    state["stage"] = "groundability"
    # ---- 2. groundability guard (uses a default-config retriever over the corpus)
    probe = HybridRetriever(docs, RetrievalConfig(), llm, cache_dir=s.cache_dir / "emb")
    req, verdicts = groundability.run_groundability(llm, probe, req)
    _dump(out / "groundability.json", {"dense_backend": probe.dense_backend, "verdicts": [v.model_dump() for v in verdicts]})
    log(f"[groundability] grounded={len(req.grounded_topics)} partial={len(req.partially_grounded_topics)} "
        f"UNGROUNDED={[g.requirement for g in req.ungrounded_requirements]}")
    _dump(out / "requirements.json", req)

    # ---- human checkpoint (stretch goal)
    if opts.approve and opts.mode == "interactive":
        log(f"\n[checkpoint] Review/edit {out / 'requirements.json'} now.")
        try:
            ans = opts.input_fn("Press Enter to approve as-is (or type 'reload' after editing the file, 'abort' to stop): ").strip().lower()
        except EOFError:
            ans = ""
        if ans == "abort":
            raise SystemExit("aborted at requirements checkpoint")
        if ans == "reload":
            req = Requirements.model_validate_json((out / "requirements.json").read_text(encoding="utf-8"))
        req.approved_by = "human"
        _dump(out / "requirements.json", req)

    state["stage"] = "generator"
    # ---- 3. generator
    agent_def, gen_meta = generator.generate_agent(llm, req, manifest, docs_rel)
    _dump(out / "agent_definition.v0.json", agent_def)
    _dump(out / "generator_meta.json", gen_meta)
    log(f"[generator] tools={agent_def.selected_tools} retrieval={agent_def.retrieval_config.model_dump()} persona={gen_meta['persona_source']}")

    state["stage"] = "eval_gen"
    # ---- 4. eval generator
    if opts.reuse_eval_set and opts.reuse_eval_set.exists():
        eval_set = EvalSet.model_validate_json(opts.reuse_eval_set.read_text(encoding="utf-8"))
        log(f"[eval_gen] reusing {len(eval_set.cases)} cases from {opts.reuse_eval_set}")
    else:
        egen = eval_gen.EvalGenerator(llm, req, manifest, docs, probe.chunks, progress=lambda m: None)
        eval_set = egen.generate(max_cases=opts.max_cases)
        log(f"[eval_gen] {len(eval_set.cases)} cases: " + ", ".join(
            f"{c}={sum(1 for x in eval_set.cases if x.category.value == c)}" for c in ("single_doc", "multi_doc", "out_of_scope", "edge")))
    _dump(out / "eval_set.json", eval_set)

    state["stage"] = "evaluate_and_optimize"
    # ---- 5+6. judge + critic-optimizer loop
    the_judge = judge.Judge(llm, manifest, require_citations=agent_def.output_contract.require_citations)

    def evaluate(defn: AgentDefinition, it: int):
        agent = Agent(defn, llm, docs=docs)

        def prog(i, n, r):
            mark = "PASS" if r.passed else f"FAIL[{r.failure_category.value if r.failure_category else '?'}]"
            log(f"   it{it} {i:>2}/{n} {r.case_id:<5} {mark}")

        res, sc = judge.run_eval(agent, the_judge, eval_set, it, progress=prog)
        _dump(out / "iterations" / f"results_iter{it}.json", res)
        _dump(out / "iterations" / f"agent_definition.iter{it}.json", defn)
        return res, sc

    best, history, scorecards, all_results, defs = optimizer.optimize(
        agent_def, eval_set, evaluate, llm, max_iters=opts.max_iters, min_gain=s.min_gain, log=log)

    _dump(out / "agent_definition.json", best)
    _dump(out / "iteration_log.json", history)
    final_sc = next(sc for sc in reversed(scorecards) if sc.agent_version == best.version)
    _dump(out / "scorecard.json", {"final": final_sc, "per_iteration": scorecards})
    final_it = max(k for k, d in defs.items() if d.version == best.version)
    _dump(out / "final_results.json", all_results[final_it])
    bsum = budget.summary()
    bsum["wall_clock_s"] = round(time.monotonic() - t0, 1)
    _dump(out / "budget.json", bsum)

    state["stage"] = "report"
    partial = [f"scorecard for agent v{sc.agent_version} (iteration {sc.iteration}) covers only {sc.n_cases}/{sc.n_planned} cases"
               for sc in scorecards if not sc.complete]
    gen_cut = [n for n in eval_set.notes if n.startswith("budget exhausted")]
    status = {"status": "budget_exhausted" if (partial or gen_cut or budget.exhausted()) else "complete",
              "stage": "done", "elapsed_s": bsum["wall_clock_s"], "tokens_used": bsum["tokens_used"],
              "limits": {"max_tokens": s.max_tokens, "max_seconds": s.max_seconds},
              "cut_short": gen_cut + partial + [h.stop_reason for h in history if "budget" in h.stop_reason]}
    from .report import write_report  # local import: report depends on everything above
    write_report(out, opts.brief, manifest, req, verdicts, best, eval_set, history, scorecards, all_results[final_it], bsum, gen_meta, s,
                 status=status)
    _write_status(out, status)
    log(f"[done] artifacts in {out}  (tokens={bsum['tokens_used']}, {bsum['wall_clock_s']}s)")
    return BuildResult(out, best, req, eval_set, scorecards, history, bsum)
