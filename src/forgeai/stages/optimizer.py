"""Stage 6 - Surgical critic-optimizer loop (max 3 iterations).

Each iteration: count failures by category -> patch the (at most two) dominant categories
with the smallest targeted change -> re-run the full eval set -> keep the patch if the overall
score did not regress, otherwise revert it and mark it as tried. Early-stop when an iteration
gains less than `min_gain` or when the budget cannot afford another full eval run.

Patches never regenerate the agent. They are:
  RETRIEVAL_MISS -> retrieval_config tweak (top_k, hybrid alpha, chunk size) from a ladder;
  HALLUCINATION / WRONG_REFUSAL / PROMPT_DEFECT -> 1-2 general prompt rules (LLM-written,
     leakage-filtered so no test answers enter the prompt) or a canned rule as fallback,
     plus a one-time output-contract adjustment where relevant.
"""
from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Callable, Optional

from ..budget import BudgetExceeded
from ..llm import LLM, LLMError, load_prompt, split_prompt
from ..schemas import AgentDefinition, CaseResult, EvalSet, FailureCategory, IterationRecord, LearnedRules, Patch, Scorecard
from ..textutil import numbers, overlap

MEANING = {
    FailureCategory.HALLUCINATION: "The assistant stated things not supported by its retrieved documents, or answered a question it should have refused.",
    FailureCategory.WRONG_REFUSAL: "The assistant refused although the retrieved documents contained the answer.",
    FailureCategory.PROMPT_DEFECT: "The documents were retrieved but the answer missed required facts, lacked valid citations, or ignored the output format.",
    FailureCategory.RETRIEVAL_MISS: "The passage containing the answer was not retrieved.",
}

CANNED_RULES = {
    FailureCategory.HALLUCINATION: [
        "Before answering, check that each sentence you write is directly stated in a context passage; delete any sentence that is not, and refuse if nothing remains.",
        "If the question asks about a topic, item, country or period that the passages do not mention by name, refuse instead of generalising from similar passages.",
    ],
    FailureCategory.WRONG_REFUSAL: [
        "If any passage contains information relevant to the question, answer from it even when the wording differs from the question; refuse only when no passage is relevant.",
        "For questions with typos, abbreviations or vague wording, infer the most likely intended topic from the passages and answer it, stating your interpretation.",
    ],
    FailureCategory.PROMPT_DEFECT: [
        "End every sentence that states a fact with the citation label of its passage, copied exactly from the passage header, e.g. [policy.pdf p.2].",
        "State every number, duration, amount and identifier from the relevant passage explicitly in the answer, and when two passages are relevant combine both, citing each.",
    ],
}

RETRIEVAL_LADDER = ["top_k_up", "alpha_to_bm25", "chunk_smaller", "alpha_to_dense", "top_k_up_2"]


def _retrieval_patch(d: AgentDefinition, key: str) -> Optional[tuple[str, object, object]]:
    rc = d.retrieval_config
    if key in ("top_k_up", "top_k_up_2") and rc.top_k < 10:
        return "retrieval_config.top_k", rc.top_k, min(10, rc.top_k + 2)
    if key == "alpha_to_bm25" and rc.hybrid_alpha > 0.2:
        return "retrieval_config.hybrid_alpha", rc.hybrid_alpha, round(max(0.15, rc.hybrid_alpha - 0.2), 2)
    if key == "alpha_to_dense" and rc.hybrid_alpha < 0.8:
        return "retrieval_config.hybrid_alpha", rc.hybrid_alpha, round(min(0.85, rc.hybrid_alpha + 0.25), 2)
    if key == "chunk_smaller" and rc.chunk_size > 80:
        return "retrieval_config.chunk_size", rc.chunk_size, int(rc.chunk_size * 0.7)
    return None


def _set_path(d: AgentDefinition, path: str, value) -> None:
    obj = d
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)
    if path == "retrieval_config.chunk_size":
        d.retrieval_config.chunk_overlap = max(10, value // 6)


class Optimizer:
    def __init__(self, llm: LLM, eval_set: EvalSet):
        self.llm = llm
        self.eval_set = eval_set
        self.tried: set[str] = set()
        self.rejected_rules: set[str] = set()
        # leakage guard: no rule may carry test-specific facts into the prompt
        self._fact_text = [f for c in eval_set.cases for f in c.key_facts]
        self._fact_numbers = set().union(*[numbers(f) for f in self._fact_text]) if self._fact_text else set()

    def reject(self, patches: list[Patch]) -> None:
        """Remember rules from a reverted patch so they are never proposed again."""
        for p in patches:
            if p.kind == "prompt_rule":
                self.rejected_rules.update(set(p.after) - set(p.before))

    def _leaks(self, rule: str) -> bool:
        if numbers(rule) & self._fact_numbers - {"1", "2"}:
            return True
        return any(overlap(f, rule) >= 0.6 and len(f.split()) >= 3 for f in self._fact_text)

    def _rules_for(self, cat: FailureCategory, d: AgentDefinition, failures: list[CaseResult]) -> tuple[list[str], str]:
        cases_txt = "\n".join(
            f"- Q: {r.question}\n  A: {r.answer[:300]}\n  problem: {r.failure_reason}" for r in failures[:4]
        )
        sys_t, user_t = split_prompt(load_prompt("optimizer_rules", self.llm.s, category=cat.value, meaning=MEANING[cat],
                                                 existing="\n".join(f"- {x}" for x in d.prompt_rules) or "(none)", cases=cases_txt))
        rules, source = [], "llm"
        try:
            proposed = self.llm.structured(LearnedRules, sys_t, user_t, stage="optimizer", temperature=0.3, max_tokens=400).rules
        except LLMError:
            proposed = []
        for r in proposed:
            r = re.sub(r"\s+", " ", r).strip()
            if not (8 <= len(r.split()) <= 50) or self._leaks(r):
                continue
            if any(overlap(r, e) >= 0.8 for e in list(d.prompt_rules) + list(self.rejected_rules)):
                continue
            rules.append(r)
        if not rules:
            source = "canned"
            rules = [r for r in CANNED_RULES[cat] if r not in d.prompt_rules and r not in self.rejected_rules][:1]
        return rules[:2], source

    def propose(self, current: AgentDefinition, results: list[CaseResult]) -> tuple[AgentDefinition, list[Patch]]:
        failed = [r for r in results if not r.passed and r.failure_category]
        counts = Counter(r.failure_category for r in failed)
        new = copy.deepcopy(current)
        patches: list[Patch] = []
        for cat, _ in counts.most_common():
            if len(patches) >= 2:
                break
            cat_fail = [r for r in failed if r.failure_category == cat]
            ids = [r.case_id for r in cat_fail]
            if cat == FailureCategory.RETRIEVAL_MISS:
                for key in RETRIEVAL_LADDER:
                    if key in self.tried:
                        continue
                    p = _retrieval_patch(new, key)
                    if p:
                        path, before, after = p
                        _set_path(new, path, after)
                        patches.append(Patch(target=cat, kind="retrieval", field=path, before=before, after=after, addresses_cases=ids,
                                             rationale=f"[{key}] {len(cat_fail)} answers failed because the source passage was not in the top-{current.retrieval_config.top_k} results"))
                        self.tried.add(key)
                        break
                continue
            rules, source = self._rules_for(cat, new, cat_fail)
            if rules:
                before = list(new.prompt_rules)
                new.prompt_rules = before + rules
                patches.append(Patch(target=cat, kind="prompt_rule", field="prompt_rules", before=before, after=new.prompt_rules, addresses_cases=ids,
                                     rationale=f"{len(cat_fail)} {cat.value} failures; added {len(rules)} {source} rule(s)"))
            # one-time contract adjustments
            if cat == FailureCategory.PROMPT_DEFECT and any(not r.citation_present or not r.citations_valid for r in cat_fail) and "contract_cite" not in self.tried:
                before = new.output_contract.citation_format
                after = before + ". Copy labels exactly from the passage headers; never invent page or row numbers."
                new.output_contract.citation_format = after
                patches.append(Patch(target=cat, kind="contract", field="output_contract.citation_format", before=before, after=after,
                                     addresses_cases=[r.case_id for r in cat_fail if not r.citations_valid or not r.citation_present],
                                     rationale="citations missing or pointing at non-existent pages/rows"))
                self.tried.add("contract_cite")
            if cat == FailureCategory.WRONG_REFUSAL and "contract_refusal_soften" not in self.tried:
                before = new.output_contract.refusal_policy
                after = before + " Partial information is not a reason to refuse: answer what the passages state and name what is missing."
                new.output_contract.refusal_policy = after
                patches.append(Patch(target=cat, kind="contract", field="output_contract.refusal_policy", before=before, after=after,
                                     addresses_cases=ids, rationale="agent refused despite relevant retrieved evidence"))
                self.tried.add("contract_refusal_soften")
        return new, patches


def optimize(
    agent_def: AgentDefinition,
    eval_set: EvalSet,
    evaluate: Callable[[AgentDefinition, int], tuple[list[CaseResult], Scorecard]],
    llm: LLM,
    max_iters: int = 3,
    min_gain: float = 0.02,
    log: Callable[[str], None] = print,
) -> tuple[AgentDefinition, list[IterationRecord], list[Scorecard], dict[int, list[CaseResult]], dict[int, AgentDefinition]]:
    """Returns (best definition, iteration log, scorecards, results per iteration, definitions per iteration)."""
    budget = llm.budget
    t_tok, t_sec = budget.tokens_used, budget.elapsed
    results, sc = evaluate(agent_def, 0)
    run_tokens, run_secs = budget.tokens_used - t_tok, budget.elapsed - t_sec
    history = [IterationRecord(iteration=0, agent_version=agent_def.version, score_before=sc.overall, score_after=sc.overall,
                               pass_rate_before=sc.pass_rate, pass_rate_after=sc.pass_rate, failures_by_type=sc.failures_by_type,
                               decision="baseline", tokens_used=run_tokens, seconds_used=round(run_secs, 1))]
    scorecards, all_results, defs = [sc], {0: results}, {0: agent_def}
    best, best_sc, best_res = agent_def, sc, results
    opt = Optimizer(llm, eval_set)
    log(f"iteration 0 (baseline): overall {sc.overall:.3f}, pass {sc.pass_rate:.0%}, failures {sc.failures_by_type}")

    for it in range(1, max_iters + 1):
        if best_sc.pass_rate >= 1.0:
            history[-1].stop_reason = "all cases pass"
            break
        # budget-aware early stop: can we afford another full eval run?
        if budget.max_tokens and budget.tokens_used + run_tokens > budget.max_tokens:
            history.append(IterationRecord(iteration=it, agent_version=best.version, score_before=best_sc.overall, pass_rate_before=best_sc.pass_rate,
                                           failures_by_type=best_sc.failures_by_type, decision="stopped",
                                           stop_reason=f"token budget: next iteration needs ~{run_tokens} tokens, {budget.max_tokens - budget.tokens_used} left"))
            break
        if budget.max_seconds and budget.elapsed + run_secs > budget.max_seconds:
            history.append(IterationRecord(iteration=it, agent_version=best.version, score_before=best_sc.overall, pass_rate_before=best_sc.pass_rate,
                                           failures_by_type=best_sc.failures_by_type, decision="stopped",
                                           stop_reason=f"time budget: next iteration needs ~{run_secs:.0f}s, {budget.max_seconds - budget.elapsed:.0f}s left"))
            break

        try:
            cand, patches = opt.propose(best, best_res)
        except BudgetExceeded as e:
            history[-1].stop_reason = str(e)
            break
        if not patches:
            history.append(IterationRecord(iteration=it, agent_version=best.version, score_before=best_sc.overall, pass_rate_before=best_sc.pass_rate,
                                           failures_by_type=best_sc.failures_by_type, decision="stopped", stop_reason="no untried targeted patch left"))
            break
        cand.version = max(defs[k].version for k in defs) + 1
        log(f"iteration {it}: applying {len(patches)} patch(es): " + "; ".join(f"{p.target.value}->{p.field}" for p in patches))
        t_tok, t_sec = budget.tokens_used, budget.elapsed
        try:
            res, csc = evaluate(cand, it)
        except BudgetExceeded as e:
            history.append(IterationRecord(iteration=it, agent_version=cand.version, score_before=best_sc.overall, pass_rate_before=best_sc.pass_rate,
                                           failures_by_type=best_sc.failures_by_type, patches=patches, decision="stopped", stop_reason=f"budget exhausted mid-run: {e}"))
            break
        run_tokens, run_secs = budget.tokens_used - t_tok, budget.elapsed - t_sec
        gain = csc.overall - best_sc.overall
        rec = IterationRecord(iteration=it, agent_version=cand.version, score_before=best_sc.overall, score_after=csc.overall,
                              pass_rate_before=best_sc.pass_rate, pass_rate_after=csc.pass_rate, failures_by_type=csc.failures_by_type,
                              patches=patches, tokens_used=run_tokens, seconds_used=round(run_secs, 1))
        scorecards.append(csc)
        all_results[it], defs[it] = res, cand
        if gain < -0.005:
            rec.decision = "reverted"
            opt.reject(patches)
            rec.stop_reason = f"overall regressed by {-gain:.3f}; patch reverted, will try a different patch"
            log(f"iteration {it}: overall {csc.overall:.3f} ({gain:+.3f}) -> REVERTED")
        else:
            rec.decision = "kept"
            best, best_sc, best_res = cand, csc, res
            log(f"iteration {it}: overall {csc.overall:.3f} ({gain:+.3f}) -> kept")
            if gain < min_gain:
                rec.stop_reason = f"early stop: gain {gain:+.3f} below min_gain {min_gain}"
                history.append(rec)
                break
        history.append(rec)
    return best, history, scorecards, all_results, defs
