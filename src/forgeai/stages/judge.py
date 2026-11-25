"""Stage 5 - Anti-bias hybrid judge.

Deterministic layer (cannot be talked out of a verdict):
  * every citation must name a file in the Evidence Manifest and an existing page/row/section;
  * refusal correctness is a boolean comparison;
  * numbers in expected facts must appear in the answer;
  * expected source retrieved / cited.
LLM layer, defended against known judge biases:
  * quote-before-score: the model only extracts verbatim evidence; scores are computed in
    code from quotes that are *verified* to exist in the context (fabricated quotes count 0);
  * verbosity: groundedness is a per-claim fraction, so extra text cannot raise it;
  * position bias: facts are checked one-by-one, never by comparing two answers;
  * self-preference: the answer is presented as "author unknown" and a separate judge model
    can be configured (FORGE_JUDGE_MODEL); final fact credit needs a verified quote or a
    deterministic match.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from ..agent_runtime import Agent, chunk_label
from ..llm import LLM, LLMError, load_prompt, split_prompt
from ..schemas import (
    AgentResponse,
    CaseResult,
    EvalCase,
    EvalCategory,
    EvalSet,
    EvidenceManifest,
    FailureCategory,
    JudgeExtraction,
    Scorecard,
)
from ..textutil import content_tokens, fact_supported, numbers, overlap, quote_in

_LOC = re.compile(r"^\[(.+?\.(?:pdf|csv|txt|md|markdown))\s+(p\.\s*(\d+)|row\s*(\d+)|§\s*(.+))\]$", re.I)

PASS_FACTUALITY = 0.5
PASS_GROUNDEDNESS = 0.6


def validate_citation(c: str, manifest: EvidenceManifest) -> tuple[bool, str]:
    m = _LOC.match(c.strip())
    if not m:
        return False, "unparseable citation"
    fname, page, row, sec = m.group(1), m.group(3), m.group(4), m.group(5)
    fe = manifest.file(fname)
    if fe is None:
        return False, f"file not in manifest: {fname}"
    if page is not None:
        if fe.kind != "pdf" or not (1 <= int(page) <= fe.n_pages):
            return False, f"page {page} does not exist in {fname} ({fe.n_pages} pages)"
    if row is not None:
        if fe.kind != "csv" or not (1 <= int(row) <= fe.n_rows):
            return False, f"row {row} does not exist in {fname} ({fe.n_rows} rows)"
    if sec is not None:
        if not any(sec.strip().lower() in s.lower() or s.lower() in sec.strip().lower() for s in fe.sections):
            return False, f"section '{sec}' not found in {fname}"
    return True, ""


def _cited_file(c: str) -> str:
    m = _LOC.match(c.strip())
    return m.group(1).lower() if m else ""


def _deterministic_groundedness(answer: str, evidence: str) -> float:
    sents = [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\[[^\]]*\]", "", answer)) if len(content_tokens(s)) >= 3]
    if not sents:
        return 0.0
    ok = sum(1 for s in sents if overlap(s, evidence) >= 0.6 and numbers(s) <= numbers(evidence))
    return ok / len(sents)


class Judge:
    def __init__(self, llm: LLM, manifest: EvidenceManifest, require_citations: bool = True):
        self.llm = llm
        self.manifest = manifest
        self.require_citations = require_citations

    def score(self, case: EvalCase, resp: AgentResponse) -> CaseResult:
        cites = resp.citations
        checks = [(c, *validate_citation(c, self.manifest)) for c in cites]
        invalid = [f"{c} ({why})" for c, ok, why in checks if not ok]
        valid_files = {_cited_file(c) for c, ok, _ in checks if ok}
        retrieved_files = list(dict.fromkeys(r.filename for r in resp.retrieved))
        exp = {s.lower() for s in case.expected_sources}
        tool_files = {str(t.output.get("file", "")).lower() for t in resp.tool_results if isinstance(t.output, dict)}
        exp_retrieved = (not exp) or bool(exp & {f.lower() for f in retrieved_files}) or bool(exp & tool_files)
        exp_cited = (not exp) or bool(exp & valid_files)

        evidence = "\n".join(f"{chunk_label(r)} {r.text}" for r in resp.retrieved)
        evidence += "\n" + "\n".join(f"{t.tool}: {t.output}" for t in resp.tool_results if t.error is None)

        should_refuse = case.expected_behavior == "refuse"
        refusal_ok = resp.refused == should_refuse
        detail: dict = {"judge_mode": "skipped"}
        groundedness = factuality = 1.0 if (should_refuse and resp.refused) else 0.0
        numeric_ok: Optional[bool] = None

        if not resp.refused and resp.answer.strip():
            fact_nums = set().union(*[numbers(f) for f in case.key_facts]) if case.key_facts else set()
            if fact_nums:
                numeric_ok = fact_nums <= numbers(resp.answer)
            if should_refuse:
                # answering an out-of-scope question fails regardless of content - no LLM call needed
                groundedness = _deterministic_groundedness(resp.answer, evidence)
                factuality, detail = 0.0, {"judge_mode": "deterministic", "note": "answered a must-refuse question"}
            else:
                det_facts = [fact_supported(f, resp.answer) for f in case.key_facts]
                groundedness, factuality, detail = self._llm_scores(case, resp, evidence, det_facts)

        cit_present = bool(valid_files) if not resp.refused else True
        cit_valid = not invalid
        if should_refuse:
            passed = resp.refused
        else:
            passed = (
                not resp.refused
                and factuality >= PASS_FACTUALITY
                and groundedness >= PASS_GROUNDEDNESS
                and cit_valid
                and (cit_present or not self.require_citations)
            )

        res = CaseResult(
            case_id=case.id, category=case.category, question=case.question, expected_behavior=case.expected_behavior,
            answer=resp.answer, refused=resp.refused, citations=cites, retrieved_files=retrieved_files,
            citations_valid=cit_valid, invalid_citations=invalid, citation_present=cit_present,
            expected_source_retrieved=exp_retrieved, expected_source_cited=exp_cited, numeric_ok=numeric_ok,
            groundedness=round(groundedness, 3), factuality=round(factuality, 3), refusal_accuracy=1.0 if refusal_ok else 0.0,
            passed=passed, judge_detail=detail, latency_s=resp.latency_s,
        )
        if not passed:
            res.failure_category, res.failure_reason = classify_failure(case, res)
        return res

    def _llm_scores(self, case: EvalCase, resp: AgentResponse, evidence: str, det_facts: list[bool]) -> tuple[float, float, dict]:
        facts_txt = "\n".join(f"- {f}" for f in case.key_facts) or "- (none: only check claims)"
        sys_t, user_t = split_prompt(load_prompt("judge_extract", self.llm.s, question=case.question, context=evidence[:6000],
                                                 answer=re.sub(r"\[[^\]]*\]", "", resp.answer)[:2500], facts=facts_txt))
        try:
            ex = self.llm.structured(JudgeExtraction, sys_t, user_t, stage="judge", model=self.llm.s.judge_model,
                                     temperature=0.0, max_tokens=1200)
        except LLMError as e:
            g = _deterministic_groundedness(resp.answer, evidence)
            f = sum(det_facts) / len(det_facts) if det_facts else g
            return g, f, {"judge_mode": "deterministic_fallback", "error": str(e)[:200]}

        claims = [c for c in ex.claims if c.claim.strip()]
        verified = [bool(c.quote) and quote_in(c.quote, evidence) for c in claims]
        fabricated = sum(1 for c, v in zip(claims, verified) if c.quote and not v)
        if claims:
            groundedness = sum(verified) / len(claims)
        else:
            groundedness = _deterministic_groundedness(resp.answer, evidence)
        # deterministic sentence-level support can rescue claims the small judge failed to quote
        groundedness = max(groundedness, 0.5 * groundedness + 0.5 * _deterministic_groundedness(resp.answer, evidence))

        llm_fact = {}
        for fc in ex.facts:
            llm_fact[fc.fact.strip().lower()] = fc.present and bool(fc.answer_quote) and quote_in(fc.answer_quote, resp.answer, min_len=2)
        fact_hits = []
        for f, det in zip(case.key_facts, det_facts):
            llm_ok = llm_fact.get(f.strip().lower())
            if llm_ok is None:  # model paraphrased the fact text; align by best overlap
                best = max(ex.facts, key=lambda fc: overlap(f, fc.fact), default=None)
                llm_ok = bool(best and overlap(f, best.fact) >= 0.6 and best.present and quote_in(best.answer_quote, resp.answer, min_len=2))
            nums = numbers(f)
            if nums and not nums <= numbers(resp.answer):
                llm_ok = False  # numbers must match exactly, whatever the model says
            fact_hits.append(det or llm_ok)
        factuality = sum(fact_hits) / len(fact_hits) if fact_hits else groundedness
        return groundedness, factuality, {
            "judge_mode": "quote_verified_llm",
            "n_claims": len(claims),
            "verified_claims": sum(verified),
            "fabricated_quotes": fabricated,
            "unsupported_claims": [c.claim for c, v in zip(claims, verified) if not v][:5],
            "facts": {f: hit for f, hit in zip(case.key_facts, fact_hits)},
        }


def classify_failure(case: EvalCase, r: CaseResult) -> tuple[FailureCategory, str]:
    """Deterministic critic: map a failed case to exactly one actionable category."""
    if case.expected_behavior == "refuse":
        return FailureCategory.HALLUCINATION, "answered a question that should have been refused"
    if r.refused:
        if not r.expected_source_retrieved:
            return FailureCategory.RETRIEVAL_MISS, "refused because the source passage was never retrieved"
        return FailureCategory.WRONG_REFUSAL, "refused although the expected source was in the retrieved context"
    if not r.expected_source_retrieved and r.factuality < PASS_FACTUALITY:
        return FailureCategory.RETRIEVAL_MISS, f"expected source {case.expected_sources} not retrieved"
    if r.groundedness < PASS_GROUNDEDNESS:
        return FailureCategory.HALLUCINATION, f"only {r.groundedness:.0%} of claims are supported by the context"
    if not r.citations_valid:
        return FailureCategory.PROMPT_DEFECT, f"invalid citations: {r.invalid_citations[:2]}"
    if not r.citation_present:
        return FailureCategory.PROMPT_DEFECT, "answer carries no valid citation"
    return FailureCategory.PROMPT_DEFECT, f"evidence was retrieved but answer missed expected facts (factuality {r.factuality:.0%})"


def scorecard(results: list[CaseResult], iteration: int, agent_version: int) -> Scorecard:
    n = len(results) or 1
    ans = [r for r in results if r.expected_behavior == "answer"] or results

    def avg(xs, key):
        xs = list(xs)
        return round(sum(key(x) for x in xs) / len(xs), 3) if xs else 0.0

    by_cat: dict[str, dict[str, float]] = {}
    for cat in EvalCategory:
        rs = [r for r in results if r.category == cat]
        if rs:
            by_cat[cat.value] = {"n": len(rs), "pass_rate": avg(rs, lambda r: r.passed), "factuality": avg(rs, lambda r: r.factuality),
                                 "groundedness": avg(rs, lambda r: r.groundedness)}
    fails: dict[str, int] = {}
    for r in results:
        if r.failure_category:
            fails[r.failure_category.value] = fails.get(r.failure_category.value, 0) + 1
    pass_rate = avg(results, lambda r: r.passed)
    g, f, ra = avg(ans, lambda r: r.groundedness), avg(ans, lambda r: r.factuality), avg(results, lambda r: r.refusal_accuracy)
    cv = avg([r for r in results if not r.refused] or results, lambda r: r.citations_valid and r.citation_present)
    overall = round(0.4 * pass_rate + 0.2 * g + 0.2 * f + 0.2 * ra, 4)
    return Scorecard(iteration=iteration, agent_version=agent_version, n_cases=len(results), pass_rate=pass_rate, groundedness=g,
                     factuality=f, refusal_accuracy=ra, citation_validity=cv, by_category=by_cat, failures_by_type=fails, overall=overall)


def run_eval(agent: Agent, judge: Judge, eval_set: EvalSet, iteration: int = 0,
             progress: Optional[Callable[[int, int, CaseResult], None]] = None) -> tuple[list[CaseResult], Scorecard]:
    results = []
    for i, case in enumerate(eval_set.cases):
        resp = agent.answer(case.question)
        r = judge.score(case, resp)
        results.append(r)
        if progress:
            progress(i + 1, len(eval_set.cases), r)
    return results, scorecard(results, iteration, agent.d.version)
