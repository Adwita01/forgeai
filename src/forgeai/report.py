"""quality_report.md - honest, evidence-first reporting."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .config import Settings
from .schemas import AgentDefinition, CaseResult, EvalSet, EvidenceManifest, FailureCategory, IterationRecord, Requirements, Scorecard

WHY_PERSISTS = {
    FailureCategory.RETRIEVAL_MISS: (
        "The passage holding the answer never reached the model. Typical causes: the question uses different vocabulary "
        "than the document (synonyms the lexical half cannot match), or the fact sits in a table / long page whose chunk "
        "is dominated by other content. Retrieval patches trade precision for recall and stop paying off after a few steps."
    ),
    FailureCategory.HALLUCINATION: (
        "The model added content not in the context or answered an out-of-scope question. Small models (3-7B) follow "
        "'answer only from context' imperfectly, especially when a near-miss passage looks relevant; prompt rules reduce "
        "but do not eliminate this."
    ),
    FailureCategory.WRONG_REFUSAL: (
        "The evidence was retrieved but the model refused - usually over-correction from strict grounding rules, or "
        "the evidence was phrased differently from the question. Loosening refusal tends to raise hallucination; "
        "the loop keeps whichever balance scores higher overall."
    ),
    FailureCategory.PROMPT_DEFECT: (
        "Evidence was present but the answer omitted expected facts or citations. Partly genuine model error, partly "
        "eval strictness: key facts are matched literally (numbers exactly), so a correct paraphrase can be under-credited."
    ),
}


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def write_report(out: Path, brief: str, manifest: EvidenceManifest, req: Requirements, verdicts, agent: AgentDefinition,
                 eval_set: EvalSet, history: list[IterationRecord], scorecards: list[Scorecard], final: list[CaseResult],
                 budget: dict, gen_meta: dict, s: Settings, status: dict | None = None) -> Path:
    final_sc = next(sc for sc in reversed(scorecards) if sc.agent_version == agent.version)
    L: list[str] = []
    w = L.append
    w(f"# Quality report - {agent.name}\n")
    scored = f"{final_sc.n_cases}" if final_sc.complete else f"{final_sc.n_cases} scored (of {final_sc.n_planned})"
    w(f"> **Final: {_pct(final_sc.pass_rate)} of {scored} generated test cases pass** "
      f"(overall score {final_sc.overall:.3f}). Model `{s.model}`, judge `{s.judge_model}`. "
      f"The numbers below are produced by an automated judge over an automatically generated test set; "
      f"see *Limitations of this evaluation* before trusting them.\n")
    if status and status.get("status") != "complete":
        lim = status.get("limits", {})
        w(f"> ⚠ **This run was cut short by its budget** (limits: {lim.get('max_tokens') or '∞'} tokens / "
          f"{lim.get('max_seconds') or '∞'}s). Everything below describes what was completed; treat the scores as provisional.")
        for c in dict.fromkeys(status.get("cut_short", [])):
            w(f">  - {c}")
        w("")
    w("## Brief\n")
    w("\n".join(f"> {line}" for line in brief.strip().splitlines()) + "\n")

    # ---------------- requirements & guard
    w("## Requirements and groundability\n")
    w(f"- **Scope:** {req.scope}\n- **Audience / tone:** {req.audience} / {req.tone}")
    w(f"- **Grounded topics:** {', '.join(req.grounded_topics) or '-'}")
    if req.partially_grounded_topics:
        w(f"- **Partially grounded (agent told to state what is missing):** {', '.join(req.partially_grounded_topics)}")
    if req.ungrounded_requirements:
        w("- **⚠ Ungrounded requirements flagged by the guard (agent will refuse these):**")
        for g in req.ungrounded_requirements:
            w(f"  - *{g.requirement}* - {g.reason} (closest evidence: {g.best_evidence or 'none'})")
    else:
        w("- **Ungrounded requirements:** none detected")
    w(f"- **Must refuse:** {', '.join(req.must_refuse_topics) or '-'}\n")

    # ---------------- agent
    w("## Generated agent\n")
    w("| Tool | Decision |\n|---|---|")
    for t, why in agent.tool_rationale.items():
        w(f"| `{t}` | {why} |")
    rc = agent.retrieval_config
    w(f"\nRetrieval: chunk {rc.chunk_size} words / overlap {rc.chunk_overlap}, top_k {rc.top_k}, hybrid alpha {rc.hybrid_alpha} "
      f"(dense weight). Rationale: {gen_meta.get('retrieval_rationale', '')}. Persona written by: {gen_meta.get('persona_source')}.")
    if agent.prompt_rules:
        w("\nRules learned by the optimizer:")
        for r in agent.prompt_rules:
            w(f"- {r}")
    w("")

    # ---------------- eval set
    cats = Counter(c.category.value for c in eval_set.cases)
    w("## Generated eval set\n")
    w(f"{len(eval_set.cases)} cases: " + ", ".join(f"{k} {v}" for k, v in sorted(cats.items())) +
      f". Every answerable case's key facts were verified to occur in its source passage before inclusion.")
    if eval_set.notes:
        w(f"Generator notes ({len(eval_set.notes)}): " + "; ".join(eval_set.notes[:6]) + ("; ..." if len(eval_set.notes) > 6 else ""))
    w("")

    # ---------------- iterations
    w("## Per-iteration scorecard\n")
    w("| Iter | Agent v | Decision | Pass rate | Groundedness | Factuality | Refusal acc. | Citation validity | Overall | Failures |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    sc_by_it = {sc.iteration: sc for sc in scorecards}
    for h in history:
        sc = sc_by_it.get(h.iteration)
        if sc is None:
            w(f"| {h.iteration} | - | {h.decision} | - | - | - | - | - | - | {h.stop_reason} |")
            continue
        fails = ", ".join(f"{k}:{v}" for k, v in sc.failures_by_type.items()) or "none"
        w(f"| {h.iteration} | {sc.agent_version} | {h.decision} | {_pct(sc.pass_rate)} | {sc.groundedness:.2f} | {sc.factuality:.2f} | "
          f"{sc.refusal_accuracy:.2f} | {sc.citation_validity:.2f} | **{sc.overall:.3f}** | {fails} |")
    w("")
    for h in history[1:]:
        w(f"### Iteration {h.iteration} - {h.decision}")
        if h.score_after is not None:
            w(f"Score {h.score_before:.3f} → {h.score_after:.3f}; pass rate {_pct(h.pass_rate_before)} → {_pct(h.pass_rate_after or 0)}. "
              f"Cost: {h.tokens_used} tokens, {h.seconds_used:.0f}s.")
        for p in h.patches:
            before, after = p.before, p.after
            if p.kind == "prompt_rule":
                added = [x for x in after if x not in before]
                change = "added rule(s): " + " / ".join(f'"{x}"' for x in added)
            else:
                change = f"`{p.field}`: {before} → {after}"
            w(f"- **{p.target.value}** ({p.kind}) - {change}. Why: {p.rationale}. Targets: {', '.join(p.addresses_cases[:8])}")
        if h.stop_reason:
            w(f"- *{h.stop_reason}*")
        w("")

    # ---------------- per category
    w("## Final results by category\n")
    w("| Category | n | Pass rate | Factuality | Groundedness |\n|---|---|---|---|---|")
    for cat, v in final_sc.by_category.items():
        w(f"| {cat} | {int(v['n'])} | {_pct(v['pass_rate'])} | {v['factuality']:.2f} | {v['groundedness']:.2f} |")
    w("")

    # ---------------- what still fails
    failed = [r for r in final if not r.passed]
    w(f"## What still fails and why ({len(failed)} of {len(final)})\n")
    if not failed:
        w("No generated case fails. That is more likely a sign the generated tests are too easy than that the agent is perfect - "
          "see the limitations below.\n")
    groups: dict[FailureCategory, list[CaseResult]] = defaultdict(list)
    for r in failed:
        groups[r.failure_category or FailureCategory.PROMPT_DEFECT].append(r)
    for cat, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        w(f"### {cat.value} - {len(rs)} case(s)")
        w(WHY_PERSISTS[cat] + "\n")
        for r in rs[:6]:
            ans = r.answer.replace("\n", " ")[:220]
            w(f"- `{r.case_id}` ({r.category.value}) *{r.question}*  \n  → {r.failure_reason}. Answer: \"{ans}\"")
        w("")

    # ---------------- judge integrity
    fabricated = sum(int(r.judge_detail.get("fabricated_quotes", 0)) for r in final)
    fallback = sum(1 for r in final if r.judge_detail.get("judge_mode") == "deterministic_fallback")
    invalid_c = sum(1 for r in final if not r.citations_valid)
    w("## Judge integrity checks\n")
    w(f"- Judge quotes rejected as not verbatim in the context (fabricated or garbled): **{fabricated}**")
    rescued = sum(int(r.judge_detail.get("claims_rescued", 0)) for r in final)
    w(f"- Of those, claims still accepted because the claim itself checks out deterministically "
      f"(exact numbers, >=60% word overlap with the context): **{rescued}**")
    w(f"- Cases scored by the deterministic fallback because the LLM judge failed schema validation: **{fallback}**")
    w(f"- Answers citing a file/page/row that does not exist in the Evidence Manifest: **{invalid_c}**")
    w(f"- Expected sources retrieved: {_pct(sum(r.expected_source_retrieved for r in final) / max(1, len(final)))}; "
      f"expected sources cited: {_pct(sum(r.expected_source_cited for r in final if r.expected_behavior == 'answer') / max(1, sum(1 for r in final if r.expected_behavior == 'answer')))}\n")

    # ---------------- limitations
    w("## Limitations of this evaluation\n")
    w("- The eval set is generated by the same small model family that powers the agent. Questions it can write tend to be ones "
      "it can answer, so pass rates are optimistic. Key facts are verified against the source text, but question *difficulty* is not.")
    w("- The optimizer tunes against this same eval set (no held-out split - the set is small). Gains may partly be overfitting; "
      "rules are leakage-filtered so no test facts enter the prompt, which limits but does not remove this.")
    w("- Factuality is matched literally (numbers must match exactly), so correct paraphrases with different units or rounding are under-credited.")
    w(f"- Judge model = agent model unless `FORGE_JUDGE_MODEL` is set (here: {'different' if s.judge_model != s.model else 'same'}); "
      "self-preference is mitigated by quote verification, not eliminated.")
    w("")

    w("## Budget\n")
    w(f"Total {budget['tokens_used']} tokens, {budget.get('wall_clock_s', budget['elapsed_s'])}s wall clock "
      f"(limits: {budget['max_tokens'] or '∞'} tokens / {budget['max_seconds'] or '∞'}s).\n")
    w("| Stage | Calls | Tokens | Seconds | Schema retries | Hard failures |\n|---|---|---|---|---|---|")
    for k, v in budget["stages"].items():
        w(f"| {k} | {v['calls']} | {v['prompt_tokens'] + v['completion_tokens']} | {v['seconds']} | {v['schema_retries']} | {v['hard_failures']} |")
    w("")
    if manifest.warnings:
        w("## Document warnings\n")
        for x in manifest.warnings:
            w(f"- {x}")
    path = out / "quality_report.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
