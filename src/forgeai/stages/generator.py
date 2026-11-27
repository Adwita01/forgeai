"""Stage 3 - Declarative agent generator.

Split of responsibility, chosen for small-model reliability:
  * LLM writes only the persona (role, domain guidance, style);
  * everything safety-relevant is deterministic: the system-prompt skeleton (scope,
    refusals, gaps, grounding rules), tool pruning, retrieval parameters, output contract.
"""
from __future__ import annotations

import re
from pathlib import Path
from statistics import mean

from ..llm import FALLBACK_ERRORS, LLM, LLMError, load_prompt, split_prompt
from ..manifest import manifest_for_prompt
from ..retriever import tokenize
from ..schemas import (
    AgentDefinition,
    DraftPersona,
    EvidenceManifest,
    OutputContract,
    Requirements,
    RetrievalConfig,
)
from ..tools import REGISTRY


def select_tools(req: Requirements, manifest: EvidenceManifest) -> tuple[list[str], dict[str, str]]:
    """Attach a tool only if its capability is required AND its preconditions hold."""
    chosen, why = [], {}
    for name, spec in REGISTRY.items():
        needed = getattr(req.capabilities, spec.capability, False)
        ev = req.capability_evidence.get(spec.capability, "")
        if needed and spec.requires_csv and not manifest.has_csv:
            why[name] = "rejected: needs a CSV table but none was provided"
        elif needed:
            chosen.append(name)
            why[name] = f"attached: {ev or spec.capability + ' required'}"
        else:
            why[name] = f"not attached: {ev or spec.capability + ' not required'}"
    return chosen, why


def retrieval_params(manifest: EvidenceManifest) -> tuple[RetrievalConfig, str]:
    prose = [f for f in manifest.files if f.kind in ("pdf", "text")]
    has_csv = manifest.has_csv
    avg_unit = mean([f.n_chars / max(1, f.n_pages or len(f.sections) or 1) for f in prose]) if prose else 0
    # identifier-heavy corpora (part numbers, clause ids) need more lexical weight
    ids = sum(1 for f in manifest.files for e in f.key_entities if re.search(r"\d", e) and re.search(r"[A-Z]", e))
    alpha = 0.5
    notes = []
    if ids >= 5 or has_csv:
        alpha = 0.35
        why = "tabular (CSV) corpus" if has_csv else f"{ids} code-like identifiers (clause/document ids)"
        notes.append(f"{why} -> more BM25 weight (alpha 0.35)")
    chunk = 160 if avg_unit < 2500 else 220
    notes.append(f"avg citable unit {avg_unit:.0f} chars -> chunk {chunk} words")
    n_units = sum(max(1, f.n_pages) for f in prose) + sum(f.n_rows for f in manifest.files if f.kind == "csv")
    top_k = 5 if n_units < 60 else 6
    if has_csv:
        top_k += 1
        notes.append("CSV rows are one chunk each -> top_k +1")
    return RetrievalConfig(chunk_size=chunk, chunk_overlap=max(20, chunk // 6), top_k=top_k, hybrid_alpha=alpha), "; ".join(notes)


def _bullets(items: list[str], empty: str = "- (none)") -> str:
    return "\n".join(f"- {x}" for x in items) if items else empty


def _citation_format(manifest: EvidenceManifest) -> str:
    kinds = {f.kind for f in manifest.files}
    parts = []
    if "pdf" in kinds:
        parts.append("[<filename> p.<page>] for PDF passages")
    if "csv" in kinds:
        parts.append("[<filename> row <n>] for table rows")
    if "text" in kinds:
        parts.append("[<filename> §<section>] for text sections")
    return "; ".join(parts) or "[<filename> <locator>]"


def _fallback_persona(req: Requirements) -> DraftPersona:
    return DraftPersona(
        role_statement=f"You are {req.agent_name}, an assistant for {req.audience}. {req.scope}",
        domain_guidance=["Quote figures, durations and identifiers exactly as written.", "Say which document a rule comes from."],
        style_rules=[f"Tone: {req.tone}.", "Lead with the direct answer, then the supporting detail."],
    )


def build_persona(llm: LLM, req: Requirements, manifest: EvidenceManifest) -> tuple[DraftPersona, str]:
    req_view = req.model_dump(include={"agent_name", "scope", "audience", "tone", "must_answer_topics", "must_refuse_topics"})
    sys_t, user_t = split_prompt(load_prompt("generator_persona", llm.s, requirements=req_view, manifest=manifest_for_prompt(manifest, 2000)))
    try:
        p = llm.structured(DraftPersona, sys_t, user_t, stage="generator", max_tokens=700)
        # drop guidance about things the corpus doesn't contain (small models copy prompt examples)
        vocab = set(tokenize(manifest_for_prompt(manifest, 20_000) + " " + " ".join(req.must_answer_topics) + " " + req.scope))
        generic = {"answer", "question", "quote", "exact", "document", "documents", "policy", "rule", "state", "cite", "source", "user"}
        kept = []
        for g in p.domain_guidance:
            toks = set(tokenize(g)) - generic
            if g.strip() and (not toks or len(toks & vocab) / len(toks) >= 0.35):
                kept.append(g)
        p.domain_guidance = kept[:6]
        p.style_rules = [s for s in p.style_rules if s.strip()][:4]
        if len(p.role_statement) < 15:
            raise LLMError("persona too short")
        return p, "llm"
    except FALLBACK_ERRORS:
        return _fallback_persona(req), "fallback"


def render_system_prompt(llm: LLM, req: Requirements, manifest: EvidenceManifest, persona: DraftPersona, tools: list[str]) -> str:
    sources = []
    for f in manifest.files:
        if f.kind == "csv":
            sources.append(f"{f.filename}: table, {f.n_rows} rows, columns {', '.join(c.name for c in f.columns)}")
        elif f.kind == "pdf":
            sources.append(f"{f.filename}: {f.n_pages}-page document - {f.title}")
        else:
            sources.append(f"{f.filename}: text document")
    tool_lines = [f"{t}: {REGISTRY[t].description}" for t in tools]
    must_answer = req.grounded_topics + [f"{t} (documents cover this only partially - state what is missing)" for t in req.partially_grounded_topics]
    return load_prompt(
        "system_prompt_template", llm.s,
        role_statement=persona.role_statement.strip(),
        scope=req.scope.strip(),
        audience=req.audience,
        tone=req.tone,
        must_answer=_bullets(must_answer or req.must_answer_topics),
        must_refuse=_bullets(req.must_refuse_topics),
        gaps=_bullets([f"{g.requirement}: {g.reason}" for g in req.ungrounded_requirements]),
        sources=_bullets(sources),
        tools=_bullets(tool_lines, "- No tools. Answer from the document passages only."),
        guidance=_bullets(persona.domain_guidance),
        style=_bullets(persona.style_rules),
    ).strip()


def generate_agent(llm: LLM, req: Requirements, manifest: EvidenceManifest, docs_folder: str | Path) -> tuple[AgentDefinition, dict]:
    persona, how = build_persona(llm, req, manifest)
    tools, why = select_tools(req, manifest)
    rcfg, rnotes = retrieval_params(manifest)
    contract = OutputContract(citation_format=_citation_format(manifest))
    if req.ungrounded_requirements:
        contract.refusal_policy += " Known gaps: " + "; ".join(g.requirement for g in req.ungrounded_requirements) + "."
    agent = AgentDefinition(
        name=req.agent_name,
        version=0,
        description=req.scope[:300],
        system_prompt=render_system_prompt(llm, req, manifest, persona, tools),
        retrieval_config=rcfg,
        selected_tools=tools,
        tool_rationale=why,
        output_contract=contract,
        docs_folder=str(docs_folder),
        model=llm.s.model,
        must_refuse_topics=req.must_refuse_topics,
    )
    return agent, {"persona_source": how, "retrieval_rationale": rnotes, "persona": persona.model_dump()}
