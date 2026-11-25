"""Stage 2 - Groundability guard: every must-answer topic is checked against the documents.

Two layers:
  * deterministic: retrieve the best passages; if nothing lexically or semantically
    overlaps the topic, it is unsupported without asking the model;
  * LLM verdict on the retrieved excerpts, which must quote its evidence - a "supported"
    verdict whose quote is not actually in the excerpts is downgraded.
Unsupported topics become GapFlags (action=refuse) and are removed from must_answer,
so the generated agent is told about the gap instead of hallucinating over it.
"""
from __future__ import annotations

from ..llm import LLM, LLMError, load_prompt, split_prompt
from ..retriever import HybridRetriever, tokenize
from ..schemas import GapFlag, Requirements, TopicVerdict
from ..textutil import overlap, quote_in


def check_topic(llm: LLM, retriever: HybridRetriever, topic: str) -> tuple[TopicVerdict, str]:
    hits = retriever.search(topic, top_k=4)
    q_tokens = set(tokenize(topic))
    lexical = max((len(q_tokens & set(tokenize(h.text))) / max(1, len(q_tokens)) for h in hits), default=0.0)
    excerpts = "\n\n".join(f"[{h.filename} {h.locator}] {h.text[:700]}" for h in hits)
    best = f"{hits[0].filename} {hits[0].locator}" if hits else ""

    # Without real embeddings there is no synonym matching, so zero word overlap is decisive.
    # With embeddings, let the model judge (e.g. topic "vacation" vs documents saying "leave").
    if not hits or (lexical == 0.0 and not retriever.dense_backend.startswith("embed")):
        return TopicVerdict(topic=topic, verdict="unsupported", reason="no passage shares any content word with the topic"), best

    sys_t, user_t = split_prompt(load_prompt("groundability", llm.s, topic=topic, excerpts=excerpts))
    try:
        v = llm.structured(TopicVerdict, sys_t, user_t, stage="groundability", max_tokens=400)
        v.topic = topic
    except LLMError:
        verdict = "supported" if lexical >= 0.6 else "partial" if lexical >= 0.3 else "unsupported"
        return TopicVerdict(topic=topic, verdict=verdict, reason=f"LLM unavailable; lexical coverage {lexical:.2f}"), best

    joined = " ".join(h.text for h in hits)
    if v.verdict == "supported" and v.evidence_quote and not quote_in(v.evidence_quote, joined):
        v.verdict = "partial"
        v.reason = (v.reason + " | downgraded: evidence quote not found verbatim in the excerpts").strip(" |")
    if v.verdict == "supported" and not v.evidence_quote and overlap(topic, joined) < 0.5:
        v.verdict = "partial"
        v.reason = (v.reason + " | downgraded: no evidence quote and weak lexical overlap").strip(" |")
    if v.verdict == "unsupported" and lexical >= 0.8:
        v.verdict = "partial"  # the model is too strict sometimes; keep the topic, but caveat it
        v.reason = (v.reason + f" | upgraded to partial: all topic words occur in {best}").strip(" |")
    return v, best


def run_groundability(llm: LLM, retriever: HybridRetriever, req: Requirements) -> tuple[Requirements, list[TopicVerdict]]:
    verdicts: list[TopicVerdict] = []
    grounded, partial, gaps = [], [], []
    for topic in req.must_answer_topics:
        v, best = check_topic(llm, retriever, topic)
        verdicts.append(v)
        if v.verdict == "supported":
            grounded.append(topic)
        elif v.verdict == "partial":
            partial.append(topic)
        else:
            gaps.append(GapFlag(requirement=topic, reason=v.reason or "documents do not address this topic", best_evidence=best, action="refuse"))
    req.grounded_topics = grounded
    req.partially_grounded_topics = partial
    req.ungrounded_requirements = gaps
    req.must_answer_topics = grounded + partial
    for g in gaps:
        label = f"{g.requirement} (not covered by the documents)"
        if label not in req.must_refuse_topics:
            req.must_refuse_topics.append(label)
    return req, verdicts
