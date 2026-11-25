"""Stage 1 - Intake interviewer: brief + manifest -> <=5 clarifying questions -> requirements."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Literal, Optional

from ..llm import LLM, LLMError, load_prompt, split_prompt
from ..manifest import manifest_for_prompt
from ..schemas import QA, ClarifyingQuestion, ClarifyingQuestions, EvidenceManifest, Requirements

Mode = Literal["interactive", "auto"]

_FALLBACK_QUESTIONS = [
    ClarifyingQuestion(question="Who exactly will use this assistant, and what tone should it take?",
                       why="audience and tone", default_answer="Staff named in the brief; professional, concise and plain-language."),
    ClarifyingQuestion(question="Which topics must the assistant refuse to answer?",
                       why="refusal scope", default_answer="Anything not covered by the documents, personal/individual case decisions, legal or medical advice."),
    ClarifyingQuestion(question="Will users need calculations, table lookups or date-dependent answers?",
                       why="tool selection", default_answer="Only if the brief explicitly mentions them."),
]


def ask_questions(llm: LLM, brief: str, manifest: EvidenceManifest) -> tuple[list[ClarifyingQuestion], str]:
    sys_t, user_t = split_prompt(load_prompt("interviewer_questions", llm.s, brief=brief, manifest=manifest_for_prompt(manifest)))
    try:
        qs = llm.structured(ClarifyingQuestions, sys_t, user_t, stage="interviewer", max_tokens=900).questions
        qs = [q for q in qs if q.question.strip()][:5]
        if qs:
            return qs, "llm"
    except LLMError:
        pass
    return list(_FALLBACK_QUESTIONS), "fallback"


def _lookup_answer(q: ClarifyingQuestion, idx: int, answers: object) -> Optional[str]:
    if isinstance(answers, list) and idx < len(answers) and str(answers[idx]).strip():
        return str(answers[idx])
    if isinstance(answers, dict):
        if str(idx + 1) in answers:
            return str(answers[str(idx + 1)])
        ql = q.question.lower()
        for k, v in answers.items():  # keyword match: {"audience": "...", "refuse": "..."}
            if k.lower() in ql:
                return str(v)
        if "default" in answers:
            return str(answers["default"])
    return None


def collect_answers(
    questions: list[ClarifyingQuestion],
    mode: Mode = "auto",
    answers: object = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> list[QA]:
    out: list[QA] = []
    for i, q in enumerate(questions):
        pre = _lookup_answer(q, i, answers)
        if mode == "interactive":
            print_fn(f"\n[Q{i + 1}/{len(questions)}] {q.question}")
            if q.why:
                print_fn(f"   (why: {q.why})")
            dflt = pre or q.default_answer
            try:
                a = input_fn(f"   answer [Enter = {dflt!r}]: ").strip()
            except EOFError:
                a = ""
            out.append(QA(question=q.question, answer=a or dflt, source="user" if a else ("answers_file" if pre else "auto_default")))
        else:
            out.append(QA(question=q.question, answer=pre or q.default_answer or "No preference.", source="answers_file" if pre else "auto_default"))
    return out


def _fallback_requirements(brief: str, manifest: EvidenceManifest) -> Requirements:
    topics = []
    for f in manifest.files:
        topics.extend(f.sections[:4] if f.sections else [f.title])
    return Requirements(
        agent_name="Document Assistant",
        scope=brief.strip()[:400],
        must_answer_topics=[t for t in dict.fromkeys(topics) if t][:8] or ["questions answered by the provided documents"],
        must_refuse_topics=["topics not covered by the documents", "legal, medical or financial advice", "personal case decisions"],
        assumptions=["requirements LLM failed; derived deterministically from brief and manifest"],
    )


_KW = {
    "arithmetic": r"\b(calculat\w*|comput\w*|total\w*|sum|how much (?:would|will|does)|cost of|budget|multiply|quote for|price for \d+|estimate)\b",
    "date_awareness": r"\b(today|current date|deadline\w*|expir\w*|due date|how many days|days (?:until|left|remaining)|overdue|this (?:week|month|year))\b",
    "live_weather": r"\b(weather|forecast|wind speed|sea state|temperature at|storm)\b",
    "tabular_lookup": r"\b(lookup|look up|inventory|stock|catalogu?e|part (?:number|no)|sku|price list|table|spreadsheet|csv|quantity on hand)\b",
}


def reconcile_capabilities(req: Requirements, brief: str, qa: list[QA], manifest: EvidenceManifest) -> Requirements:
    """Deterministic gate on top of the model's capability flags, so tool selection can't drift.

    A capability is enabled only with explicit evidence in the brief / requester answers
    (or, for table lookup, a CSV that actually exists). Model opinion alone is never enough
    for weather or arithmetic: an HR agent with a weather tool is a factory defect.
    """
    # Only the requester's own words count as evidence - model-invented default answers do not.
    text = " ".join([brief] + [x.answer for x in qa if x.source != "auto_default"])
    low = text.lower()
    caps = req.capabilities
    ev: dict[str, str] = {}

    def hit(name: str) -> Optional[str]:
        m = re.search(_KW[name], low)
        return m.group(0) if m else None

    # negative answers like "no calculations needed" override keyword hits
    def negated(word_pat: str) -> bool:
        return bool(re.search(r"\b(no|not|never|without|don't|doesn't|none)\b[^.]{0,40}" + word_pat, low))

    csv_files = [f.filename for f in manifest.files if f.kind == "csv"]
    if csv_files and (caps.tabular_lookup or hit("tabular_lookup")):
        caps.tabular_lookup, ev["tabular_lookup"] = True, f"CSV present ({', '.join(csv_files)}); evidence: {hit('tabular_lookup') or 'model flag'}"
    else:
        caps.tabular_lookup = False
        ev["tabular_lookup"] = "off: no CSV in the documents" if not csv_files else "off: brief never asks for table lookups"

    for name, neg in (("arithmetic", r"(calculat|comput|arithmetic|math)"), ("date_awareness", r"(date|deadline)"), ("live_weather", r"(weather|forecast)")):
        h = hit(name)
        if h and not negated(neg):
            setattr(caps, name, True)
            ev[name] = f"on: brief/answers mention '{h}'"
        elif (name == "arithmetic" and caps.arithmetic and caps.tabular_lookup
              and any(c.dtype in ("number", "integer") for f in manifest.files for c in f.columns)):
            ev[name] = "on: model flagged arithmetic, table lookups are required and the CSV has numeric columns"
        else:
            if getattr(caps, name):
                ev[name] = "off: model flagged it but nothing in the brief or answers requires it"
            else:
                ev[name] = "off: not requested"
            setattr(caps, name, False)
    req.capabilities = caps
    req.capability_evidence = ev
    return req


def draft_requirements(llm: LLM, brief: str, qa: list[QA], manifest: EvidenceManifest) -> tuple[Requirements, str]:
    qa_txt = "\n".join(f"Q: {x.question}\nA: {x.answer}" for x in qa) or "none"
    sys_t, user_t = split_prompt(load_prompt("interviewer_requirements", llm.s, brief=brief, qa=qa_txt, manifest=manifest_for_prompt(manifest)))
    try:
        req, how = llm.structured(Requirements, sys_t, user_t, stage="interviewer", max_tokens=1200), "llm"
    except LLMError:
        req, how = _fallback_requirements(brief, manifest), "fallback"
    if not req.must_answer_topics:
        req.must_answer_topics = _fallback_requirements(brief, manifest).must_answer_topics
    req.must_answer_topics = list(dict.fromkeys(t.strip() for t in req.must_answer_topics if t.strip()))[:10]
    # small models sometimes turn interview meta-questions into "topics" (e.g. "ambiguous terms")
    meta = re.compile(r"^(ambiguous|unclear|vague|none|n/?a|no specific|other)\b", re.I)
    req.must_refuse_topics = list(dict.fromkeys(t.strip() for t in req.must_refuse_topics if t.strip() and not meta.match(t.strip())))[:8]
    req.clarifications = qa
    return reconcile_capabilities(req, brief, qa, manifest), how


def load_answers(path: Optional[str | Path]) -> object:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))
