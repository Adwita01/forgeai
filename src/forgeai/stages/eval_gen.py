"""Stage 4 - Grounded eval generator.

Cases are generated FROM the documents, then verified deterministically:
  * single_doc: one informative passage -> question + key facts; every key fact must be
    found in the source passage, else it is dropped (and the case if none survive);
  * table/arithmetic cases: built deterministically from CSV rows (exact expected numbers);
  * multi_doc: two passages from different files with the highest term overlap;
  * out_of_scope: must-refuse topics, groundability gaps and near-miss traps;
  * edge: rewrites of verified single-doc cases (vague / typo / indirect / negation).
"""
from __future__ import annotations

import random
import re
from typing import Callable, Optional

from ..llm import LLM, LLMError, load_prompt, split_prompt
from ..parsing import Chunk, LoadedDoc
from ..retriever import tokenize
from ..schemas import EvalCase, EvalCategory, EvalSet, EvidenceManifest, GenQA, GenQuestions, Requirements
from ..textutil import fact_supported, norm, numbers
from ..tools.calculator import evaluate

MIN_CASES = 22
EDGE_STYLES = ["vague", "typo", "indirect", "negation", "vague"]


def _informative(c: Chunk) -> float:
    words = len(c.text.split())
    if c.kind != "csv" and words < 35:
        return 0.0
    nums = len(numbers(c.text))
    caps = len(re.findall(r"\b[A-Z][a-z]{3,}", c.text))
    return min(words, 200) / 200 + 0.15 * min(nums, 6) + 0.02 * min(caps, 10)


def _pick_spread(chunks: list[Chunk], n: int, rng: random.Random) -> list[Chunk]:
    """Round-robin over files so every document gets tested, best passages first."""
    by_file: dict[str, list[Chunk]] = {}
    for c in sorted(chunks, key=_informative, reverse=True):
        if _informative(c) > 0:
            by_file.setdefault(c.filename, []).append(c)
    for lst in by_file.values():  # mild shuffle among the top passages to avoid always page 1
        head = lst[: max(4, len(lst) // 2)]
        rng.shuffle(head)
        lst[: len(head)] = head
    out, i = [], 0
    files = sorted(by_file)
    while len(out) < n and any(by_file.values()):
        f = files[i % len(files)]
        if by_file[f]:
            out.append(by_file[f].pop(0))
        i += 1
    return out


def _verified_facts(facts: list[str], source: str) -> list[str]:
    return [f.strip() for f in facts if f.strip() and fact_supported(f, source)][:4]


class EvalGenerator:
    def __init__(self, llm: LLM, req: Requirements, manifest: EvidenceManifest, docs: list[LoadedDoc], chunks: list[Chunk],
                 seed: int = 7, progress: Optional[Callable[[str], None]] = None):
        self.llm, self.req, self.manifest, self.docs, self.chunks = llm, req, manifest, docs, chunks
        self.rng = random.Random(seed)
        self.cases: list[EvalCase] = []
        self.notes: list[str] = []
        self.progress = progress or (lambda m: None)
        self._seen: set[str] = set()
        self._used_chunks: set[str] = set()

    # ------------------------------------------------------------------ helpers
    def _add(self, case: EvalCase) -> bool:
        key = norm(case.question)
        if len(key) < 8 or key in self._seen:
            return False
        self._seen.add(key)
        self.cases.append(case)
        return True

    def _id(self, prefix: str) -> str:
        return f"{prefix}-{sum(1 for c in self.cases if c.id.startswith(prefix)) + 1:02d}"

    def _gen_single(self, c: Chunk) -> Optional[EvalCase]:
        sys_t, user_t = split_prompt(load_prompt("eval_gen_single", self.llm.s, audience=self.req.audience,
                                                 label=f"[{c.filename} {c.locator}]", passage=c.text[:1800]))
        try:
            g = self.llm.structured(GenQA, sys_t, user_t, stage="eval_gen", temperature=0.4, max_tokens=500)
        except LLMError:
            return None
        facts = _verified_facts(g.key_facts, c.text)
        if not facts and g.answer and fact_supported(g.answer, c.text):
            facts = [g.answer]
        if not facts or not g.question.strip():
            self.notes.append(f"discarded single-doc case from {c.chunk_id}: key facts not found in source")
            return None
        return EvalCase(id="", category=EvalCategory.single_doc, question=g.question.strip(), expected_behavior="answer",
                        expected_sources=[c.filename], key_facts=facts, rationale=f"generated from {c.filename} {c.locator}",
                        generated_from=[c.chunk_id])

    # ------------------------------------------------------------------ categories
    def single_doc(self, n: int) -> None:
        pool = [c for c in _pick_spread(self.chunks, n * 3, self.rng) if c.chunk_id not in self._used_chunks]
        made = 0
        for c in pool:
            if made >= n:
                break
            self.progress(f"eval_gen single_doc {made + 1}/{n}")
            case = self._gen_single(c)
            if case:
                case.id = self._id("S")
                if self._add(case):
                    self._used_chunks.add(c.chunk_id)
                    made += 1

    def table_cases(self, n: int) -> None:
        """Deterministic CSV cases: exact lookups, and (if arithmetic is required) computed totals."""
        for d in self.docs:
            if d.kind != "csv" or not d.table:
                continue
            fe = self.manifest.file(d.filename)
            cols = fe.columns if fe else []
            id_col = next((c.name for c in cols if c.dtype == "text" and c.n_unique == len(d.table)), None)
            id_col = id_col or next((c.name for c in cols if c.n_unique == len(d.table)), None)
            num_cols = [c.name for c in cols if c.dtype in ("number", "integer") and c.name != id_col]
            if not id_col or not num_cols:
                continue
            price = next((c for c in num_cols if re.search(r"price|cost|rate|usd|eur|amount|value", c, re.I)), None)
            rows = list(enumerate(d.table, start=1))
            self.rng.shuffle(rows)
            for i, (rno, row) in enumerate(rows[:n]):
                col = num_cols[i % len(num_cols)]
                pretty = col.replace("_", " ")
                q = f"What is the {pretty} for {row[id_col]}?"
                self._add(EvalCase(id=self._id("T"), category=EvalCategory.single_doc, question=q, expected_behavior="answer",
                                   expected_sources=[d.filename], key_facts=[f"{row[col]}"], needs_tool="csv_lookup",
                                   rationale=f"deterministic lookup of {d.filename} row {rno}, column {col}",
                                   generated_from=[f"{d.filename}::row {rno}"]))
            if price and self.req.capabilities.arithmetic and rows:
                rno, row = rows[-1]
                qty = self.rng.choice([3, 4, 6, 12])
                try:
                    total = evaluate(f"{qty} * {row[price]}")
                    self._add(EvalCase(id=self._id("T"), category=EvalCategory.single_doc,
                                       question=f"How much would {qty} units of {row[id_col]} cost in total, based on the listed {price.replace('_', ' ')}?",
                                       expected_behavior="answer", expected_sources=[d.filename],
                                       key_facts=[f"{total}"], needs_tool="calculator",
                                       rationale=f"deterministic: {qty} x {row[price]} from {d.filename} row {rno}",
                                       generated_from=[f"{d.filename}::row {rno}"]))
                except Exception:  # noqa: BLE001
                    pass

    def multi_doc(self, n: int) -> None:
        prose = [c for c in self.chunks if _informative(c) > 0]
        files = sorted({c.filename for c in prose})
        toks = {c.chunk_id: set(tokenize(c.text)) for c in prose}
        pairs = []
        for i, a in enumerate(prose):
            for b in prose[i + 1:]:
                if a.filename == b.filename and len(files) > 1:
                    continue
                if a.filename == b.filename and a.locator == b.locator:
                    continue
                j = len(toks[a.chunk_id] & toks[b.chunk_id]) / max(1, len(toks[a.chunk_id] | toks[b.chunk_id]))
                pairs.append((j, a, b))
        pairs.sort(key=lambda x: -x[0])
        used: set[str] = set()
        made = 0
        for _, a, b in pairs:
            if made >= n:
                break
            if a.chunk_id in used or b.chunk_id in used:
                continue
            self.progress(f"eval_gen multi_doc {made + 1}/{n}")
            sys_t, user_t = split_prompt(load_prompt("eval_gen_multi", self.llm.s, audience=self.req.audience,
                                                     label_a=f"[{a.filename} {a.locator}]", passage_a=a.text[:1400],
                                                     label_b=f"[{b.filename} {b.locator}]", passage_b=b.text[:1400]))
            try:
                g = self.llm.structured(GenQA, sys_t, user_t, stage="eval_gen", temperature=0.4, max_tokens=600)
            except LLMError:
                continue
            fa = _verified_facts(g.key_facts, a.text)
            fb = _verified_facts(g.key_facts, b.text)
            if not fa or not fb:
                self.notes.append(f"discarded multi-doc case {a.chunk_id} + {b.chunk_id}: facts not traceable to both passages")
                used.update({a.chunk_id, b.chunk_id})
                continue
            if self._add(EvalCase(id=self._id("M"), category=EvalCategory.multi_doc, question=g.question.strip(), expected_behavior="answer",
                                  expected_sources=list(dict.fromkeys([a.filename, b.filename])), key_facts=list(dict.fromkeys(fa + fb))[:4],
                                  rationale=f"synthesis of {a.filename} {a.locator} and {b.filename} {b.locator}",
                                  generated_from=[a.chunk_id, b.chunk_id])):
                made += 1
            used.update({a.chunk_id, b.chunk_id})

    def out_of_scope(self, n: int) -> None:
        gaps = [g.requirement for g in self.req.ungrounded_requirements]
        # every groundability gap becomes a refusal test - deterministic, so the guard is always exercised
        for g in gaps[: max(1, n // 2)]:
            self._add(EvalCase(id=self._id("R"), category=EvalCategory.out_of_scope,
                               question=f"Can you tell me about {g}?", expected_behavior="refuse",
                               rationale=f"groundability gap: '{g}' is not supported by the documents"))
        remaining = n - sum(1 for c in self.cases if c.category == EvalCategory.out_of_scope)
        if remaining <= 0:
            return
        self.progress("eval_gen out_of_scope")
        covered = ", ".join((self.req.grounded_topics or self.req.must_answer_topics)[:10])
        sys_t, user_t = split_prompt(load_prompt("eval_gen_refusal", self.llm.s, n=str(remaining + 2), scope=self.req.scope,
                                                 refuse_topics=", ".join(self.req.must_refuse_topics) or "none",
                                                 gaps=", ".join(gaps) or "none", covered=covered))
        qs: list[str] = []
        try:
            qs = self.llm.structured(GenQuestions, sys_t, user_t, stage="eval_gen", temperature=0.6, max_tokens=700).questions
        except LLMError:
            pass
        if not qs:
            qs = [f"What does the guidance say about {t}?" for t in self.req.must_refuse_topics]
            self.notes.append("refusal questions fell back to templates from must-refuse topics")
        for q in qs:
            if remaining <= 0:
                break
            if self._add(EvalCase(id=self._id("R"), category=EvalCategory.out_of_scope, question=q.strip(), expected_behavior="refuse",
                                  rationale="outside the documented scope / must-refuse topic")):
                remaining -= 1

    def edge(self, n: int) -> None:
        bases = [c for c in self.cases if c.category in (EvalCategory.single_doc, EvalCategory.multi_doc) and c.needs_tool is None]
        self.rng.shuffle(bases)
        made = 0
        for i, base in enumerate(bases):
            if made >= n:
                break
            style = EDGE_STYLES[made % len(EDGE_STYLES)]
            self.progress(f"eval_gen edge {made + 1}/{n} ({style})")
            sys_t, user_t = split_prompt(load_prompt("eval_gen_edge", self.llm.s, style=style, question=base.question, facts="; ".join(base.key_facts)))
            try:
                g = self.llm.structured(GenQA, sys_t, user_t, stage="eval_gen", temperature=0.7, max_tokens=250)
            except LLMError:
                continue
            q = g.question.strip()
            if not q or norm(q) == norm(base.question):
                continue
            if self._add(EvalCase(id=self._id("E"), category=EvalCategory.edge, question=q, expected_behavior="answer",
                                  expected_sources=base.expected_sources, key_facts=base.key_facts,
                                  rationale=f"{style} rewrite of {base.id}", generated_from=base.generated_from)):
                made += 1

    # ------------------------------------------------------------------ driver
    def generate(self, n_single: int = 8, n_table: int = 2, n_multi: int = 5, n_refuse: int = 5, n_edge: int = 5,
                 max_cases: Optional[int] = None) -> EvalSet:
        has_csv = any(d.kind == "csv" for d in self.docs)
        if max_cases:  # quick/verification mode: keep every category represented, skip the top-up
            k = max(4, max_cases)
            n_refuse = n_multi = n_edge = max(1, k // 5)
            n_answerable = max(1, k - n_refuse - n_multi - n_edge)
            n_table = 1 if has_csv and n_answerable >= 2 else 0
            self.single_doc(n_answerable - n_table)
            if has_csv:
                self.table_cases(n_table)
            self.multi_doc(n_multi)
            self.out_of_scope(n_refuse)
            self.edge(n_edge)
            self.notes.append(f"quick mode: capped at {max_cases} cases (below the {MIN_CASES}-case target by request)")
            return EvalSet(cases=self.cases[:max_cases] if len(self.cases) > max_cases else self.cases,
                           generator_model=self.llm.s.model, notes=list(dict.fromkeys(self.notes)))
        self.single_doc(n_single if not has_csv else max(4, n_single - n_table))
        if has_csv:
            self.table_cases(n_table)
        self.multi_doc(n_multi)
        self.out_of_scope(n_refuse)
        self.edge(n_edge)
        # top-up: never ship fewer than MIN_CASES if the documents allow it
        attempts = 0
        while len(self.cases) < MIN_CASES and attempts < 3:
            self.single_doc(MIN_CASES - len(self.cases))
            attempts += 1
        if len(self.cases) < MIN_CASES:
            self.notes.append(f"only {len(self.cases)} verifiable cases could be generated (target {MIN_CASES}) - corpus too small or model output unverifiable")
        return EvalSet(cases=self.cases, generator_model=self.llm.s.model, notes=list(dict.fromkeys(self.notes)))
