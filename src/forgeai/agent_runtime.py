"""Generic runtime: executes ANY agent_definition.json. Used by `forge run`, the UI, and the evaluator."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

from .llm import LLM, LLMError, load_prompt, split_prompt
from .parsing import LoadedDoc, load_folder
from .retriever import HybridRetriever
from .schemas import AgentAnswerDraft, AgentDefinition, AgentResponse, RetrievedChunk, ToolPlan, ToolResult
from .tools import REGISTRY, ToolContext, describe, run_tool

_REFUSAL_HINTS = re.compile(
    r"\b(i can(?:no|')t answer|cannot answer|unable to (?:answer|find|provide)|not (?:covered|addressed|mentioned|available|specified) in (?:the|my) (?:documents|context|sources)|"
    r"(?:documents|context) do(?:es)? not (?:contain|include|mention|specify|cover)|outside (?:my|the) scope|no information (?:about|on|regarding))",
    re.I,
)
_CITE = re.compile(r"\[([^\[\]]+?\.(?:pdf|csv|txt|md|markdown))\s*(?:,|\s)\s*((?:p\.?|page|row|§)\s*[^\]]+)\]", re.I)


def chunk_label(c: RetrievedChunk) -> str:
    return f"[{c.filename} {c.locator}]"


def parse_citations(text: str) -> list[str]:
    out = []
    for f, loc in _CITE.findall(text or ""):
        loc = loc.strip()
        loc = re.sub(r"^page\s*", "p.", loc, flags=re.I)
        loc = re.sub(r"^p\s*\.?\s*", "p.", loc, flags=re.I)
        out.append(f"[{f.strip()} {loc}]")
    return list(dict.fromkeys(out))


def normalize_citation(c: str) -> str:
    c = c.strip()
    if not c.startswith("["):
        c = f"[{c}]"
    parsed = parse_citations(c)
    return parsed[0] if parsed else c


class Agent:
    def __init__(self, definition: AgentDefinition, llm: LLM, docs: Optional[list[LoadedDoc]] = None, base_dir: Optional[Path] = None):
        self.d = definition
        self.llm = llm
        folder = Path(definition.docs_folder)
        if not folder.is_absolute() and base_dir is not None:
            folder = (base_dir / folder).resolve()
        if docs is None:
            docs, _ = load_folder(folder)
        self.docs = docs
        self.retriever = HybridRetriever(docs, definition.retrieval_config, llm, cache_dir=llm.s.cache_dir / "emb")
        self.tool_ctx = ToolContext(tables={d.filename: (d.table or []) for d in docs if d.kind == "csv"})
        self.model = definition.model or llm.s.model
        self.model_note = ""
        if self.model != llm.s.model:
            # a definition built elsewhere may pin a model this server doesn't have -> use the configured one
            avail = llm.health().get("models") or []
            if avail and not any(m == self.model or m.split(":")[0] == self.model for m in avail):
                self.model_note = f"model {self.model!r} not available; using {llm.s.model!r}"
                self.model = llm.s.model

    @classmethod
    def from_file(cls, path: str | Path, llm: LLM) -> "Agent":
        p = Path(path)
        d = AgentDefinition.model_validate_json(p.read_text(encoding="utf-8"))
        return cls(d, llm, base_dir=p.parent)

    # ------------------------------------------------------------------ tools
    def _plan_tools(self, question: str, retrieved: list[RetrievedChunk], prior: list[ToolResult]) -> list[ToolResult]:
        callable_tools = [t for t in self.d.selected_tools if t != "current_date"]
        if not callable_tools:
            return []
        ctx_preview = "\n".join(f"{chunk_label(c)} {c.text[:300]}" for c in retrieved[:4])
        sys_t, user_t = split_prompt(
            load_prompt(
                "agent_tool_plan", self.llm.s,
                tools=describe(callable_tools), question=question, context=ctx_preview,
                prior=json.dumps([r.model_dump() for r in prior], default=str)[:2500] if prior else "none",
            )
        )
        try:
            plan = self.llm.structured(ToolPlan, sys_t, user_t, stage="agent_tool_plan", model=self.model, max_tokens=400)
        except LLMError:
            return []
        results = []
        for call in plan.calls[:3]:
            if call.tool not in callable_tools:  # hard guard: agent may only use attached tools
                continue
            args = self._sanitise_call(call.tool, dict(call.args or {}), question, retrieved, prior)
            if args is None:
                continue
            call.args = args
            out, err = run_tool(call.tool, call.args, self.tool_ctx)
            results.append(ToolResult(tool=call.tool, args=call.args, output=out, error=err))
        return results

    _ARITH = re.compile(r"\b(total|cost|costs|sum|how much|multipl\w*|times|difference|percent\w*|average|combined|"
                        r"surcharge|price for|order of|value of)\b|%|\d+\s*(?:units?|pcs|pieces|x)\b", re.I)

    def _sanitise_call(self, tool: str, args: dict, question: str, retrieved: list[RetrievedChunk],
                       prior: list[ToolResult]) -> Optional[dict]:
        """Deterministic guard rails around the small model's tool plan (returns None to drop the call)."""
        if tool == "csv_lookup":
            tables = list(self.tool_ctx.tables)
            f = str(args.get("file") or args.get("filename") or "")
            if f and f not in self.tool_ctx.tables:
                if f.lower().endswith((".pdf", ".txt", ".md")) or " p." in f or len(tables) != 1:
                    return None  # lookups on PDFs or invented tables are noise
                args["file"] = tables[0]
            return args
        if tool == "calculator":
            if not self._ARITH.search(question) and not prior:
                return None  # no arithmetic in the question: a calculator call can only add noise
            expr = str(args.get("expression", ""))
            names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", expr)) - {"max", "min", "round", "abs", "sum"}
            if names:  # e.g. "4 * unit_price_usd": fill in values from the one catalogue row the question names
                vals = self._row_values(question, retrieved, prior)
                if all(n in vals for n in names):
                    for n in sorted(names, key=len, reverse=True):
                        expr = re.sub(rf"\b{re.escape(n)}\b", vals[n], expr)
                    args["expression"] = expr
            return args
        return args

    @staticmethod
    def _row_values(question: str, retrieved: list[RetrievedChunk], prior: list[ToolResult]) -> dict[str, str]:
        """Numeric fields of the single CSV row whose identifier the question mentions."""
        qlow = question.lower()
        rows = []
        for c in retrieved:
            if not c.locator.startswith("row"):
                continue
            line = next((ln for ln in c.text.splitlines() if re.match(r"\s*row\s+\d+\s*:", ln)), "")
            body = line.split(":", 1)[1] if line else ""
            fields = {k.strip(): v.strip() for k, v in (part.split(":", 1) for part in body.split(";") if ":" in part)}
            ident = next((v for v in fields.values() if re.search(r"[A-Za-z]+-[A-Za-z]*-?\d+", v)), "")
            if ident and ident.strip().lower() in qlow:
                rows.append(fields)
        if len(rows) != 1:
            return {}
        out = {}
        for k, v in rows[0].items():
            num = re.sub(r"[^\d.\-]", "", v)
            if num and re.fullmatch(r"-?\d+(?:\.\d+)?", num):
                out[k.strip()] = num
        return out

    def _tools(self, question: str, retrieved: list[RetrievedChunk]) -> list[ToolResult]:
        results: list[ToolResult] = []
        if "current_date" in self.d.selected_tools:
            out, err = run_tool("current_date", {}, self.tool_ctx)
            results.append(ToolResult(tool="current_date", args={}, output=out, error=err))
        first = self._plan_tools(question, retrieved, [])
        results += first
        # one follow-up round so lookups can feed the calculator
        if first and "calculator" in self.d.selected_tools and not any(r.tool == "calculator" for r in first):
            results += [r for r in self._plan_tools(question, retrieved, first) if r.tool == "calculator"]
        return results

    # ------------------------------------------------------------------ answer
    def answer(self, question: str, history: Optional[list[dict]] = None) -> AgentResponse:
        t0 = time.monotonic()
        query = question
        if history:  # light conversational context for follow-ups in the chat runner
            last_user = [h["content"] for h in history if h.get("role") == "user"][-1:]
            if last_user and len(question.split()) < 8:
                query = f"{last_user[0]} {question}"
        retrieved = self.retriever.search(query)
        tool_results = self._tools(question, retrieved) if self.d.selected_tools else []

        context = "\n\n".join(f"{chunk_label(c)}\n{c.text}" for c in retrieved) or "(no relevant passages found)"
        tools_txt = (
            "\n".join(f"- {r.tool}({json.dumps(r.args, default=str)}) -> {json.dumps(r.output, default=str)[:1500] if r.error is None else 'ERROR ' + r.error}" for r in tool_results)
            or "none"
        )
        c = self.d.output_contract
        sys_t, user_t = split_prompt(
            load_prompt(
                "agent_answer", self.llm.s,
                system_prompt=self.d.full_system_prompt(),
                citation_format=c.citation_format,
                refusal_policy=c.refusal_policy,
                refusal_message=c.refusal_message,
                max_words=str(c.max_answer_words),
                context=context,
                tool_results=tools_txt,
                question=question,
            )
        )
        try:
            draft = self.llm.structured(AgentAnswerDraft, sys_t, user_t, stage="agent_answer", model=self.model, max_tokens=700)
            answer, refused = draft.answer.strip(), bool(draft.refused)
            cites = [normalize_citation(x) for x in draft.citations] + parse_citations(answer)
        except LLMError as e:
            answer, refused, cites = f"{c.refusal_message} (internal error: {e})", True, []
        if not refused and _REFUSAL_HINTS.search(answer) and len(answer.split()) < 60:
            refused = True
        if refused:
            cites = []
        return AgentResponse(
            question=question,
            answer=answer,
            citations=list(dict.fromkeys(x for x in cites if x and x != "[]")),
            refused=refused,
            retrieved=retrieved,
            tool_results=tool_results,
            latency_s=round(time.monotonic() - t0, 2),
        )


__all__ = ["Agent", "parse_citations", "normalize_citation", "chunk_label", "REGISTRY"]
