"""Shared fixtures: a tiny mixed-format corpus and a scripted fake LLM (no network)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from forgeai.budget import Budget
from forgeai.config import Settings
from forgeai.llm import LLM

POLICY_P1 = (
    "1. Annual Leave\nShore staff are entitled to 24 working days of annual leave per year. "
    "Leave requests must be submitted at least 14 days in advance through the HR portal. "
    "Unused leave of up to 5 days may be carried over to the next year."
)
POLICY_P2 = (
    "2. Shore Leave for Crew\nCrew members may take shore leave when the vessel is alongside for more than 8 hours, "
    "subject to Master approval. Crew must return on board at least 2 hours before scheduled departure."
)


def make_pdf(path: Path, pages: list[str]) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    for text in pages:
        y = 800
        for para in text.split("\n"):
            words, line = para.split(), ""
            for wd in words:
                if len(line) + len(wd) > 90:
                    c.drawString(50, y, line)
                    y -= 16
                    line = ""
                line += wd + " "
            c.drawString(50, y, line)
            y -= 24
        c.showPage()
    c.save()


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    make_pdf(d / "hr_policy.pdf", [POLICY_P1, POLICY_P2])
    (d / "parts.csv").write_text(
        "part_no,description,unit_price_usd,stock_qty\nFP-1001,Fuel pump seal kit,45.50,12\nAV-2002,Air valve assembly,310.00,3\n"
        "GS-3003,Gasket set,18.25,40\n",
        encoding="utf-8",
    )
    (d / "notes.md").write_text("# Procurement\nUrgent orders are shipped within 48 hours.\n", encoding="utf-8")
    return d


def fake_settings(tmp_path: Path) -> Settings:
    return Settings(base_url="http://fake.local/v1", api_key="x", model="fake", judge_model="fake", embed_model="",
                    cache_dir=tmp_path / "cache", max_retries=3)


class FakeLLM(LLM):
    """Answers every stage with plausible, schema-valid JSON derived from the prompt itself."""

    def __init__(self, tmp_path: Path, script: dict | None = None):
        super().__init__(fake_settings(tmp_path), Budget(), use_cache=False)
        self.script = script or {}
        self.calls: list[str] = []

    def embed(self, texts, batch=32):
        return None

    def health(self):
        return {"reachable": True, "model_available": True, "models": ["fake"]}

    def chat(self, messages, *, stage="misc", **kw):
        self.calls.append(stage)
        self.budget.check()  # like the real client: uncached calls are budget-gated
        self.budget.record(stage, 100, 50, 0.01, retry=kw.get("retry", False))
        if stage in self.script:
            v = self.script[stage]
            return v(messages) if callable(v) else v
        sys, user = messages[0]["content"], messages[-1]["content"] if len(messages) == 2 else messages[1]["content"]
        return json.dumps(self._respond(stage, sys, user))

    def _respond(self, stage, sys, user):
        if stage == "interviewer" and "clarifying questions" in sys:
            return {"questions": [{"question": "Who is the audience?", "why": "tone", "default_answer": "shore staff"}]}
        if stage == "interviewer":
            return {"agent_name": "HR Helper", "scope": "Answer HR leave questions", "audience": "shore staff", "tone": "friendly",
                    "must_answer_topics": ["annual leave", "shore leave", "pension scheme"],
                    "must_refuse_topics": ["legal advice"],
                    "capabilities": {"tabular_lookup": False, "arithmetic": False, "date_awareness": False, "live_weather": True}}
        if stage == "groundability":
            topic = re.search(r"TOPIC: (.*)", user).group(1).strip()
            if "pension" in topic:
                return {"topic": topic, "verdict": "unsupported", "evidence_quote": "", "reason": "not mentioned"}
            body = user.split("EXCERPTS:", 1)[1]
            text = re.sub(r"\[[^\]]+\]", "", body).strip()
            return {"topic": topic, "verdict": "supported", "evidence_quote": text[:40], "reason": "ok"}
        if stage == "generator":
            return {"role_statement": "You are HR Helper for shore staff.", "domain_guidance": ["Quote exact day counts."], "style_rules": ["Be brief."]}
        if stage == "eval_gen":
            if "MUST refuse" in sys:
                return {"questions": ["What is the company pension contribution rate?", "Can you give me legal advice on my contract?",
                                      "What is the weather in Rotterdam?", "Who won the football match?", "What is the dress code?"]}
            if "EDGE CASE" in sys:
                q = re.search(r"ORIGINAL QUESTION: (.*)", user).group(1)
                return {"question": "hey quick one - " + q.lower()}
            passages = re.findall(r"PASSAGE[^\n]*\]:\n(.*?)(?:\n\n|\Z)", user, re.S)
            facts = [max(p.split(". "), key=len).split(": ")[-1][:120] for p in passages]
            prefix = "How do these relate:" if "BOTH passages" in sys else "What does the policy say about"
            return {"question": f"{prefix} {' / '.join(f[:40] for f in facts)}?", "key_facts": facts, "answer": facts[0]}
        if stage == "agent_tool_plan":
            return {"calls": []}
        if stage == "agent_answer":
            q = user.split("QUESTION", 1)[1]
            if re.search(r"pension|legal|weather|football|dress", q, re.I):
                return {"answer": "I can't answer that from the documents I have access to.", "citations": [], "refused": True}
            m = re.search(r"CONTEXT\n(\[[^\]]+\])\n(.*?)(?:\n\n|\nTOOL RESULTS)", user, re.S)
            label, text = m.group(1), m.group(2)
            return {"answer": f"{text[:400]} {label}", "citations": [label], "refused": False}
        if stage == "judge":
            ans = user.split("<<<", 1)[1].split(">>>", 1)[0].strip()
            facts = re.findall(r"^- (.*)$", user.split("EXPECTED FACTS:", 1)[1], re.M)
            return {"claims": [{"claim": ans[:200], "quote": ans[:120]}],
                    "facts": [{"fact": f, "present": f[:20].lower() in ans.lower(), "answer_quote": f[:40]} for f in facts]}
        if stage == "optimizer":
            return {"rules": ["Answer using the wording of the passage and always include the citation label for every fact you state."]}
        return {}


@pytest.fixture()
def fake_llm(tmp_path):
    return FakeLLM(tmp_path)
