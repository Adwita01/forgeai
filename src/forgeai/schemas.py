"""All Pydantic models used across ForgeAI.

Models that an LLM fills in are deliberately *lenient on shape* (a string where a
list was expected is coerced, scores are clamped, enum values are normalised)
but *strict on meaning* (required fields, allowed values). Small models get the
shape wrong far more often than the content.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TOOL_NAMES = ("csv_lookup", "calculator", "open_meteo_weather", "current_date")


def _as_list(v: Any) -> Any:
    if v is None:
        return []
    if isinstance(v, str):
        parts = [p.strip(" -*•\t") for p in v.replace(";", "\n").split("\n")]
        return [p for p in parts if p]
    if isinstance(v, dict):
        return list(v.values())
    return v


def _clamp01(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    if f > 1.0 and f <= 10.0:  # model answered on a 0-10 scale
        f = f / 10.0
    elif f > 10.0:  # model answered on a 0-100 scale
        f = f / 100.0
    return max(0.0, min(1.0, f))


class Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# --------------------------------------------------------------------------- manifest
class ColumnInfo(BaseModel):
    name: str
    dtype: str
    n_unique: int
    n_null: int = 0
    samples: list[str] = []
    min: Optional[float] = None
    max: Optional[float] = None


class PageInfo(BaseModel):
    page: int
    n_chars: int
    headings: list[str] = []
    preview: str = ""


class FileEntry(BaseModel):
    filename: str
    kind: Literal["pdf", "csv", "text"]
    n_pages: int = 0
    n_rows: int = 0
    n_chars: int = 0
    pages: list[PageInfo] = []
    sections: list[str] = []
    columns: list[ColumnInfo] = []
    key_entities: list[str] = []
    numbers_with_units: list[str] = []
    title: str = ""
    warnings: list[str] = []


class EvidenceManifest(BaseModel):
    folder: str
    files: list[FileEntry]
    corpus_key_terms: list[str] = []
    total_chars: int = 0
    warnings: list[str] = []

    def file(self, name: str) -> Optional[FileEntry]:
        low = name.lower().strip()
        for f in self.files:
            if f.filename.lower() == low:
                return f
        return None

    @property
    def has_csv(self) -> bool:
        return any(f.kind == "csv" for f in self.files)


# --------------------------------------------------------------------------- interviewer
class ClarifyingQuestion(Lenient):
    question: str
    why: str = ""
    default_answer: str = Field("", description="Best-guess answer used in unattended mode")


class ClarifyingQuestions(Lenient):
    questions: list[ClarifyingQuestion] = Field(default_factory=list, max_length=5)

    @field_validator("questions", mode="before")
    @classmethod
    def _trim(cls, v):
        v = _as_list(v)
        if isinstance(v, list):
            v = [({"question": q} if isinstance(q, str) else q) for q in v][:5]
        return v


class QA(BaseModel):
    question: str
    answer: str
    source: Literal["user", "auto_default", "answers_file"] = "user"


class Capabilities(Lenient):
    """What the agent needs to *do*, beyond reading documents. Drives tool pruning."""
    tabular_lookup: bool = False
    arithmetic: bool = False
    date_awareness: bool = False
    live_weather: bool = False


class GapFlag(Lenient):
    requirement: str
    reason: str = ""
    best_evidence: str = ""
    action: Literal["refuse", "clarify", "drop"] = "refuse"

    @field_validator("action", mode="before")
    @classmethod
    def _norm(cls, v):
        v = str(v or "refuse").lower().strip()
        return v if v in ("refuse", "clarify", "drop") else "refuse"


class Requirements(Lenient):
    agent_name: str = "Document Assistant"
    scope: str
    audience: str = "general staff"
    tone: str = "professional, concise"
    must_answer_topics: list[str] = []
    must_refuse_topics: list[str] = []
    capabilities: Capabilities = Capabilities()
    assumptions: list[str] = []
    # filled by the pipeline, not the model
    clarifications: list[QA] = []
    ungrounded_requirements: list[GapFlag] = []
    grounded_topics: list[str] = []
    partially_grounded_topics: list[str] = []
    capability_evidence: dict[str, str] = {}
    approved_by: str = "auto"

    @field_validator("must_answer_topics", "must_refuse_topics", "assumptions", mode="before")
    @classmethod
    def _coerce_lists(cls, v):
        return [str(x) for x in _as_list(v)]


class TopicVerdict(Lenient):
    topic: str
    verdict: Literal["supported", "partial", "unsupported"]
    evidence_quote: str = ""
    reason: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _norm(cls, v):
        v = str(v or "").lower().strip()
        if v.startswith("support") or v in ("yes", "true", "grounded"):
            return "supported"
        if v.startswith("partial"):
            return "partial"
        return "unsupported"


# --------------------------------------------------------------------------- agent definition
class RetrievalConfig(BaseModel):
    chunk_size: int = Field(180, ge=40, le=1200, description="words per chunk")
    chunk_overlap: int = Field(30, ge=0, le=400)
    top_k: int = Field(5, ge=1, le=15)
    hybrid_alpha: float = Field(0.5, ge=0.0, le=1.0, description="weight on dense; 1-alpha on BM25")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="drop fused hits below this")

    @model_validator(mode="after")
    def _overlap_lt_size(self):
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = self.chunk_size // 4
        return self


class OutputContract(BaseModel):
    citation_format: str = "[<filename> p.<page>] for PDFs, [<filename> row <n>] for CSV rows, [<filename> §<section>] for text"
    require_citations: bool = True
    refusal_policy: str = (
        "Refuse when the retrieved context does not contain the answer, or when the question is in a must-refuse topic. "
        "Never answer from general knowledge."
    )
    refusal_message: str = "I can't answer that from the documents I have access to."
    max_answer_words: int = 180


class AgentDefinition(BaseModel):
    name: str
    version: int = 0
    description: str = ""
    system_prompt: str
    prompt_rules: list[str] = Field(default_factory=list, description="targeted rules appended by the optimizer")
    retrieval_config: RetrievalConfig = RetrievalConfig()
    selected_tools: list[str] = []
    tool_rationale: dict[str, str] = {}
    output_contract: OutputContract = OutputContract()
    docs_folder: str
    model: str = ""
    must_refuse_topics: list[str] = []

    @field_validator("selected_tools")
    @classmethod
    def _known_tools(cls, v):
        bad = [t for t in v if t not in TOOL_NAMES]
        if bad:
            raise ValueError(f"unknown tools {bad}; allowed: {TOOL_NAMES}")
        return list(dict.fromkeys(v))

    def full_system_prompt(self) -> str:
        if not self.prompt_rules:
            return self.system_prompt
        rules = "\n".join(f"- {r}" for r in self.prompt_rules)
        return f"{self.system_prompt}\n\nADDITIONAL RULES (learned from failed tests):\n{rules}"


class DraftPersona(Lenient):
    """The only part of the agent definition the LLM writes free-form."""
    role_statement: str
    domain_guidance: list[str] = []
    style_rules: list[str] = []

    @field_validator("domain_guidance", "style_rules", mode="before")
    @classmethod
    def _coerce_lists(cls, v):
        return [str(x) for x in _as_list(v)]


# --------------------------------------------------------------------------- eval set
class EvalCategory(str, Enum):
    single_doc = "single_doc"
    multi_doc = "multi_doc"
    out_of_scope = "out_of_scope"
    edge = "edge"


class EvalCase(BaseModel):
    id: str
    category: EvalCategory
    question: str
    expected_behavior: Literal["answer", "refuse"]
    expected_sources: list[str] = []
    key_facts: list[str] = []
    rationale: str = ""
    generated_from: list[str] = Field(default_factory=list, description="chunk ids the case was generated from")
    needs_tool: Optional[str] = None


class EvalSet(BaseModel):
    cases: list[EvalCase]
    generator_model: str = ""
    notes: list[str] = []


class GenQA(Lenient):
    question: str
    key_facts: list[str] = []
    answer: str = ""

    @field_validator("key_facts", mode="before")
    @classmethod
    def _coerce_lists(cls, v):
        return [str(x) for x in _as_list(v)]


class GenQuestions(Lenient):
    questions: list[str] = []

    @field_validator("questions", mode="before")
    @classmethod
    def _q(cls, v):
        v = _as_list(v)
        return [q if isinstance(q, str) else str(q.get("question", "")) if isinstance(q, dict) else str(q) for q in v]


# --------------------------------------------------------------------------- runtime
class ToolCall(Lenient):
    tool: str
    args: dict[str, Any] = {}

    @field_validator("args", mode="before")
    @classmethod
    def _args(cls, v):
        return v if isinstance(v, dict) else {}


class ToolPlan(Lenient):
    calls: list[ToolCall] = []


class AgentAnswerDraft(Lenient):
    answer: str
    citations: list[str] = []
    refused: bool = False

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce_lists(cls, v):
        return [str(x) for x in _as_list(v)]


class RetrievedChunk(BaseModel):
    chunk_id: str
    filename: str
    locator: str
    text: str
    score: float


class ToolResult(BaseModel):
    tool: str
    args: dict[str, Any]
    output: Any = None
    error: Optional[str] = None


class AgentResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    refused: bool
    retrieved: list[RetrievedChunk]
    tool_results: list[ToolResult] = []
    latency_s: float = 0.0


# --------------------------------------------------------------------------- judge
class ClaimQuote(Lenient):
    claim: str
    quote: str = ""


class FactCheck(Lenient):
    fact: str
    present: bool = False
    answer_quote: str = ""

    @field_validator("present", mode="before")
    @classmethod
    def _b(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1", "present", "supported")
        return bool(v)


class JudgeExtraction(Lenient):
    """Quote-first judging: the model must point at evidence before anything is scored."""
    claims: list[ClaimQuote] = []
    facts: list[FactCheck] = []


class FailureCategory(str, Enum):
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    PROMPT_DEFECT = "PROMPT_DEFECT"
    HALLUCINATION = "HALLUCINATION"
    WRONG_REFUSAL = "WRONG_REFUSAL"


class CaseResult(BaseModel):
    case_id: str
    category: EvalCategory
    question: str
    expected_behavior: str
    answer: str
    refused: bool
    citations: list[str]
    retrieved_files: list[str]
    # deterministic checks
    citations_valid: bool
    invalid_citations: list[str] = []
    citation_present: bool
    expected_source_retrieved: bool
    expected_source_cited: bool
    numeric_ok: Optional[bool] = None
    # scores 0..1
    groundedness: float
    factuality: float
    refusal_accuracy: float
    passed: bool
    failure_category: Optional[FailureCategory] = None
    failure_reason: str = ""
    judge_detail: dict[str, Any] = {}
    latency_s: float = 0.0
    tools_called: list[str] = []  # audit trail: which tools the agent actually used


class Scorecard(BaseModel):
    iteration: int
    agent_version: int
    n_cases: int
    pass_rate: float
    groundedness: float
    factuality: float
    refusal_accuracy: float
    citation_validity: float
    by_category: dict[str, dict[str, float]]
    failures_by_type: dict[str, int]
    overall: float
    n_planned: int = 0  # cases in the eval set; > n_cases when the budget ran out mid-run
    complete: bool = True


# --------------------------------------------------------------------------- optimizer
class Patch(BaseModel):
    target: FailureCategory
    kind: Literal["retrieval", "prompt_rule", "contract"]
    field: str
    before: Any
    after: Any
    rationale: str
    addresses_cases: list[str] = []


class LearnedRules(Lenient):
    rules: list[str] = []

    @field_validator("rules", mode="before")
    @classmethod
    def _coerce_lists(cls, v):
        return [str(x) for x in _as_list(v)]


class IterationRecord(BaseModel):
    iteration: int
    agent_version: int
    score_before: float
    score_after: Optional[float] = None
    pass_rate_before: float
    pass_rate_after: Optional[float] = None
    failures_by_type: dict[str, int]
    patches: list[Patch] = []
    decision: Literal["kept", "reverted", "baseline", "stopped"] = "baseline"
    stop_reason: str = ""
    tokens_used: int = 0
    seconds_used: float = 0.0
