"""Regression tests for defects found when auditing the first full demo runs."""
from __future__ import annotations

from forgeai.schemas import GapFlag, Requirements
from forgeai.stages.eval_gen import EvalGenerator
from forgeai.stages.groundability import missing_topic_words
from forgeai.stages.judge import _claim_supported, _is_restated_question


class _Chunk:
    def __init__(self, text):
        self.text = text


class _Retriever:
    def __init__(self, texts):
        self.chunks = [_Chunk(t) for t in texts]


CORPUS = _Retriever([
    "Urgent orders are dispatched within 72 hours at a surcharge of 20%. Emergency orders add 25% plus actual freight cost.",
    "Spares are warranted against defects in material and workmanship for 18 months from delivery.",
    "Purchase orders must be approved according to their total value. Approval limits: up to 5,000 Technical Superintendent.",
])


def test_guard_flags_topic_word_the_corpus_never_uses():
    # demo 2: the documents talk about freight *costs*, never freight *rates*
    assert missing_topic_words("freight rates", CORPUS) == {"rates"}
    # prefix matching: "warranty" is covered by "warranted"; task words like "lookup" are ignored
    assert missing_topic_words("warranty", CORPUS) == set()
    assert missing_topic_words("surcharge lookup", CORPUS) == set()
    assert missing_topic_words("approval limits", CORPUS) == set()


def test_judge_drops_restated_questions_and_rescues_checkable_claims():
    q = "What is the minimum and maximum value for a single quotation?"
    assert _is_restated_question("What is the minimum and maximum value for a single quotation", q)
    assert _is_restated_question("Is this allowed?", q)
    assert not _is_restated_question("The surcharge for urgent orders is 20%.", q)
    ev = CORPUS.chunks[0].text
    assert _claim_supported("The surcharge for urgent orders dispatched within 72 hours is 20%", ev)
    assert not _claim_supported("The surcharge for urgent orders dispatched within 72 hours is 30%", ev)  # numbers exact
    assert not _claim_supported("Crew must carry the emergency contact card ashore", ev)


def _gen(req):
    g = EvalGenerator.__new__(EvalGenerator)
    g.req, g.chunks = req, CORPUS.chunks
    return g


def test_requester_exclusions_become_refusal_traps():
    req = Requirements(scope="spares procurement", must_answer_topics=["warranty"], must_refuse_topics=[
        "market prices", "freight rates not in documents", "anything not written in our policies",
        "pension contributions (not covered by the documents)", "technical repair advice"])
    req.ungrounded_requirements = [GapFlag(requirement="pension contributions", reason="absent")]
    traps = _gen(req)._policy_refusals()
    assert traps == ["market prices", "freight rates", "technical repair advice"]  # cleaned, no catch-alls, no gap duplicates


def test_refusal_questions_the_documents_answer_are_rejected():
    g = _gen(Requirements(scope="spares procurement", must_answer_topics=["x"]))
    assert g._answerable("What surcharge applies to urgent orders dispatched within 72 hours?")
    assert not g._answerable("What is the current market price of marine diesel in Rotterdam?")


# ---------------------------------------------------------------- second audit round


def test_sentence_final_words_match():
    from forgeai.textutil import content_tokens, fact_supported
    assert "singapore" in content_tokens("It is stocked in Singapore.")
    assert {"p.2", "6s50me-c"} <= content_tokens("see p.2 for the 6S50ME-C")


def test_catalogue_field_facts_match_on_value():
    from forgeai.textutil import fact_supported
    a = "The minimum stock for the fuel valve nozzle is 6, and it is stocked in Singapore."
    assert fact_supported("min_stock: 6", a) and fact_supported("warehouse: Singapore", a)
    assert not fact_supported("min_stock: 8", a)


def test_one_step_arithmetic_is_grounded_but_wrong_totals_are_not():
    ev = "part_no: HEP-ME-1004; unit_price_usd: 2380.00"
    q = "How much would 4 units of HEP-ME-1004 cost in total?"
    assert _claim_supported("4 units of HEP-ME-1004 cost 9520.00 in total", ev, q)
    assert not _claim_supported("4 units of HEP-ME-1004 cost 9250.00 in total", ev, q)


def test_document_code_citations_resolve_to_the_unique_file():
    from forgeai.schemas import EvidenceManifest, FileEntry
    from forgeai.stages.judge import canonical_citation, validate_citation
    m = EvidenceManifest(folder="d", files=[FileEntry(filename="CRT-03_Crew_Leave.pdf", kind="pdf", n_pages=2),
                                            FileEntry(filename="CSL-02_Shore_Leave.pdf", kind="pdf", n_pages=2)])
    c = canonical_citation("[CRT-03 p.2]", m)
    assert c == "[CRT-03_Crew_Leave.pdf p.2]" and validate_citation(c, m)[0]
    assert canonical_ciation_untouched(m)


def canonical_ciation_untouched(m):
    from forgeai.stages.judge import canonical_citation
    return canonical_citation("[C p.1]", m) == "[C p.1]"  # ambiguous / too short: left alone, so it fails validation


def test_generated_cases_whose_facts_miss_the_question_are_rejected():
    from forgeai.stages.eval_gen import _bad_question
    src = "Shore staff get 24 working days of annual leave, rising to 30 after 5 years of continuous service."
    assert _bad_question("What leave do staff with 5 years of service get?", ["Shore staff get 24 working days of annual leave"], src)
    assert not _bad_question("What leave do staff with 5 years of service get?", ["rising to 30 after 5 years"], src)
    assert _bad_question("Which of the following is the restocking fee: 20% or 15%?", ["15%"], "fee 15%") == "multiple-choice format"


def test_must_answer_topics_must_come_from_the_requester():
    from forgeai.stages.interviewer import _shares_word
    hr = "answers HR policy questions: annual leave, sick and parental leave, crew rotation"
    assert not _shares_word("part number lookup", hr)
    assert _shares_word("spare part lookup", "look up spare parts in our catalogue")  # plural-insensitive


# ---------------------------------------------------------------- third audit round

ROW = ("[spare_parts_catalogue.csv] columns: part_no, unit_price_usd\n"
       "row 4: part_no: HEP-ME-1004; description: Piston ring set; unit_price_usd: 2380.00; stock_qty: 8")


def _agent_with_csv():
    from forgeai.agent_runtime import Agent
    from forgeai.tools import ToolContext
    a = Agent.__new__(Agent)
    a.tool_ctx = ToolContext(tables={"spare_parts_catalogue.csv": []})
    return a


def test_tool_guard_rails():
    from forgeai.schemas import RetrievedChunk
    a = _agent_with_csv()
    rc = [RetrievedChunk(chunk_id="x", filename="spare_parts_catalogue.csv", locator="row 4", text=ROW, score=1.0)]
    q = "How much would 4 units of HEP-ME-1004 cost in total?"
    assert a._sanitise_call("calculator", {"expression": "4 * unit_price_usd"}, q, rc, [])["expression"] == "4 * 2380.00"
    assert a._sanitise_call("calculator", {"expression": "100 / 3"}, "Which items are consumables?", rc, []) is None
    assert a._sanitise_call("csv_lookup", {"file": "Kestrel_SLA.pdf p.1"}, q, rc, []) is None
    assert a._sanitise_call("csv_lookup", {"file": "positions.csv"}, q, rc, [])["file"] == "spare_parts_catalogue.csv"


def test_unreadable_rewrites_and_questions_about_absent_things_are_rejected():
    from forgeai.stages.eval_gen import _absent_terms, _readable
    vocab = {"seafarer", "leave", "annual", "working", "days", "unused", "carry", "over", "watch", "alcohol"}
    assert _readable("Can u carry over more than 5 working days of unused annual leave?", vocab)
    assert not _readable("whch mcssn ls nmx tnsntn 4 sfrr'r's trw r n brn n brr", vocab)
    stems = {w[:5] for w in ["approval", "limit", "technical", "director", "frame", "supplier"]}
    assert len(_absent_terms("What is the approval limit for Technical Director positions and the corresponding salary range?", stems)) >= 3
    assert len(_absent_terms("Under what conditions does the approval limit apply?", stems)) < 3


def test_top_up_asks_for_a_different_fact_on_reused_passages():
    from forgeai.stages.eval_gen import EvalGenerator
    seen = []
    g = EvalGenerator.__new__(EvalGenerator)
    g.chunks = [type("C", (), {"chunk_id": "a::p.1::0", "filename": "a.pdf", "kind": "pdf", "text": "Annual leave is 24 working days for shore staff. " * 8})()]
    g.cases, g._used_chunks, g.rng, g.progress = [], {"a::p.1::0"}, __import__("random").Random(1), lambda m: None
    g._gen_single = lambda c, avoid=None: seen.append(avoid)
    g.single_doc(1)              # fresh passages only: the one passage is used -> nothing tried
    g.single_doc(1, reuse=True)  # top-up: the used passage is retried with an "avoid" list
    assert seen == [["(none yet)"]]
