# Quality report - Bluewater Fleet Management HR Assistant

> **Final: 77% of 22 generated test cases pass** (overall score 0.802). Model `qwen2.5:3b`, judge `qwen2.5:3b`. The numbers below are produced by an automated judge over an automatically generated test set; see *Limitations of this evaluation* before trusting them.

## Brief

> I need an assistant that answers HR policy questions for Bluewater Fleet Management. It should help both our shore staff in the Singapore, Rotterdam and Mumbai offices and our seafarers on board: annual leave and how to apply for it, sick and parental leave, the rules for crew shore leave in port, crew rotation, relief and travel arrangements, and also questions about our pension contributions. Answers must point to the policy they come from.

## Requirements and groundability

- **Scope:** HR policy questions for Bluewater Fleet Management
- **Audience / tone:** Bluewater Fleet Management shore staff and seafarers in Singapore, Rotterdam, and Mumbai / plain, friendly and precise
- **Grounded topics:** annual leave, sick leave, parental leave, crew shore leave, crew rotation
- **Partially grounded (agent told to state what is missing):** relief and travel arrangements
- **⚠ Ungrounded requirements flagged by the guard (agent will refuse these):**
  - *pension contributions* - The provided excerpts do not contain any information related to pension contributions. They discuss leave policies and related matters, but do not mention anything about pension contributions. (closest evidence: HRP-01_Shore_Staff_Leave_Policy.pdf p.1)
- **Must refuse:** individual disputes, disciplinary cases, legal advice, salary and payroll questions, anything not written in policies, pension contributions (not covered by the documents)

## Generated agent

| Tool | Decision |
|---|---|
| `csv_lookup` | not attached: off: no CSV in the documents |
| `calculator` | not attached: off: not requested |
| `open_meteo_weather` | not attached: off: not requested |
| `current_date` | not attached: off: not requested |

Retrieval: chunk 160 words / overlap 26, top_k 5, hybrid alpha 0.35 (dense weight). Rationale: 6 code-like identifiers (clause/document ids) -> more BM25 weight (alpha 0.35); avg citable unit 1652 chars -> chunk 160 words. Persona written by: llm.

Rules learned by the optimizer:
- Ensure answers include all required facts and do not omit essential details.
- Provide direct citations or references from the documents to support the answer.
- Include a direct citation or reference to the HRP-01 policy for the Chief Officer's tour length, ensuring the answer is grounded in the provided documents.

## Generated eval set

22 cases: edge 2, multi_doc 3, out_of_scope 5, single_doc 12. Every answerable case's key facts were verified to occur in its source passage before inclusion.
Generator notes (15): discarded single-doc case from HRP-01_Shore_Staff_Leave_Policy.pdf::p.1::0: question asks about ['5'] but no key fact mentions it; discarded single-doc case from CSL-02_Crew_Shore_Leave_and_Port_Conduct.pdf::p.1::1: question asks about things the documents never mention (number, positions, represented); discarded multi-doc case CRT-03_Crew_Leave_Rotation_and_Travel.pdf::p.2::0 + CSL-02_Crew_Shore_Leave_and_Port_Conduct.pdf::p.2::1: facts not traceable to both passages; discarded multi-doc case CSL-02_Crew_Shore_Leave_and_Port_Conduct.pdf::p.2::0 + HRP-01_Shore_Staff_Leave_Policy.pdf::p.2::1: facts not traceable to both passages; discarded multi-doc case CSL-02_Crew_Shore_Leave_and_Port_Conduct.pdf::p.1::0 + HRP-01_Shore_Staff_Leave_Policy.pdf::p.1::1: question asks about ['1'] but no key fact mentions it; discarded refusal question the documents largely answer: What is the policy for dealing with individual disputes in Bluewater Fleet Management?; ...

## Per-iteration scorecard

| Iter | Agent v | Decision | Pass rate | Groundedness | Factuality | Refusal acc. | Citation validity | Overall | Failures |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | baseline | 77% | 0.88 | 0.63 | 0.95 | 1.00 | **0.802** | PROMPT_DEFECT:3, WRONG_REFUSAL:1, HALLUCINATION:1 |
| 1 | 1 | kept | 77% | 0.88 | 0.63 | 0.95 | 1.00 | **0.802** | PROMPT_DEFECT:3, WRONG_REFUSAL:1, HALLUCINATION:1 |

### Iteration 1 - kept
Score 0.802 → 0.802; pass rate 77% → 77%. Cost: 56069 tokens, 68s.
- **PROMPT_DEFECT** (prompt_rule) - added rule(s): "Ensure answers include all required facts and do not omit essential details." / "Provide direct citations or references from the documents to support the answer.". Why: 3 PROMPT_DEFECT failures; added 2 llm rule(s). Targets: S-03, S-05, E-02
- **WRONG_REFUSAL** (prompt_rule) - added rule(s): "Include a direct citation or reference to the HRP-01 policy for the Chief Officer's tour length, ensuring the answer is grounded in the provided documents.". Why: 1 WRONG_REFUSAL failures; added 1 llm rule(s). Targets: M-01
- **WRONG_REFUSAL** (contract) - `output_contract.refusal_policy`: Refuse when the retrieved context does not contain the answer, or when the question is in a must-refuse topic. Never answer from general knowledge. Known gaps: pension contributions. → Refuse when the retrieved context does not contain the answer, or when the question is in a must-refuse topic. Never answer from general knowledge. Known gaps: pension contributions. Partial information is not a reason to refuse: answer what the passages state and name what is missing.. Why: agent refused despite relevant retrieved evidence. Targets: M-01
- *early stop: gain +0.000 below min_gain 0.02*

## Final results by category

| Category | n | Pass rate | Factuality | Groundedness |
|---|---|---|---|---|
| single_doc | 12 | 83% | 0.72 | 1.00 |
| multi_doc | 3 | 67% | 0.33 | 0.67 |
| out_of_scope | 5 | 100% | 1.00 | 1.00 |
| edge | 2 | 0% | 0.50 | 0.50 |

## What still fails and why (5 of 22)

### PROMPT_DEFECT - 3 case(s)
Evidence was present but the answer omitted expected facts or citations. Partly genuine model error, partly eval strictness: key facts are matched literally (numbers exactly), so a correct paraphrase can be under-credited.

- `S-03` (single_doc) *What is the maximum number of working days of unused annual leave that can be carried over to the next calendar year?*  
  → evidence was retrieved but answer missed expected facts (factuality 0%). Answer: "The maximum number of working days of unused annual leave that can be carried over to the next calendar year is up to 5 working days."
- `S-05` (single_doc) *How many hours before a scheduled watch must alcohol consumption be avoided according to the document?*  
  → evidence was retrieved but answer missed expected facts (factuality 0%). Answer: "According to the document, no alcohol may be consumed within 4 hours before a scheduled watch."
- `E-02` (edge) *what is the combined tour length for a Chief Officer (rank group) who has completed 5 yrs of continuous service and is entitled to 30 wd of paid annual leave per cal yr, according to the HRP-01 and CRT-03 policies? (assume the CRT-03 states 4 mo tour for Senior officers, and HRP-01 states 24 wd for shore staff)*  
  → evidence was retrieved but answer missed expected facts (factuality 0%). Answer: "The combined tour length for a Chief Officer (rank group) who has completed 5 years of continuous service and is entitled to 30 working days of paid annual leave per calendar year, according to the HRP-01 policy, is 4 mo"

### WRONG_REFUSAL - 1 case(s)
The evidence was retrieved but the model refused - usually over-correction from strict grounding rules, or the evidence was phrased differently from the question. Loosening refusal tends to raise hallucination; the loop keeps whichever balance scores higher overall.

- `M-01` (multi_doc) *What is the combined tour length for a Chief Officer (rank group) who has completed 5 years of continuous service and is entitled to 30 working days of paid annual leave per calendar year, according to the HRP-01 and CRT-03 policies?*  
  → refused although the expected source was in the retrieved context. Answer: "According to the HRP-01 policy, a shore staff member with 5 years of continuous service is entitled to 30 working days of paid annual leave per calendar year. The CRT-03 policy does not provide specific tour lengths for "

### HALLUCINATION - 1 case(s)
The model added content not in the context or answered an out-of-scope question. Small models (3-7B) follow 'answer only from context' imperfectly, especially when a near-miss passage looks relevant; prompt rules reduce but do not eliminate this.

- `E-01` (edge) *How many working days of annual leave can be kept over to next yr and hv to b used by march 31?*  
  → only 0% of claims are supported by the context. Answer: "Up to 5 working days of unused annual leave may be carried over to the next calendar year and must be used by 31 March; carried-over days not used by that date are forfeited."

## Judge integrity checks

- Judge quotes rejected as not verbatim in the context (fabricated or garbled): **2**
- Of those, claims still accepted because the claim itself checks out deterministically (exact numbers, >=60% word overlap with the context): **2**
- Cases scored by the deterministic fallback because the LLM judge failed schema validation: **0**
- Answers citing a file/page/row that does not exist in the Evidence Manifest: **0**
- Expected sources retrieved: 100%; expected sources cited: 94%

## Limitations of this evaluation

- The eval set is generated by the same small model family that powers the agent. Questions it can write tend to be ones it can answer, so pass rates are optimistic. Key facts are verified against the source text, but question *difficulty* is not.
- The optimizer tunes against this same eval set (no held-out split - the set is small). Gains may partly be overfitting; rules are leakage-filtered so no test facts enter the prompt, which limits but does not remove this.
- Factuality is matched literally (numbers must match exactly), so correct paraphrases with different units or rounding are under-credited.
- Judge model = agent model unless `FORGE_JUDGE_MODEL` is set (here: same); self-preference is mitigated by quote verification, not eliminated.

## Budget

Total 164059 tokens, 452.2s wall clock (limits: ∞ tokens / 14400.0s).

| Stage | Calls | Tokens | Seconds | Schema retries | Hard failures |
|---|---|---|---|---|---|
| agent_answer | 44 | 90569 | 89.1 | 0 | 0 |
| embed | 53 | 6888 | 33.9 | 0 | 0 |
| eval_gen | 32 | 17083 | 65.8 | 0 | 0 |
| generator | 1 | 1204 | 4.3 | 0 | 0 |
| groundability | 7 | 7114 | 18.7 | 0 | 0 |
| interviewer | 2 | 3477 | 138.5 | 0 | 0 |
| judge | 21 | 36739 | 99.0 | 0 | 0 |
| optimizer | 2 | 985 | 2.1 | 0 | 0 |

