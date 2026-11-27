# Quality report - Procurement Assistant

> **Final: 54% of 24 generated test cases pass** (overall score 0.674). Model `qwen2.5:3b`, judge `qwen2.5:3b`. The numbers below are produced by an automated judge over an automatically generated test set; see *Limitations of this evaluation* before trusting them.

## Brief

> We need a procurement assistant for our technical superintendents and chief engineers. It should look up spare parts in our catalogue (price, stock on hand, minimum stock, lead time, which supplier), calculate the total cost of an order including any urgent or emergency surcharges, tell us who has to approve a purchase order of a given value, and answer questions about the warranty, claims and delivery service levels in our supplier agreements with Kestrel and Halcyon. It should also give us current freight rates for shipping spares to the vessel.

## Requirements and groundability

- **Scope:** Technical superintendents and chief engineers for procurement of spare parts
- **Audience / tone:** Technical superintendents and chief engineers / Formal and technical
- **Grounded topics:** price, stock on hand, minimum stock, lead time, supplier, warranty, claims, delivery service levels
- **Partially grounded (agent told to state what is missing):** spare parts
- **⚠ Ungrounded requirements flagged by the guard (agent will refuse these):**
  - *freight rates* - The excerpt provides specific details about freight rates for both standard and urgent orders, including surcharges for urgent deliveries. | rejected: the documents never use 'rates' (closest evidence: Halcyon_Engine_Parts_Warranty_and_SLA.pdf p.2)
- **Must refuse:** market prices, technical repair advice, legal interpretation of contracts, suppliers other than Kestrel and Halcyon, freight rates (not covered by the documents)

## Generated agent

| Tool | Decision |
|---|---|
| `csv_lookup` | attached: CSV present (spare_parts_catalogue.csv); evidence: look up |
| `calculator` | attached: on: brief/answers mention 'calculate' |
| `open_meteo_weather` | not attached: off: not requested |
| `current_date` | not attached: off: model flagged it but nothing in the brief or answers requires it |

Retrieval: chunk 160 words / overlap 26, top_k 6, hybrid alpha 0.35 (dense weight). Rationale: tabular (CSV) corpus -> more BM25 weight (alpha 0.35); avg citable unit 1346 chars -> chunk 160 words; CSV rows are one chunk each -> top_k +1. Persona written by: llm.

Rules learned by the optimizer:
- State every number, duration, amount and identifier from the relevant passage explicitly in the answer, and when two passages are relevant combine both, citing each.
- When answering, ensure all relevant information is extracted from the documents, even if it requires synthesis of data from multiple sources, and do not refuse unless explicitly instructed to do so.
- If the answer is not found directly, synthesize the information from the available data to provide a comprehensive response.

## Generated eval set

24 cases: edge 5, multi_doc 5, out_of_scope 5, single_doc 9. Every answerable case's key facts were verified to occur in its source passage before inclusion.
Generator notes (4): discarded multi-doc case Halcyon_Engine_Parts_Warranty_and_SLA.pdf::p.1::1 + Kestrel_Marine_Supply_Warranty_and_SLA.pdf::p.1::1: facts not traceable to both passages; discarded refusal question the documents largely answer: What is the lead time for Kestrel spare parts?; discarded refusal question the documents largely answer: What is the warranty period for Halcyon products?; discarded unreadable typo rewrite of S-01: Thx to subm. a warrnty reclm w/in 3wknds & flr lmt'd t hwln 

## Per-iteration scorecard

| Iter | Agent v | Decision | Pass rate | Groundedness | Factuality | Refusal acc. | Citation validity | Overall | Failures |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | baseline | 54% | 0.95 | 0.40 | 0.92 | 1.00 | **0.669** | PROMPT_DEFECT:9, WRONG_REFUSAL:1, HALLUCINATION:1 |
| 1 | 1 | reverted | 50% | 0.90 | 0.42 | 0.83 | 1.00 | **0.630** | PROMPT_DEFECT:8, WRONG_REFUSAL:2, HALLUCINATION:2 |
| 2 | 2 | reverted | 50% | 0.90 | 0.40 | 0.88 | 1.00 | **0.633** | PROMPT_DEFECT:9, WRONG_REFUSAL:2, HALLUCINATION:1 |
| 3 | 3 | kept | 54% | 0.95 | 0.42 | 0.92 | 1.00 | **0.674** | PROMPT_DEFECT:9, WRONG_REFUSAL:1, HALLUCINATION:1 |

### Iteration 1 - reverted
Score 0.669 → 0.630; pass rate 54% → 50%. Cost: 76963 tokens, 77s.
- **PROMPT_DEFECT** (prompt_rule) - added rule(s): "Ensure answers include all required facts and do not omit essential details." / "Provide valid citations or references to support the given information.". Why: 9 PROMPT_DEFECT failures; added 2 llm rule(s). Targets: S-02, S-03, S-05, M-01, M-04, M-05, E-01, E-03
- **WRONG_REFUSAL** (prompt_rule) - added rule(s): "Ensure all answers are derived from the provided documents, even if they are not explicitly stated." / "Refer to the documents for context and answer all implicit questions within the given context.". Why: 1 WRONG_REFUSAL failures; added 2 llm rule(s). Targets: M-02
- **WRONG_REFUSAL** (contract) - `output_contract.refusal_policy`: Refuse when the retrieved context does not contain the answer, or when the question is in a must-refuse topic. Never answer from general knowledge. Known gaps: freight rates. → Refuse when the retrieved context does not contain the answer, or when the question is in a must-refuse topic. Never answer from general knowledge. Known gaps: freight rates. Partial information is not a reason to refuse: answer what the passages state and name what is missing.. Why: agent refused despite relevant retrieved evidence. Targets: M-02
- *overall regressed by 0.039; patch reverted, will try a different patch*

### Iteration 2 - reverted
Score 0.669 → 0.633; pass rate 54% → 50%. Cost: 70767 tokens, 80s.
- **PROMPT_DEFECT** (prompt_rule) - added rule(s): "End every sentence that states a fact with the citation label of its passage, copied exactly from the passage header, e.g. [policy.pdf p.2].". Why: 9 PROMPT_DEFECT failures; added 1 canned rule(s). Targets: S-02, S-03, S-05, M-01, M-04, M-05, E-01, E-03
- **WRONG_REFUSAL** (prompt_rule) - added rule(s): "Ensure the assistant references specific passages to support their answers, even when they seem to contain the answer." / "Clarify the answer by referencing the exact context or passage that provides the information, even if it's not explicitly stated.". Why: 1 WRONG_REFUSAL failures; added 2 llm rule(s). Targets: M-02
- *overall regressed by 0.036; patch reverted, will try a different patch*

### Iteration 3 - kept
Score 0.669 → 0.674; pass rate 54% → 54%. Cost: 74503 tokens, 103s.
- **PROMPT_DEFECT** (prompt_rule) - added rule(s): "State every number, duration, amount and identifier from the relevant passage explicitly in the answer, and when two passages are relevant combine both, citing each.". Why: 9 PROMPT_DEFECT failures; added 1 canned rule(s). Targets: S-02, S-03, S-05, M-01, M-04, M-05, E-01, E-03
- **WRONG_REFUSAL** (prompt_rule) - added rule(s): "When answering, ensure all relevant information is extracted from the documents, even if it requires synthesis of data from multiple sources, and do not refuse unless explicitly instructed to do so." / "If the answer is not found directly, synthesize the information from the available data to provide a comprehensive response.". Why: 1 WRONG_REFUSAL failures; added 2 llm rule(s). Targets: M-02
- *early stop: gain +0.005 below min_gain 0.02*

## Final results by category

| Category | n | Pass rate | Factuality | Groundedness |
|---|---|---|---|---|
| single_doc | 9 | 67% | 0.67 | 1.00 |
| multi_doc | 5 | 20% | 0.10 | 0.80 |
| out_of_scope | 5 | 80% | 0.80 | 1.00 |
| edge | 5 | 40% | 0.30 | 1.00 |

## What still fails and why (11 of 24)

### PROMPT_DEFECT - 9 case(s)
Evidence was present but the answer omitted expected facts or citations. Partly genuine model error, partly eval strictness: key facts are matched literally (numbers exactly), so a correct paraphrase can be under-credited.

- `S-02` (single_doc) *What is the surcharge for urgent orders to be delivered within 72 hours to a vessel's port at one of the three hubs in Singapore, Rotterdam, or Fujairah?*  
  → evidence was retrieved but answer missed expected facts (factuality 0%). Answer: "The surcharge for urgent orders to be delivered within 72 hours to a vessel's port at one of the three hubs in Singapore, Rotterdam, or Fujairah is 20% of the item price."
- `S-03` (single_doc) *What is the approval limit for orders placed with a frame-agreement supplier for Technical Director positions?*  
  → evidence was retrieved but answer missed expected facts (factuality 0%). Answer: "The approval limit for orders placed with a frame-agreement supplier for Technical Director positions is USD 100,000.01."
- `S-05` (single_doc) *What is the maximum percentage of credit applied for late delivery of standard orders, and under what condition does this apply?*  
  → evidence was retrieved but answer missed expected facts (factuality 0%). Answer: "The maximum percentage of credit applied for late delivery of standard orders is 8%. This applies when the order is delivered after its committed date, up to a maximum of 8% of the late items' value, with a maximum of 8%"
- `M-01` (multi_doc) *How does the surcharge for urgent orders compare between Halcyon Engine Parts and Kestrel Marine Supply in terms of percentage for delivery to the nearest stock point within 48 hours?*  
  → evidence was retrieved but answer missed expected facts (factuality 0%). Answer: "The surcharge for urgent orders for delivery to the nearest stock point within 48 hours is 25% for Halcyon Engine Parts and 20% for Kestrel Marine Supply."
- `M-04` (multi_doc) *What is the minimum stock quantity for the Emergency generator battery (part_no: KMS-EL-4005) from Kestrel Marine Supply Ltd. that requires a single quotation for orders placed with a frame-agreement supplier, given its criticality level?*  
  → evidence was retrieved but answer missed expected facts (factuality 0%). Answer: "The minimum stock quantity for the Emergency generator battery (part_no: KMS-EL-4005) from Kestrel Marine Supply Ltd. that requires a single quotation for orders placed with a frame-agreement supplier is 2."
- `M-05` (multi_doc) *For which equipment must a warranty claim be validated with records of installation and delivery within 3 working days, and what is the supplier for a high-criticality navigation spare part with a unit price of $395.00?*  
  → evidence was retrieved but answer missed expected facts (factuality 0%). Answer: "For a high-criticality navigation spare part with a unit price of $395.00, the supplier is Kestrel Marine Supply Ltd. A warranty claim for this part must be validated with records of installation and delivery within 3 wo"

### WRONG_REFUSAL - 1 case(s)
The evidence was retrieved but the model refused - usually over-correction from strict grounding rules, or the evidence was phrased differently from the question. Loosening refusal tends to raise hallucination; the loop keeps whichever balance scores higher overall.

- `M-02` (multi_doc) *Which spares under the warranty terms of Frame Agreement KMS-FA-2026 are covered by a warranty period of 12 months from the date of installation on board, provided that this period does not exceed the 18 months from the date of delivery or installation, whichever expires first?*  
  → refused although the expected source was in the retrieved context. Answer: "The spares under the warranty terms of Frame Agreement KMS-FA-2026 that are covered by a warranty period of 12 months from the date of installation on board, provided that this period does not exceed the 18 months from t"

### HALLUCINATION - 1 case(s)
The model added content not in the context or answered an out-of-scope question. Small models (3-7B) follow 'answer only from context' imperfectly, especially when a near-miss passage looks relevant; prompt rules reduce but do not eliminate this.

- `R-05` (out_of_scope) *How many units are in stock for Kestrel's inventory?*  
  → answered a question that should have been refused. Answer: "For Kestrel Marine Supply Ltd., there are 20 units in stock across all parts in the inventory."

## Judge integrity checks

- Judge quotes rejected as not verbatim in the context (fabricated or garbled): **3**
- Of those, claims still accepted because the claim itself checks out deterministically (exact numbers, >=60% word overlap with the context): **3**
- Cases scored by the deterministic fallback because the LLM judge failed schema validation: **0**
- Answers citing a file/page/row that does not exist in the Evidence Manifest: **0**
- Expected sources retrieved: 100%; expected sources cited: 84%

## Limitations of this evaluation

- The eval set is generated by the same small model family that powers the agent. Questions it can write tend to be ones it can answer, so pass rates are optimistic. Key facts are verified against the source text, but question *difficulty* is not.
- The optimizer tunes against this same eval set (no held-out split - the set is small). Gains may partly be overfitting; rules are leakage-filtered so no test facts enter the prompt, which limits but does not remove this.
- Factuality is matched literally (numbers must match exactly), so correct paraphrases with different units or rounding are under-credited.
- Judge model = agent model unless `FORGE_JUDGE_MODEL` is set (here: same); self-preference is mitigated by quote verification, not eliminated.

## Budget

Total 390776 tokens, 559.9s wall clock (limits: ∞ tokens / 14400.0s).

| Stage | Calls | Tokens | Seconds | Schema retries | Hard failures |
|---|---|---|---|---|---|
| agent_answer | 96 | 243204 | 217.5 | 0 | 0 |
| agent_tool_plan | 36 | 34009 | 54.1 | 0 | 0 |
| embed | 110 | 15894 | 7.3 | 0 | 0 |
| eval_gen | 19 | 10172 | 41.7 | 0 | 0 |
| generator | 1 | 1431 | 6.4 | 0 | 0 |
| groundability | 10 | 9554 | 26.9 | 0 | 0 |
| interviewer | 2 | 4716 | 24.8 | 0 | 0 |
| judge | 35 | 69899 | 176.3 | 1 | 0 |
| optimizer | 4 | 1897 | 3.6 | 0 | 0 |

