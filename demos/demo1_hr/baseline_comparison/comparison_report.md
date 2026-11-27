# Agent comparison: factory vs baseline

Both agents were run on the same eval set (`demos/demo1_hr/run/eval_set.json`), the same model and the same judge.

| Metric | factory | baseline | Δ |
|---|---|---|---|
| pass_rate | 0.773 | 0.773 | +0.000 |
| overall | 0.802 | 0.829 | -0.027 |
| groundedness | 0.882 | 1.000 | -0.118 |
| factuality | 0.627 | 0.598 | +0.029 |
| refusal_accuracy | 0.955 | 1.000 | -0.045 |
| citation_validity | 1.000 | 0.941 | +0.059 |

## By category

| Category | factory pass | baseline pass |
|---|---|---|
| edge | 0% | 50% |
| multi_doc | 67% | 33% |
| out_of_scope | 100% | 100% |
| single_doc | 83% | 83% |

## Where factory wins (2)

- `M-03` *How does the travel per diem policy for seafarers relate to the sick leave policy for shore staff regarding the requirement for a medical certificate?* - baseline failed: PROMPT_DEFECT (invalid citations: ['[CRT-03_Crew_Leave_Rotation_and_Travel.pdf p.5] (page 5 does not exist in CRT-03_Crew_Leave_Rotation_and_Travel.pdf (2 pages))'])
- `S-12` *What is the minimum percentage of crew members that must remain on board at all times?* - baseline failed: PROMPT_DEFECT (evidence was retrieved but answer missed expected facts (factuality 0%))

## Where baseline wins (2)

- `S-03` *What is the maximum number of working days of unused annual leave that can be carried over to the next calendar year?* - factory failed: PROMPT_DEFECT (evidence was retrieved but answer missed expected facts (factuality 0%))
- `E-01` *How many working days of annual leave can be kept over to next yr and hv to b used by march 31?* - factory failed: HALLUCINATION (only 0% of claims are supported by the context)

## Both fail (3)

- `S-05` *How many hours before a scheduled watch must alcohol consumption be avoided according to the document?* - PROMPT_DEFECT
- `M-01` *What is the combined tour length for a Chief Officer (rank group) who has completed 5 years of continuous service and is entitled to 30 working days of paid annual leave per calendar year, according to the HRP-01 and CRT-03 policies?* - WRONG_REFUSAL
- `E-02` *what is the combined tour length for a Chief Officer (rank group) who has completed 5 yrs of continuous service and is entitled to 30 wd of paid annual leave per cal yr, according to the HRP-01 and CRT-03 policies? (assume the CRT-03 states 4 mo tour for Senior officers, and HRP-01 states 24 wd for shore staff)* - PROMPT_DEFECT
