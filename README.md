# ForgeAI - an agent factory for small local models

ForgeAI turns a plain-language **brief** and a **folder of documents** (PDF, CSV, TXT/MD) into a tested,
self-optimised, document-grounded agent, unattended, on a 3B model running locally under Ollama.
Every stage writes an inspectable artifact, and the final `quality_report.md` says what still fails and why.

```
brief + docs ─► evidence manifest ─► interviewer (≤5 Qs) ─► groundability guard ─► generator
            (deterministic)        requirements.json      flags gaps → refusals    agent_definition.json
                                                                                        │
     quality_report.md ◄── critic-optimizer (≤3 iters, revert on regression) ◄── hybrid judge ◄── eval generator (≥22 cases)
```

## Quick start

```powershell
# 1. Ollama + models (defaults; change them in .env)
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

# 2. Install
python -m pip install -e .
Copy-Item .env.example .env          # optional: model, judge model, budgets

# 3. Check the backend
python -m forgeai doctor

# 4. Build an agent (interactive interview)
python -m forgeai build -b brief.txt -d path\to\docs -o out\my_agent
#    ...or fully unattended
python -m forgeai build -b brief.txt -d path\to\docs -o out\my_agent --auto-answer

# 5. Chat with it
python -m forgeai run out\my_agent\agent_definition.json
streamlit run app.py                 # Runs / Chat / Build tabs
```

`--brief` accepts a file path or literal text. `python -m forgeai --help` lists every command:
`build`, `run`, `eval`, `compare`, `manifest`, `doctor`.

## How each requirement is met

| # | Requirement | Where | How |
|---|---|---|---|
| - | Evidence manifest | `manifest.py` | Built **before any LLM call**: files, pages, CSV columns and dtypes, section headings, key entities. Every later prompt and every citation check uses it. |
| 1 | Interviewer | `stages/interviewer.py` | At most 5 questions grounded in brief and manifest. Interactive, `--answers file.json` or `--auto-answer`. Writes `requirements.json`; `--approve` adds a human checkpoint. |
| 2 | Groundability guard | `stages/groundability.py` | Probes the corpus for each must-answer topic and needs a verbatim quote to accept it. Unsupported topics become `ungrounded_requirements` and are moved to must-refuse. |
| 3 | Declarative generator | `stages/generator.py` | System prompt, retrieval config and output contract. **Tools are gated deterministically** from the brief and the requester's answers (with negation detection), never from model opinion alone. Each tool's decision is logged. |
| 4 | Eval generator | `stages/eval_gen.py` | ≥22 cases: single-doc, multi-doc, CSV and calculator, refusal traps (including the gaps found by the guard), and edge cases (vague, typo, indirect, negation). Key facts are verified to occur in the source passage before a case is kept. |
| 5 | Hybrid judge | `stages/judge.py` | Deterministic checks first (refusal, citation validity against the manifest, exact numbers). Then a quote-before-score LLM judge: every claim must carry a quote, and any quote not found verbatim in the retrieved context is rejected as fabricated. |
| 6 | Critic-optimizer | `stages/optimizer.py` | Classifies each failure as RETRIEVAL_MISS, PROMPT_DEFECT, HALLUCINATION or WRONG_REFUSAL and applies a targeted patch (retrieval ladder, leakage-filtered rules, contract fixes). Reverts on regression, stops early on small gains or when the budget runs out. Writes `iteration_log.json`. |
| 7 | Quality report | `report.py` | Per-iteration scorecard, what still fails and why, judge integrity counters, and the limitations of the evaluation. |
| 8 | Runners | `cli.py`, `app.py` | `forge run` CLI chat (`/context` shows retrieved chunks) and a Streamlit UI. |
| 9 | Demos | `demos/` | Two complete runs with all artifacts committed (below). |

**Robustness on small models.** Structured calls use grammar-constrained JSON (Ollama `format` = JSON schema) and are validated with Pydantic. A self-healing loop feeds validation errors back to the model, up to 3 times. Every LLM stage also has a deterministic fallback, so a broken model output degrades the run instead of crashing it. Retrieval is hybrid (BM25 plus `nomic-embed-text`), falling back to TF-IDF when embeddings are unavailable.

**Failure behaviour.** If the time or token budget runs out in any stage, the build finishes cleanly. The setup stages fall back to deterministic defaults, evaluation scores only the cases it completed (the scorecard is marked `complete: false`), the optimizer stops, and the report opens with a "cut short" banner. A dead, out-of-memory or stalled model server is a different case: the build stops within a couple of retries, not after hours of timeouts. Every run leaves a `run_status.json` (`complete`, `budget_exhausted` or `aborted`, plus the stage it reached). Because completed LLM calls are cached, rerunning the same command resumes almost instantly.

**Stretch goals.** Token and time budgets with early stopping (`--max-tokens`, `--max-seconds`, per-stage accounting in `budget.json`); a factory-vs-hand-crafted baseline comparison (`forge compare`); a human checkpoint on requirements (`--approve`).

## Demos

Both corpora are synthetic and fictional. The generator scripts and prompts are committed in each demo's `synthesis/` folder.

| Demo | Corpus | Deliberate gap in the brief | Expected tools |
|---|---|---|---|
| `demos/demo1_hr` | 3 HR / shore-leave policy PDFs | pension contributions | none |
| `demos/demo2_spares` | parts catalogue CSV + 2 supplier warranty/SLA PDFs + procurement procedure | current freight rates | `csv_lookup`, `calculator` |

Each `run/` folder contains `requirements.json`, `agent_definition.json`, `eval_set.json`, `iteration_log.json`,
`scorecard.json` and `quality_report.md`. It also keeps the intermediate artifacts: `evidence_manifest.json`, `clarifications.json`, `groundability.json`, per-iteration definitions and results, and `budget.json`.

## Results

<!-- RESULTS:START -->
### Demo 1 - Maritime HR & Shore-Leave

- **Agent:** Bluewater Fleet Management HR Assistant (v1)
- **Tools attached:** none
- **Retrieval:** chunk 160 / overlap 26, top_k 5, alpha 0.35
- **Ungrounded requirements caught:** pension contributions
- **Eval set:** 22 cases (edge 2, multi_doc 3, out_of_scope 5, single_doc 12)
- **Cost:** 164059 tokens, 452.2 s wall clock

| Iter | Agent v | Pass rate | Groundedness | Factuality | Refusal acc. | Citation validity | Overall |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 77% | 0.88 | 0.63 | 0.95 | 1.00 | 0.802 |
| 1 | 1 | 77% | 0.88 | 0.63 | 0.95 | 1.00 | 0.802 |

Final (best kept version v1): **77% pass, overall 0.802**. Full honest write-up: [`demos/demo1_hr/run/quality_report.md`](demos/demo1_hr/run/quality_report.md).

### Demo 2 - Vessel Spares & Procurement

- **Agent:** Procurement Assistant (v3)
- **Tools attached:** `csv_lookup`, `calculator`
- **Retrieval:** chunk 160 / overlap 26, top_k 6, alpha 0.35
- **Ungrounded requirements caught:** freight rates
- **Eval set:** 24 cases (edge 5, multi_doc 5, out_of_scope 5, single_doc 9)
- **Cost:** 390776 tokens, 559.9 s wall clock

| Iter | Agent v | Pass rate | Groundedness | Factuality | Refusal acc. | Citation validity | Overall |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 54% | 0.95 | 0.40 | 0.92 | 1.00 | 0.669 |
| 1 | 1 | 50% | 0.90 | 0.42 | 0.83 | 1.00 | 0.630 |
| 2 | 2 | 50% | 0.90 | 0.40 | 0.88 | 1.00 | 0.633 |
| 3 | 3 | 54% | 0.95 | 0.42 | 0.92 | 1.00 | 0.674 |

Final (best kept version v3): **54% pass, overall 0.674**. Full honest write-up: [`demos/demo2_spares/run/quality_report.md`](demos/demo2_spares/run/quality_report.md).

### Factory vs hand-crafted baseline (Demo 1)

Same eval set, same model, same judge. Baseline: [`demos/demo1_hr/baseline/agent_definition.json`](demos/demo1_hr/baseline/agent_definition.json).

| Metric | Factory | Hand-crafted | Δ |
|---|---|---|---|
| pass_rate | 0.773 | 0.773 | +0.000 |
| overall | 0.802 | 0.829 | -0.027 |
| groundedness | 0.882 | 1.000 | -0.118 |
| factuality | 0.627 | 0.598 | +0.029 |
| refusal_accuracy | 0.955 | 1.000 | -0.045 |
| citation_validity | 1.000 | 0.941 | +0.059 |

Case-by-case breakdown: [`demos/demo1_hr/baseline_comparison/comparison_report.md`](demos/demo1_hr/baseline_comparison/comparison_report.md).
<!-- RESULTS:END -->

## Reproducing

```powershell
python -m forgeai build -b demos\demo1_hr\brief.txt -d demos\demo1_hr\docs -o demos\demo1_hr\run --answers demos\demo1_hr\answers.json --auto-answer --max-iters 3
python -m forgeai build -b demos\demo2_spares\brief.txt -d demos\demo2_spares\docs -o demos\demo2_spares\run --answers demos\demo2_spares\answers.json --auto-answer --max-iters 3
python -m forgeai compare demos\demo1_hr\run\agent_definition.json demos\demo1_hr\baseline\agent_definition.json -e demos\demo1_hr\run\eval_set.json -o demos\demo1_hr\baseline_comparison
python scripts\update_readme.py
```

**No suitable local hardware?** [`kaggle/forgeai_kaggle.ipynb`](kaggle/forgeai_kaggle.ipynb) runs all of the above on a free Kaggle T4 GPU with the same Ollama model. Create the upload with `python scripts/make_kaggle_bundle.py`; the setup steps are in the notebook's first cell.

LLM responses are cached in `.forge_cache/`, so a rerun with unchanged inputs replays quickly. Pass `--no-cache` to force fresh calls.
To re-score an edited agent against a fixed test set, use `--reuse-eval-set path\to\eval_set.json`.

## Honest limitations

- A full demo build takes about 10 minutes on a free Kaggle T4 GPU, but roughly 1-3 hours on a laptop CPU, mostly agent answers and judge calls (around 15-60 s each).
- The eval set is written by the same small model family the agent uses, so pass rates are optimistic. The optimizer tunes on that same set (there is no held-out split), so part of any gain may be overfitting.
- Factuality is matched literally, with numbers matched exactly. A correct paraphrase can be under-credited, and a partially correct answer can still pass the 50% factuality threshold.
- The judge model is the agent model unless `FORGE_JUDGE_MODEL` is set. Quote verification limits self-preference but does not remove it.

## Tests

```powershell
python -m pytest -q      # uses a fake LLM; no Ollama needed
```
