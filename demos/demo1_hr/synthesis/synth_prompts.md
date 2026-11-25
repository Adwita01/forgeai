# Demo 1 - synthesis prompts

The three policy PDFs are **LLM-synthesised** (the assignment allows this). The prompts below were given to
an LLM (Claude); its output was reviewed for internal consistency (no contradictory figures across documents,
every number stated once) and frozen as data in `make_docs.py`, which renders the PDFs deterministically.
The company, its offices and all figures are fictional.

## Shared system prompt

> You write realistic internal policy documents for a fictional ship-management company, "Bluewater Fleet
> Management Pte. Ltd.", with offices in Singapore, Rotterdam and Mumbai. Write in the dry, numbered-clause style of
> real HR policies. Every rule must carry concrete, checkable specifics (numbers of days, notice periods,
> amounts, thresholds, who approves). Do not mention pensions, salaries, payroll dates or bonuses - those are
> deliberately out of scope for this corpus. Return sections as `(heading, [paragraphs])`; tables as row lists.

## Prompt 1 - HRP-01 Shore Staff Leave Policy

> Write the shore-staff leave policy (3 pages): scope (shore staff only, seafarers excluded and referred to CRT-03),
> annual leave by length of service with a table, accrual and probation, how to request leave and notice periods,
> carry-over and encashment, sick leave and medical certificates, parental leave (maternity, paternity, adoption),
> compassionate, marriage, unpaid and study leave, public holidays.

## Prompt 2 - CSL-02 Crew Shore Leave and Port Conduct Policy

> Write the seafarers' shore-leave policy grounded in MLC 2006 Regulation 2.4 (2-3 pages): eligibility (berth vs
> anchorage, minimum port stay, Master's approval, minimum crew on board), when shore leave is not granted,
> return-on-board deadline, ISPS security levels 2 and 3, high-risk port curfew, documents and gangway log, alcohol
> limits, and the procedure when port authorities deny shore leave. Include one cross-reference to CRT-03.

## Prompt 3 - CRT-03 Crew Leave, Rotation and Travel Policy

> Write the seafarers' rotation and travel policy (2-3 pages): tour lengths by rank group as a table, maximum
> continuous service under MLC Standard A2.5.1, leave accrual (MLC minimum and the officers' permanent-contract
> rate), relief notice and delay allowance, compassionate early sign-off and repatriation, travel class by rank and
> flight length, per diem, excess baggage, medical fitness on joining.

## Design intent (why the corpus looks like this)

- **Two leave regimes in separate documents** (shore staff vs seafarers) - forces multi-document questions
  ("compare compassionate leave for office staff and crew") and punishes agents that mix them up.
- **Many exact figures** (14 / 30 days notice, 0.05%, USD 40, 72 hours) - makes factuality deterministically checkable.
- **A deliberate gap**: the brief asks about pension contributions, which no document covers - the
  groundability guard must flag it.
- **No arithmetic, dates or weather needed** - the factory must not attach calculator or weather tools.
