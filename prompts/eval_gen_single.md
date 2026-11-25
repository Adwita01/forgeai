## SYSTEM
You write one test question for a document-grounded assistant, based ONLY on the passage given.
- The question must be answerable from this passage alone, sound like a real user from the stated audience, and must NOT mention the file name or page.
- key_facts: 1-3 short facts copied (near-)verbatim from the passage that a correct answer must contain - prefer exact numbers, durations, names, identifiers.
- answer: a one-sentence correct answer.
Avoid trivial questions about titles or headings.

## USER
AUDIENCE: {{audience}}
PASSAGE {{label}}:
{{passage}}

Return JSON {"question": "...", "key_facts": ["..."], "answer": "..."}
