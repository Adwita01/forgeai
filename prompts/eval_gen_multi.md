## SYSTEM
You write one test question that can only be answered correctly by combining BOTH passages below (they come from different documents).
- The question must need at least one fact from passage A and one from passage B (e.g. compare, combine, apply a rule from one to an item in the other).
- key_facts: 2-4 short facts copied (near-)verbatim, at least one from each passage.
- Do not mention file names. If the passages are unrelated, still write the most natural question linking them.

## USER
AUDIENCE: {{audience}}
PASSAGE A {{label_a}}:
{{passage_a}}

PASSAGE B {{label_b}}:
{{passage_b}}

Return JSON {"question": "...", "key_facts": ["..."], "answer": "..."}
