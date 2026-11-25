## SYSTEM
You write the persona section of a system prompt for a document-grounded assistant. Be specific to the domain and audience; do not invent facts, numbers or policies. Do not mention tools.
- role_statement: 1-2 sentences: who the assistant is and whom it serves.
- domain_guidance: 3-6 short rules about how to handle THIS domain's questions well. Every rule must be about things that actually appear in the manifest (its topics, entities, columns, figures). Do not write rules about item types the documents do not contain.
- style_rules: 2-4 rules on tone and format suited to the audience.

## USER
REQUIREMENTS:
{{requirements}}

DOCUMENT MANIFEST:
{{manifest}}

Return JSON {"role_statement": "...", "domain_guidance": ["..."], "style_rules": ["..."]}
