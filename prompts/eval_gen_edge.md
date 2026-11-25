## SYSTEM
You rewrite a test question into a harder EDGE CASE with the same correct answer, using the style requested:
- "vague": ambiguous / informal phrasing, missing the key noun, as a hurried user would type it;
- "typo": realistic typos, abbreviations and lowercase;
- "indirect": a scenario (for example "I joined 3 months ago and ...") that requires applying the rule to a situation;
- "negation": asks what is NOT allowed, or the exception to the rule.
The rewritten question must still be answerable from the same facts. Keep it to one user message.

## USER
STYLE: {{style}}
ORIGINAL QUESTION: {{question}}
FACTS THE ANSWER RELIES ON: {{facts}}

Return JSON {"question": "..."}
