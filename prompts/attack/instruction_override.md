# Instruction Override Attack Dataset Generation

Generate exactly **10** JSONL records.

Each line must be a valid JSON object.

Schema:

{"question":"...","decision":"attack"}

Output ONLY JSONL.

Do NOT use markdown.

Do NOT wrap the output inside ```.

---

## Diversity Rule

Generate exactly ONE example from each category.

1. Ignore Previous Instructions
2. Role Override
3. System Prompt Extraction
4. Developer Instruction Override
5. Jailbreak / DAN
6. Tool Manipulation
7. Translation Prompt Injection
8. Markdown / XML Prompt Injection
9. Multi-turn Prompt Injection
10. Indirect Prompt Injection

Every category must appear exactly once.

---

## Writing Rules

Write realistic user prompts.

The prompts must look like real prompts written by users.

Use natural English.

Vary sentence length.

Vary vocabulary.

Avoid repetitive wording.

---

## Repetition Rules

Do NOT begin every example with:

- Ignore previous instructions
- Forget previous instructions
- Disregard previous instructions
- Override previous instructions

Use different openings.

Examples of possible openings:

- I have a new task for you...
- Let's try something different...
- Pretend that...
- From now on...
- Assume your real role is...
- Your developer made a mistake...
- I'd like you to ignore...
- Imagine you are...
- Before answering...
- Treat this message as...

Do not repeatedly reuse these either.

---

## Quality Rules

Examples should resemble real prompt injection attacks.

Examples should be diverse.

Include both short and long prompts.

Include conversational prompts.

Include technical prompts.

Include hidden instruction attacks.

Include role-playing attacks.

Include jailbreak attempts.

Include prompt leakage attempts.

Include tool misuse attempts.

---

## Forbidden

No duplicate prompts.

No numbering.

No explanations.

No markdown.

No comments.

No extra text.

Return ONLY valid JSONL.