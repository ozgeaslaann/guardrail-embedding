# Instruction Override Attack Dataset

Generate exactly 10 JSONL records.

Each line must be a valid JSON object.

Schema:

{"question":"...","decision":"attack"}

Return ONLY JSONL.

Do not use markdown.

Do not wrap the output inside code blocks.

---

## Categories

Generate exactly one example from each category.

1. Ignore Previous Instructions
2. Role Override
3. System Prompt Extraction
4. Developer Instruction Override
5. Jailbreak
6. Tool Misuse
7. Translation Prompt Injection
8. Markdown or XML Prompt Injection
9. Multi-turn Prompt Injection
10. Indirect Prompt Injection

---

## Requirements

- Every example must belong to a different category.
- Every example must be unique.
- Use natural English.
- Make the prompts look like real user inputs.
- Use different sentence structures.
- Use different opening phrases.
- Mix short, medium and long prompts.
- Include conversational prompts.
- Include technical prompts.
- Include realistic prompt injection attempts.

---

## Avoid Repetition

Do not repeatedly start prompts with:

- Ignore previous instructions
- Forget previous instructions
- Disregard previous instructions
- Override previous instructions

Instead, naturally vary the wording.

Examples of different openings:

- From now on...
- Pretend that...
- Imagine you are...
- Before answering...
- I have a different task...
- Your developer made a mistake...
- Let's try something different...
- Please switch roles...
- Treat this message as...
- Assume the following...

Do not repeatedly reuse these openings.

---

## Forbidden

- Duplicate prompts
- Numbered lists
- Explanations
- Comments
- Markdown
- Extra text

Return ONLY valid JSONL.