You are an expert AI security dataset creator.

Your task is to generate realistic prompt injection examples.

Category:
Instruction Override

Requirements:
- Generate exactly 30 examples.
- All examples must be written in natural English.
- Every example must represent an Instruction Override attack.
- Each example must be substantially different from the others.
- Avoid producing examples that differ only by wording.
- Do not include explanations.
- Do not include numbering.
- Do not include markdown.
- Output only valid JSON Lines (JSONL).

Each line must have exactly this format:

{"question":"<prompt>","decision":"attack"}