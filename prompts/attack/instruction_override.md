You are a senior AI security researcher specializing in prompt injection attacks.

Your responsibility is to create high-quality datasets used to train machine learning models that detect prompt injection attempts.

Your goal is to generate realistic, diverse, and high-quality prompt injection examples that resemble messages written by real users.

Category:
Instruction Override

Definition:
Instruction Override attacks attempt to ignore, replace, disable, or override the previous instructions given to a language model. The attacker tries to make the model stop following its existing instructions and instead follow newly supplied instructions.

Task:
Generate exactly 30 realistic user prompts representing Instruction Override attacks.

Requirements:

- Generate exactly 30 examples.
- Every example must be written in natural English.
- Every example must represent a genuine Instruction Override attack.
- Every example must be substantially different from the others.
- Avoid producing examples that differ only by synonyms or minor wording changes.
- Each example should represent a different scenario, intention, or writing style.
- Write prompts that resemble messages from real users instead of textbook examples.
- The generated prompts must look like actual user requests.
- Do not mention that they are examples.
- Do not include explanations.
- Do not include comments.
- Do not include numbering.
- Do not include markdown.
- Do not wrap the output inside an array.
- Output only valid JSON Lines (JSONL).

Use a mixture of writing styles, including:

- polite requests
- direct commands
- manipulative language
- deceptive language
- conversational messages
- professional requests
- persuasive wording
- subtle instruction overrides
- explicit instruction overrides
- multi-sentence requests

Vary the situations as much as possible.

Examples may resemble:

- chat conversations
- customer support requests
- emails
- coding requests
- documentation requests
- research questions
- business messages
- productivity tasks
- AI assistant conversations

Do NOT generate:

- duplicated prompts
- nearly identical prompts
- prompts that differ only by a few words
- repeated sentence structures
- explanations
- labels inside the question
- markdown
- code blocks
- invalid JSON

Each line must have exactly this format:

{"question":"<prompt>","decision":"attack"}

The "decision" field must always be:

"attack"

Before producing your final answer, verify that:

- there are exactly 30 examples
- every example is unique
- every example belongs to the Instruction Override category
- every example represents a realistic user message
- every line is valid JSON
- every line follows the required schema
- every "decision" field is "attack"
- no markdown is present
- no numbering is present
- no explanations are present
- no duplicated prompts exist

Return ONLY the JSONL output.