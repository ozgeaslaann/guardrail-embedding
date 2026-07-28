"""
Dataset generation script.

This script:
1. Loads a generation prompt.
2. Loads the language model.
3. Generates synthetic examples.
4. Validates JSONL output.
5. Saves the dataset.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROMPT_PATH = (
    PROJECT_ROOT
    / "prompts"
    / "attack_instruction_override.md"
)


def load_prompt(prompt_path: Path) -> str:
    """Load a prompt file and return its contents."""
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file was not found: {prompt_path}"
        )

    if not prompt_path.is_file():
        raise ValueError(
            f"Prompt path is not a file: {prompt_path}"
        )

    prompt = prompt_path.read_text(encoding="utf-8").strip()

    if not prompt:
        raise ValueError(
            f"Prompt file is empty: {prompt_path}"
        )

    return prompt


def load_model():
    """Load the language model."""
    pass


def generate_examples():
    """Generate dataset examples."""
    pass


def validate_jsonl():
    """Validate generated JSONL records."""
    pass


def save_dataset():
    """Save validated examples into dataset.jsonl."""
    pass


def main() -> None:
    """Run the dataset generation workflow."""
    prompt = load_prompt(PROMPT_PATH)

    print("Prompt loaded successfully.")
    print("-" * 50)
    print(prompt)


if __name__ == "__main__":
    main()