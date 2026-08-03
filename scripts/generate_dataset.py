"""
Gemma 3 kullanarak sentetik veri seti üretir.
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers import Gemma3ForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROMPT_PATH = (
    PROJECT_ROOT
    / "prompts"
    / "attack"
    / "instruction_override.md"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dataset.jsonl"
)

MODEL_NAME = "google/gemma-3-4b-it"
EXPECTED_COUNT = 10
EXPECTED_DECISION = "attack"


def load_prompt(prompt_path: Path) -> str:
    """Prompt dosyasını okur."""

    if not prompt_path.is_file():
        raise FileNotFoundError(
            f"Prompt dosyası bulunamadı: {prompt_path}"
        )

    prompt = prompt_path.read_text(
        encoding="utf-8"
    ).strip()

    if not prompt:
        raise ValueError("Prompt dosyası boş.")

    return prompt


def load_model():
    """Gemma 3 tokenizer ve modelini yükler."""

    print(f"Model yükleniyor: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=True,
    )

    model = Gemma3ForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.float16,
        token=True,
    )

    model.eval()

    print(f"Model cihazı: {model.device}")
    print("Model başarıyla yüklendi.")

    return tokenizer, model


def generate_text(
    tokenizer,
    model,
    prompt: str,
) -> str:
    """Promptu modele gönderir ve metin çıktısını döndürür."""

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You create high-quality AI security datasets. "
                        "Follow the requested JSONL format exactly."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        },
    ]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_length = model_inputs["input_ids"].shape[-1]

    print("Model veri üretiyor...")

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=700,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_length:]

    generated_text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    if not generated_text:
        raise ValueError("Model boş çıktı üretti.")

    return generated_text


def validate_jsonl(generated_text: str) -> list[dict]:
    """Model çıktısındaki geçerli JSONL kayıtlarını kontrol eder."""

    records = []
    seen_questions = set()

    for line in generated_text.splitlines():
        line = line.strip()

        if not line or line.startswith("```"):
            continue

        line = line.removesuffix(",")

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        if set(record) != {"question", "decision"}:
            continue

        question = record["question"]
        decision = record["decision"]

        if not isinstance(question, str) or not question.strip():
            continue

        if decision != EXPECTED_DECISION:
            continue

        normalized_question = " ".join(
            question.lower().split()
        )

        if normalized_question in seen_questions:
            continue

        seen_questions.add(normalized_question)

        records.append(
            {
                "question": question.strip(),
                "decision": decision,
            }
        )

    if len(records) != EXPECTED_COUNT:
        raise ValueError(
            f"{EXPECTED_COUNT} geçerli kayıt bekleniyordu, "
            f"{len(records)} kayıt bulundu."
        )

    return records


def save_dataset(
    records: list[dict],
    output_path: Path,
) -> None:
    """Doğrulanmış kayıtları JSONL dosyasına ekler."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_questions = set()

    if output_path.exists():
        for line in output_path.read_text(
            encoding="utf-8"
        ).splitlines():
            try:
                record = json.loads(line)
                question = record.get("question", "")
                existing_questions.add(
                    " ".join(question.lower().split())
                )
            except json.JSONDecodeError:
                continue

    new_records = [
        record
        for record in records
        if " ".join(
            record["question"].lower().split()
        ) not in existing_questions
    ]

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        for record in new_records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"{len(new_records)} yeni kayıt kaydedildi.")
    print(f"Dosya: {output_path}")


def main() -> None:
    """Veri üretim sürecini çalıştırır."""

    prompt = load_prompt(PROMPT_PATH)
    tokenizer, model = load_model()

    generated_text = generate_text(
        tokenizer,
        model,
        prompt,
    )

    print("-" * 50)
    print("Model çıktısı:")
    print(generated_text)
    print("-" * 50)

    records = validate_jsonl(generated_text)

    save_dataset(
        records,
        OUTPUT_PATH,
    )

    print("Veri üretimi tamamlandı.")


if __name__ == "__main__":
    main()