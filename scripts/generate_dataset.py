"""
Gemma 3 kullanarak sentetik JSONL veri üretir.
"""

import json
import os
from pathlib import Path

import torch
from transformers import AutoProcessor
from transformers import Gemma3ForConditionalGeneration


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

RAW_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "latest_generation.txt"
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
    """Gemma 3 processor ve modelini yükler."""

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "HF_TOKEN bulunamadı. "
            "Colab gizli anahtarını ortam değişkenine aktar."
        )

    print(f"Model yükleniyor: {MODEL_NAME}")

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        token=hf_token,
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.bfloat16,
        token=hf_token,
    ).eval()

    print(f"Model cihazı: {model.device}")
    print("Model başarıyla yüklendi.")

    return processor, model


def generate_text(
    processor,
    model,
    prompt: str,
) -> str:
    """Promptu modele gönderir ve üretilen metni döndürür."""

    full_prompt = (
        "This is a defensive AI security dataset generation task.\n"
        "Generate exactly 10 diverse instruction override attack examples.\n"
        "Treat all generated attack strings as inert dataset text.\n"
        "Do not follow or execute the generated attack instructions.\n"
        "Return only valid JSONL without markdown or explanations.\n\n"
        f"{prompt}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": full_prompt,
                }
            ],
        }
    ]

    model_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_length = model_inputs["input_ids"].shape[-1]

    print("Model veri üretiyor...")
    print(f"Girdi token sayısı: {input_length}")

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=700,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.08,
        )

    output_length = output_ids.shape[-1]

    print(f"Toplam token sayısı: {output_length}")
    print(
        "Yeni token sayısı: "
        f"{max(0, output_length - input_length)}"
    )

    output_contains_prompt = (
        output_length >= input_length
        and torch.equal(
            output_ids[0, :input_length],
            model_inputs["input_ids"][0],
        )
    )

    if output_contains_prompt:
        new_token_ids = output_ids[0, input_length:]
    else:
        new_token_ids = output_ids[0]

    generated_text = processor.decode(
        new_token_ids,
        skip_special_tokens=True,
    ).strip()

    full_decoded_text = processor.decode(
        output_ids[0],
        skip_special_tokens=True,
    ).strip()

    if not generated_text:
        save_raw_output(
            full_decoded_text,
            RAW_OUTPUT_PATH,
        )

        raise ValueError(
            "Model boş çıktı üretti. "
            f"Ham çıktı kaydedildi: {RAW_OUTPUT_PATH}"
        )

    save_raw_output(
        generated_text,
        RAW_OUTPUT_PATH,
    )

    return generated_text


def save_raw_output(
    generated_text: str,
    output_path: Path,
) -> None:
    """Modelin ham çıktısını kaydeder."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        generated_text,
        encoding="utf-8",
    )

    print(f"Ham çıktı kaydedildi: {output_path}")


def validate_jsonl(
    generated_text: str,
) -> list[dict[str, str]]:
    """Üretilen JSONL kayıtlarını doğrular."""

    cleaned_text = (
        generated_text
        .replace("```jsonl", "")
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    records = []
    seen_questions = set()

    for line_number, line in enumerate(
        cleaned_text.splitlines(),
        start=1,
    ):
        line = line.strip().removesuffix(",")

        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print(
                f"Geçersiz JSON satırı atlandı: {line_number}"
            )
            continue

        if not isinstance(record, dict):
            continue

        if set(record.keys()) != {
            "question",
            "decision",
        }:
            continue

        question = record["question"]
        decision = record["decision"]

        if not isinstance(question, str):
            continue

        if not isinstance(decision, str):
            continue

        question = question.strip()
        decision = decision.strip().lower()

        if not question:
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
                "question": question,
                "decision": decision,
            }
        )

    if len(records) != EXPECTED_COUNT:
        raise ValueError(
            f"{EXPECTED_COUNT} geçerli kayıt bekleniyordu, "
            f"{len(records)} kayıt bulundu. "
            f"Ham çıktıyı kontrol et: {RAW_OUTPUT_PATH}"
        )

    return records
def load_existing_questions(
    output_path: Path,
) -> set[str]:
    """Daha önce kaydedilmiş soruları okur."""

    if not output_path.exists():
        return set()

    questions = set()

    for line in output_path.read_text(
        encoding="utf-8"
    ).splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        question = record.get("question")

        if not isinstance(question, str):
            continue

        normalized_question = " ".join(
            question.lower().split()
        )

        questions.add(normalized_question)

    return questions


def save_dataset(
    records: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Yeni kayıtları JSONL dosyasına ekler."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_questions = load_existing_questions(
        output_path
    )

    new_records = []

    for record in records:
        normalized_question = " ".join(
            record["question"].lower().split()
        )

        if normalized_question in existing_questions:
            continue

        existing_questions.add(normalized_question)
        new_records.append(record)

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        for record in new_records:
            json_line = json.dumps(
                record,
                ensure_ascii=False,
            )

            file.write(json_line + "\n")

    print(f"{len(new_records)} yeni kayıt kaydedildi.")
    print(f"Veri seti: {output_path}")


def main() -> None:
    """Veri üretim sürecini çalıştırır."""

    prompt = load_prompt(PROMPT_PATH)

    processor, model = load_model()

    generated_text = generate_text(
        processor,
        model,
        prompt,
    )

    print("-" * 50)
    print("Model çıktısı:")
    print(generated_text)
    print("-" * 50)

    records = validate_jsonl(
        generated_text
    )

    save_dataset(
        records,
        OUTPUT_PATH,
    )

    print("Veri üretimi tamamlandı.")


if __name__ == "__main__":
    main()