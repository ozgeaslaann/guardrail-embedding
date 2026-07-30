"""
Veri seti üretim scripti.

Bu script:
1. Veri üretim promptunu yükler.
2. Dil modelini yükler.
3. Sentetik örnekler üretir.
4. JSONL çıktısını doğrular.
5. Veri setini kaydeder.
"""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROMPT_PATH = (
    PROJECT_ROOT
    / "prompts"
    / "attack"
    / "instruction_override.md"
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_prompt(prompt_path: Path) -> str:
    """Prompt dosyasını okur ve içeriğini metin olarak döndürür."""

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt dosyası bulunamadı: {prompt_path}"
        )

    if not prompt_path.is_file():
        raise ValueError(
            f"Verilen prompt yolu bir dosya değil: {prompt_path}"
        )

    prompt = prompt_path.read_text(
        encoding="utf-8"
    ).strip()

    if not prompt:
        raise ValueError(
            f"Prompt dosyası boş: {prompt_path}"
        )

    return prompt


def load_model():
    """Tokenizer'ı ve dil modelini yükler."""

    print(f"Model yükleniyor: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )

    model.eval()

    print(f"Model cihazı: {model.device}")
    print(f"CUDA kullanılabilir mi?: {torch.cuda.is_available()}")


    print("Model başarıyla yüklendi.")

    return tokenizer, model


def generate_examples(
    tokenizer,
    model,
    prompt: str,
) -> str:
    """Promptu modele gönderir ve üretilen metni döndürür."""

    print("Veri üretimine başlanıyor...")

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_inputs = model_inputs.to(model.device)

    print("Prompt başarıyla tokenlara dönüştürüldü.")
    print("-" * 50)
    print("Girdi tensor boyutu:")
    print(model_inputs["input_ids"].shape)

    input_token_count = model_inputs["input_ids"].shape[-1]

    print("-" * 50)
    print("Model çıktı üretiyor...")

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_token_ids = generated_ids[
        0,
        input_token_count:
    ]

    generated_text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
    ).strip()

    if not generated_text:
        raise ValueError(
            "Model boş bir çıktı üretti."
        )

    print("Model çıktıyı başarıyla üretti.")
    print("-" * 50)
    print("Üretilen çıktı:")
    print(generated_text)

    return generated_text


def validate_jsonl():
    """Üretilen JSONL kayıtlarını doğrular."""
    pass


def save_dataset():
    """Doğrulanmış örnekleri dataset.jsonl dosyasına kaydeder."""
    pass


def main() -> None:
    """Veri seti üretim sürecini çalıştırır."""

    prompt = load_prompt(PROMPT_PATH)

    print("Prompt başarıyla yüklendi.")

    tokenizer, model = load_model()

    generated_text = generate_examples(
        tokenizer,
        model,
        prompt,
    )

    print("-" * 50)
    print("Üretim süreci tamamlandı.")
    print(f"Üretilen metin uzunluğu: {len(generated_text)} karakter")


if __name__ == "__main__":
    main()