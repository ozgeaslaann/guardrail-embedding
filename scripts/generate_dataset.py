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

    print("Model başarıyla yüklendi.")

    return tokenizer, model


def generate_examples(
    tokenizer,
    model,
    prompt: str,
):
    """Promptu modelin anlayacağı giriş biçimine dönüştürür."""

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

    print("Prompt başarıyla tokenlara dönüştürüldü.")
    print("-" * 50)
    print("Girdi tensor boyutu:")
    print(model_inputs["input_ids"].shape)


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

    generate_examples(
        tokenizer,
        model,
        prompt,
    )

    print("-" * 50)
    print("Yüklenen tokenizer:")
    print(tokenizer.__class__.__name__)

    print("-" * 50)
    print("Yüklenen model:")
    print(model.__class__.__name__)

    print("-" * 50)
    print("Kullanılan prompt:")
    print(prompt)


if __name__ == "__main__":
    main()