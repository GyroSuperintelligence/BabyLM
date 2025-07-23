"""Download and install pre-trained tokenizers."""

from pathlib import Path
from transformers import AutoTokenizer


def install_bert(base_path: Path = Path(__file__).resolve().parents[2]) -> None:
    """Download and save BERT tokenizer. base_path controls where 'memories/' is rooted."""
    root = base_path / "memories/public/tokenizers"
    name = "bert-base-uncased"
    dest = root / name
    dest.mkdir(parents=True, exist_ok=True)

    # Download from Hugging Face
    hf_tokenizer = AutoTokenizer.from_pretrained(name)  # type: ignore[no-untyped-call]

    # Save in fast tokenizer format
    hf_tokenizer.backend_tokenizer.save(str(dest / "tokenizer.json"))

    # Also save vocabulary for reference
    hf_tokenizer.save_vocabulary(str(dest))

    print(f"✓ Installed {name} to {dest}")
    print(f"  Vocab size: {len(hf_tokenizer.get_vocab())}")


def main() -> None:
    """Install all default tokenizers."""
    install_bert()
    # Add more tokenizers here as needed


if __name__ == "__main__":
    main()
