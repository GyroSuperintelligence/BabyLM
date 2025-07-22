"""Train custom tokenizers on domain-specific data."""

from pathlib import Path

from tokenizers import Tokenizer, trainers, pre_tokenizers
from tokenizers.models import WordPiece


def train_wordpiece(files: list[str], vocab_size: int = 30000, save_name: str = "custom-wordpiece") -> None:
    """Train a new WordPiece tokenizer on provided files."""

    # Initialize tokenizer with WordPiece model
    tokenizer = Tokenizer(WordPiece(vocab={}, unk_token="[UNK]", max_input_chars_per_word=100))

    # Pre-tokenization (BERT-style)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()  # pyright: ignore[reportAttributeAccessIssue]

    # Training configuration
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=2,
        show_progress=True,
    )

    # Train
    tokenizer.train(files, trainer)

    # Save
    save_dir = Path(f"memories/public/tokenizers/{save_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_dir / "tokenizer.json"))

    print(f"✓ Trained tokenizer saved to {save_dir}")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        train_wordpiece(sys.argv[1:])
    else:
        print("Usage: python train_tokenizer.py file1.txt file2.txt ...")
