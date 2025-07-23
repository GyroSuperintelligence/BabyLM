"""
Hugging Face tokenizer bridge for GyroSI.

Provides reversible text↔bytes encoding using LEB128 to ensure all bytes ≤ 255.
"""

from pathlib import Path
from typing import List, Dict, Any, cast
from tokenizers import Tokenizer
import os


def get_tokenizer_root(base_path: Path = Path(__file__).resolve().parents[2]) -> Path:
    return (base_path / "memories/public/tokenizers").resolve()


# Module-level tokenizer cache keyed by (name, mtime)
_tokenizer_cache: Dict[Any, Any] = {}


def _load(name: str = "bert-base-uncased", base_path: Path = Path(__file__).resolve().parents[2]) -> Tokenizer:
    """Load and cache a tokenizer from disk, auto-reload if file changes. Uses base_path for root."""
    path = get_tokenizer_root(base_path) / name / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer '{name}' not installed at {path}. Run: python toys/communication/setup_tokenizers.py"
        )
    mtime = os.path.getmtime(path)
    cache_key = (str(path), mtime)
    if cache_key in _tokenizer_cache:
        return _tokenizer_cache[cache_key]
    tokenizer = Tokenizer.from_file(str(path))
    _tokenizer_cache.clear()  # Only keep one loaded at a time (intentional)
    _tokenizer_cache[cache_key] = tokenizer
    return tokenizer


# ---------- LEB128 encoding ----------
def _id_to_bytes(idx: int) -> List[int]:
    """Convert token ID to variable-length bytes."""
    if idx < 0:
        raise ValueError(f"Token ID must be non-negative, got {idx}")

    out: List[int] = []
    while True:
        byte = idx & 0x7F
        idx >>= 7
        if idx:
            out.append(byte | 0x80)  # set continuation bit
        else:
            out.append(byte)
            break
    return out


def _bytes_to_ids(blob: bytes) -> List[int]:
    """Decode LEB128 bytes back to token IDs.
    The overflow guard (shift > 28) assumes token IDs are at most 32 bits, which is true for HuggingFace tokenizers.
    """
    ids, cur, shift = [], 0, 0
    for i, b in enumerate(blob):
        if shift > 28:  # Prevent overflow (32-bit token ID assumption)
            raise ValueError(f"Token ID too large at byte {i}")
        cur |= (b & 0x7F) << shift
        if b & 0x80:
            shift += 7
        else:
            ids.append(cur)
            cur, shift = 0, 0
    if shift:
        raise ValueError("Incomplete token ID sequence")
    return ids


# ---------- Public API ----------
def encode(text: str, name: str = "bert-base-uncased", base_path: Path = Path(__file__).resolve().parents[2]) -> bytes:
    """Encode text to bytes via tokenizer + LEB128 (vectorized). Uses base_path for root."""
    ids = _load(name, base_path).encode(text).ids
    # Estimate max output size: each id can take up to 5 bytes (for 32-bit int)
    out = bytearray(len(ids) * 5)
    pos = 0
    for i in ids:
        val = i
        while True:
            b = val & 0x7F
            val >>= 7
            if val:
                out[pos] = b | 0x80
                pos += 1
            else:
                out[pos] = b
                pos += 1
                break
    return bytes(out[:pos])


def decode(blob: bytes, name: str = "bert-base-uncased", base_path: Path = Path(__file__).resolve().parents[2]) -> str:
    """Decode LEB128 bytes back to text via tokenizer. Uses base_path for root."""
    try:
        ids = _bytes_to_ids(blob)
        return cast(str, _load(name, base_path).decode(ids, skip_special_tokens=True))
    except Exception:
        # Fallback to UTF-8 if tokenizer decode fails
        return blob.decode("utf-8", errors="replace")


def vocab_size(name: str = "bert-base-uncased", base_path: Path = Path(__file__).resolve().parents[2]) -> int:
    """Get vocabulary size of a tokenizer. Uses base_path for root."""
    return int(_load(name, base_path).get_vocab_size())
