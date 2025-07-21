"""
Hugging Face tokenizer bridge for GyroSI.

Provides reversible text↔bytes encoding using LEB128 to ensure all bytes ≤ 255.
"""

from pathlib import Path
from functools import lru_cache
from typing import List
from tokenizers import Tokenizer

_ROOT = Path("memories/public/tokenizers")

@lru_cache(maxsize=8)
def _load(name: str = "bert-base-uncased") -> Tokenizer:
    """Load and cache a tokenizer from disk."""
    path = _ROOT / name / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer '{name}' not installed at {path}. "
            "Run: python toys/communication/setup_tokenizers.py"
        )
    return Tokenizer.from_file(str(path))

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
    """Decode LEB128 bytes back to token IDs."""
    ids, cur, shift = [], 0, 0
    for i, b in enumerate(blob):
        if shift > 28:  # Prevent overflow
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
def encode(text: str, name: str = "bert-base-uncased") -> bytes:
    """Encode text to bytes via tokenizer + LEB128."""
    ids = _load(name).encode(text).ids
    flat: List[int] = []
    for i in ids:
        flat.extend(_id_to_bytes(i))
    return bytes(flat)

def decode(blob: bytes, name: str = "bert-base-uncased") -> str:
    """Decode LEB128 bytes back to text via tokenizer."""
    try:
        ids = _bytes_to_ids(blob)
        return _load(name).decode(ids, skip_special_tokens=True)
    except Exception:
        # Fallback to UTF-8 if tokenizer decode fails
        return blob.decode("utf-8", errors="replace")

def vocab_size(name: str = "bert-base-uncased") -> int:
    """Get vocabulary size of a tokenizer."""
    return _load(name).get_vocab_size()