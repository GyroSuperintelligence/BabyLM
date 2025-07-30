"""
Hugging Face tokenizer bridge for GyroSI.

Provides reversible text↔bytes encoding using LEB128 to ensure all bytes ≤ 255.
"""

from pathlib import Path
from typing import List, cast
from tokenizers import Tokenizer
import os


def get_tokenizer_root(base_path: Path = Path(__file__).resolve().parents[2]) -> Path:
    return (base_path / "memories/public/tokenizers").resolve()


# Module-level tokenizer cache keyed by (path, mtime)
_tokenizer_cache: dict[tuple[str, float], Tokenizer] = {}


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
            print(f"[TOK-DBG] raw blob: {blob.hex()}")
            print(f"[TOK-DBG] first 10 bytes: {[hex(b) for b in blob[:10]]}")
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


# ---------- helpers ----------

# Mask / unmask one byte stream -----------------------------------
_MASK = 0xAA


def _apply_mask(buf: bytes) -> bytes:
    """XOR every byte with 0xAA – vectorised & memory‑efficient."""
    # bytes ↔ introns is a pure involution: f(f(x)) == x
    return bytes(b ^ _MASK for b in buf)


# ---------- Public API ----------


def encode(text: str, name: str = "bert-base-uncased", base_path: Path = Path(__file__).resolve().parents[2]) -> bytes:
    """Encode text to bytes via tokenizer + LEB128 (vectorized). Uses base_path for root."""
    # 1. text → token IDs ----------------------------------------------------
    ids = _load(name, base_path).encode(text).ids

    # 2. IDs → pure‑LEB128 intron stream ------------------------------------
    introns = bytearray(len(ids) * 5)  # worst‑case pre‑alloc
    pos = 0
    for tid in ids:
        val = tid
        while True:
            byte = val & 0x7F
            val >>= 7
            introns[pos] = byte | (0x80 if val else 0x00)
            pos += 1
            if not val:
                break

    # 3. intron stream → external masked bytes ------------------------------
    return _apply_mask(bytes(introns[:pos]))


def decode(blob: bytes, name: str = "bert-base-uncased", base_path: Path = Path(__file__).resolve().parents[2]) -> str:
    """Decode LEB128 bytes back to text via tokenizer. Uses base_path for root."""
    # 0. Trim at EOS sentinel (remains valid after masking)
    if 0x00 in blob:
        blob = blob.split(b"\x00", 1)[0]

    # 1. external bytes → intron stream -------------------------------------
    introns = _apply_mask(blob)

    # 2. intron stream → token IDs ------------------------------------------
    try:
        ids = _bytes_to_ids(introns)
        # 3. IDs → text ------------------------------------------------------
        return cast(str, _load(name, base_path).decode(ids, skip_special_tokens=True))
    except Exception:
        # malformed stream fallback
        return blob.decode("utf-8", errors="replace")


def vocab_size(name: str = "bert-base-uncased", base_path: Path = Path(__file__).resolve().parents[2]) -> int:
    """Get vocabulary size of a tokenizer. Uses base_path for root."""
    return int(_load(name, base_path).get_vocab_size())


def bytes_from_ids(ids: list[int]) -> bytes:
    """Encode a list of token IDs to bytes via LEB128 and apply the 0xAA mask."""
    introns = bytearray(len(ids) * 5)  # worst-case pre-alloc
    pos = 0
    for tid in ids:
        val = tid
        while True:
            byte = val & 0x7F
            val >>= 7
            introns[pos] = byte | (0x80 if val else 0x00)
            pos += 1
            if not val:
                break
    return _apply_mask(bytes(introns[:pos]))


def id_to_bytes(tok_id: int) -> bytes:
    """Convert a single token ID to bytes via LEB128 and apply the 0xAA mask."""
    return bytes_from_ids([tok_id])


def bytes_to_id(bs: bytes) -> int:
    """Convert bytes back to a single token ID. Assumes complete token."""
    # First unmask the bytes, then decode
    unmasked = _apply_mask(bs)
    return _bytes_to_ids(unmasked)[0]


def bytes_to_ids(bs: bytes) -> List[int]:
    """Convert bytes back to multiple token IDs. Assumes complete tokens."""
    # First unmask the bytes, then decode
    unmasked = _apply_mask(bs)
    return _bytes_to_ids(unmasked)


# SEP token constant
SEP_ID = 102


def sep_bytes(count: int = 1) -> bytes:
    """Generate SEP token bytes for sentence/article boundaries."""
    return bytes_from_ids([SEP_ID] * count)


def encode_with_sep(
    text: str, name: str = "bert-base-uncased", base_path: Path = Path(__file__).resolve().parents[2]
) -> bytes:
    """Encode text and append a single SEP token."""
    return encode(text, name, base_path) + sep_bytes()


__all__ = [
    "encode",
    "decode",
    "vocab_size",
    "id_to_bytes",
    "bytes_to_id",
    "bytes_to_ids",
    "bytes_from_ids",
    "SEP_ID",
    "sep_bytes",
    "encode_with_sep",
]
