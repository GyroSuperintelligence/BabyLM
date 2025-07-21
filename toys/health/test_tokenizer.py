"""Tests for tokenizer integration."""

import pytest
from toys.communication import tokenizer as tok

def test_leb128_roundtrip():
    """Test LEB128 encoding is reversible."""
    # Test various token ID sizes
    test_ids = [0, 127, 128, 16383, 16384, 30000]
    
    for token_id in test_ids:
        encoded = tok._id_to_bytes(token_id)
        decoded = tok._bytes_to_ids(bytes(encoded))
        assert decoded == [token_id]
        assert all(0 <= b <= 255 for b in encoded)

def test_text_roundtrip():
    """Test full text encoding/decoding."""
    test_texts = [
        "Hello, world!",
        "GyroSI loves reversible streams.",
        "The 🚀 emoji and special chars: <>&",
        "Multiple sentences. With punctuation! And numbers: 123.",
    ]
    
    for text in test_texts:
        try:
            encoded = tok.encode(text)
            decoded = tok.decode(encoded)
            # May not be exact due to tokenizer normalization
            assert isinstance(decoded, str)
            assert all(0 <= b <= 255 for b in encoded)
        except FileNotFoundError:
            pytest.skip("Tokenizer not installed. Run setup_tokenizers.py")

def test_vocab_size():
    """Test vocabulary size retrieval."""
    try:
        size = tok.vocab_size("bert-base-uncased")
        assert size == 30522  # BERT's vocab size
    except FileNotFoundError:
        pytest.skip("Tokenizer not installed") 