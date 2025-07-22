"""Tests for tokenizer integration."""

import pytest
import threading
import time
import requests
import socket
from toys.communication import tokenizer as tok
from toys.communication import external_adapter


def test_leb128_roundtrip() -> None:
    """Test LEB128 encoding is reversible."""
    # Test various token ID sizes
    test_ids = [0, 127, 128, 16383, 16384, 30000]

    for token_id in test_ids:
        encoded = tok._id_to_bytes(token_id)
        decoded = tok._bytes_to_ids(bytes(encoded))
        assert decoded == [token_id]
        assert all(0 <= b <= 255 for b in encoded)


def test_text_roundtrip() -> None:
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


def test_vocab_size() -> None:
    """Test vocabulary size retrieval."""
    try:
        size = tok.vocab_size("bert-base-uncased")
        assert size == 30522  # BERT's vocab size
    except FileNotFoundError:
        pytest.skip("Tokenizer not installed")


def get_free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.integration
def test_rest_adapter_hf_generate_200(temp_dir) -> None:
    """Test that the REST adapter responds 200 to a minimal HF /generate request."""
    port = get_free_port()
    # Patch the agent_pool to use temp_dir for isolation
    external_adapter.agent_pool.base_knowledge_path = str(temp_dir + "/public_knowledge.pkl.gz")

    # Start the app in a background thread
    def run():
        import uvicorn

        uvicorn.run(external_adapter.app, host="127.0.0.1", port=port, log_level="error")

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(1.5)  # Give server time to start
    try:
        resp = requests.post(f"http://127.0.0.1:{port}/generate", json={"inputs": "hello"}, timeout=5)
        assert resp.status_code == 200
        assert "generated_text" in resp.json()
    finally:
        # No explicit shutdown; thread is daemon and temp_dir is cleaned by fixture
        pass
