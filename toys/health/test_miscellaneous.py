import pytest
from pathlib import Path
from toys.communication import tokenizer as gyrotok
from baby.policies import OrbitStore
import tempfile
from baby.intelligence import GyroSI
from baby.contracts import AgentConfig


def test_debug_wiki_test_tokenizer():
    """Debug test: print wiki_test content, tokenize, print tokens/bytes, decode, print first tokens as words."""
    test_file = Path(__file__).parent.parent / "training" / "wiki_test"
    assert test_file.exists(), f"Test file not found: {test_file}"

    # Read file content
    text = test_file.read_text(encoding="utf-8")
    print("\n--- Raw file content ---\n", text)

    # Tokenize
    tokenizer_name = "bert-base-uncased"
    tok = gyrotok._load(tokenizer_name)
    encoding = tok.encode(text)
    token_ids = encoding.ids
    print("\n--- Token IDs ---\n", token_ids)

    # Encode to bytes (GyroSI format)
    encoded_bytes = gyrotok.encode(text, name=tokenizer_name)
    print("\n--- Encoded bytes (masked) ---\n", list(encoded_bytes))

    # Decode back to text
    decoded_text = gyrotok.decode(encoded_bytes, name=tokenizer_name)
    print("\n--- Decoded text ---\n", decoded_text)

    # Print first 1-2 tokens as words (if possible)
    if token_ids:
        first_token = tok.decode([token_ids[0]], skip_special_tokens=True)
        print(f"\nFirst token as word: '{first_token}' (id: {token_ids[0]})")
        if len(token_ids) > 1:
            second_token = tok.decode([token_ids[1]], skip_special_tokens=True)
            print(f"Second token as word: '{second_token}' (id: {token_ids[1]})")

    # Basic checks
    assert isinstance(decoded_text, str)
    assert len(token_ids) > 0
    assert len(encoded_bytes) > 0


def test_print_wikipedia_knowledge_bin_entries():
    """Print the first 3 entries from the wikipedia_knowledge.bin file for inspection."""
    knowledge_path = Path(__file__).parent.parent / "training" / "knowledge" / "wikipedia_knowledge.bin"
    assert knowledge_path.exists(), f"Knowledge file not found: {knowledge_path}"

    store = OrbitStore(str(knowledge_path), append_only=True)
    print("\n--- First 3 entries in wikipedia_knowledge.bin ---")
    for i, (context_key, entry) in enumerate(store.iter_entries()):
        print(f"Entry {i+1}:")
        print(f"  context_key: {context_key}")
        print(f"  entry: {entry}")
        if i >= 2:
            break
    store.close()


def test_agent_with_wikipedia_knowledge_speaks():
    """Create an isolated agent with wikipedia_knowledge.bin as public knowledge and see if it speaks."""
    from pathlib import Path
    from toys.communication import tokenizer as gyrotok
    from baby.policies import OrbitStore

    # Paths
    knowledge_path = Path(__file__).parent.parent / "training" / "knowledge" / "wikipedia_knowledge.bin"
    ontology_path = Path(__file__).parent.parent.parent / "memories" / "public" / "meta" / "ontology_keys.npy"
    phenomenology_path = Path(__file__).parent.parent.parent / "memories" / "public" / "meta" / "phenomenology_map.npy"
    tokenizer_name = "bert-base-uncased"

    # Create a temp dir for private knowledge
    with tempfile.TemporaryDirectory() as tmpdir:
        private_knowledge_path = Path(tmpdir) / "private_knowledge.bin"
        config: AgentConfig = {
            "ontology_path": str(ontology_path),
            "public_knowledge_path": str(knowledge_path),
            "private_knowledge_path": str(private_knowledge_path),
            "phenomenology_map_path": str(phenomenology_path),
            "base_path": str(tmpdir),
            "tokenizer_name": tokenizer_name,
            "preferences": {"tokenizer": {"name": tokenizer_name}},
        }
        agent = GyroSI(config, agent_id="test_agent", base_path=Path(tmpdir))

        # Prepare a simple prompt
        prompt = "What is an algorithm?"
        encoded = gyrotok.encode(prompt, name=tokenizer_name)
        response_bytes = agent.respond(encoded, max_new_tokens=64)
        response_text = gyrotok.decode(response_bytes, name=tokenizer_name)
        print("\n--- Agent response (decoded) ---\n", response_text)
        agent.close()
