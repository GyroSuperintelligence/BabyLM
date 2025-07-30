from pathlib import Path
from typing import List
from toys.communication import tokenizer as gyrotok
from baby.policies import OrbitStore
import tempfile
from baby.intelligence import GyroSI
from baby.contracts import AgentConfig


def test_print_simple_wikipedia_knowledge_bin_entries() -> None:
    """Print the first 5 entries from the simple_wikipedia_1.bin file for inspection."""
    knowledge_path = Path(__file__).parent.parent / "training" / "knowledge" / "simple_wikipedia_1.bin"
    assert knowledge_path.exists(), f"Knowledge file not found: {knowledge_path}"

    store = OrbitStore(str(knowledge_path), append_only=True)
    print("\n--- First 5 entries in simple_wikipedia_1.bin ---")
    for i, (key, entry) in enumerate(store.iter_entries()):
        print(f"Entry {i+1}:")
        print(f"  key: {key}")  # key is (state_index, token_id)
        print(f"  entry keys: {list(entry.keys())}")
        if "key" in entry:
            print(f"  key: {entry['key']}")
        if "mask" in entry:
            print(f"  mask: {entry['mask']}")
        if "conf" in entry:
            print(f"  conf: {entry['conf']}")
        print()
        if i >= 4:  # Show first 5 entries
            break
    store.close()


def test_complete_metadata_analysis() -> None:
    """Display all metadata fields from the knowledge file entries."""
    knowledge_path = Path(__file__).parent.parent / "training" / "knowledge" / "simple_wikipedia_1.bin"
    assert knowledge_path.exists(), f"Knowledge file not found: {knowledge_path}"

    store = OrbitStore(str(knowledge_path), append_only=True)
    print("\n--- Complete Metadata Analysis of simple_wikipedia_1.bin ---")

    for i, (key, entry) in enumerate(store.iter_entries()):
        print(f"\nEntry {i+1} - Complete Metadata:")
        print(f"  Key: {key}")  # key is (state_index, token_id)
        print(f"  All available fields: {list(entry.keys())}")

        # Display each field with its value
        for field_name, field_value in entry.items():
            if field_name in ["key", "mask", "conf"]:
                print(f"  {field_name}: {field_value}")
            else:
                print(f"  {field_name}: {field_value}")

        print("-" * 50)

    store.close()


def test_simple_wikipedia_knowledge_file_stats() -> None:
    """Get statistics about the simple_wikipedia_1.bin file."""
    knowledge_path = Path(__file__).parent.parent / "training" / "knowledge" / "simple_wikipedia_1.bin"
    assert knowledge_path.exists(), f"Knowledge file not found: {knowledge_path}"

    store = OrbitStore(str(knowledge_path), append_only=True)

    total_entries = 0
    conf_sum = 0.0
    phenotype_samples: List[str] = []

    for key, entry in store.iter_entries():
        total_entries += 1
        if "mask" in entry:
            mask_value = entry["mask"]
            if len(phenotype_samples) < 3:  # Keep first 3 mask samples
                phenotype_samples.append(f"mask: {mask_value}")
        if "conf" in entry:
            conf_sum += entry["conf"]

    avg_conf = conf_sum / total_entries if total_entries > 0 else 0

    print("\n--- simple_wikipedia_1.bin Statistics ---")
    print(f"Total entries: {total_entries}")
    print(f"Average conf: {avg_conf:.3f}")
    print(f"File size: {knowledge_path.stat().st_size:,} bytes")

    if phenotype_samples:
        print("\nSample mask entries:")
        for i, sample in enumerate(phenotype_samples, 1):
            print(f"  {i}. {sample}")

    store.close()


def test_agent_with_simple_wikipedia_knowledge_speaks() -> None:
    """Create an isolated agent with simple_wikipedia_1.bin as public knowledge and see if it speaks."""
    # Paths
    knowledge_path = Path(__file__).parent.parent / "training" / "knowledge" / "simple_wikipedia_1.bin"
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
            "preferences": {"tokenizer": {"name": tokenizer_name}},  # type: ignore
        }
        agent = GyroSI(config, agent_id="test_agent", base_path=Path(tmpdir))

        # Test multiple prompts to see if the agent can respond based on the knowledge
        test_prompts = [
            "What is an algorithm?",
            "Tell me about science.",
            "What is mathematics?",
            "Explain technology.",
            "What is history?",
        ]

        print("\n--- Testing agent with simple_wikipedia_1.bin knowledge ---")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nPrompt {i}: {prompt}")
            encoded = gyrotok.encode(prompt, name=tokenizer_name)
            response_bytes = agent.respond(encoded, max_new_tokens=64)
            response_text = gyrotok.decode(response_bytes, name=tokenizer_name)
            print(f"Response: {response_text}")

        agent.close()


def test_knowledge_retrieval_from_simple_wikipedia() -> None:
    """Test if we can retrieve specific knowledge entries from the simple_wikipedia_1.bin file."""
    knowledge_path = Path(__file__).parent.parent / "training" / "knowledge" / "simple_wikipedia_1.bin"
    assert knowledge_path.exists(), f"Knowledge file not found: {knowledge_path}"

    store = OrbitStore(str(knowledge_path), append_only=True)

    # Get all entries to analyze
    entries = list(store.iter_entries())
    print("\n--- Knowledge Retrieval Test ---")
    print(f"Total entries available: {len(entries)}")

    # Show some sample context keys and their associated content
    print("\nSample knowledge entries:")
    for i, (context_key, entry) in enumerate(entries[:5]):
        print(f"\nEntry {i+1}:")
        print(f"  Context key: {context_key}")
        if "phenotype" in entry:
            phenotype_preview = (
                str(entry["phenotype"])[:150] + "..." if len(str(entry["phenotype"])) > 150 else str(entry["phenotype"])
            )
            print(f"  Phenotype: {phenotype_preview}")
        if "conf" in entry:
            print(f"  Conf: {entry['conf']:.3f}")
        if "usage_count" in entry:
            print(f"  Usage count: {entry['usage_count']}")

    store.close()
