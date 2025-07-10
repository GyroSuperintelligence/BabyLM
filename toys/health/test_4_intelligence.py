"""
Tests for the intelligence engine, learning state, and thread processing in the BabyLM system.
Covers format management, input processing, response generation, and thread statistics.
"""

import os
import uuid
import json
import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from typing import cast

# Import modules from baby package
from baby.inference import InferenceEngine
from baby.information import (
    ensure_agent_uuid,
    assign_agent_uuid,
    create_thread,
    save_thread,
    load_thread,
    store_thread_key,
    load_thread_key,
    store_gene_keys,
    load_gene_keys,
    parent,
    children,
    list_formats,
    load_format,
    store_format,
    get_memory_preferences,
    shard_path,
    update_registry,
    atomic_write,
    PatternIndex,
)
from baby.intelligence import IntelligenceEngine, weighted_choice, initialize_intelligence_engine
from baby.governance import derive_canonical_patterns
from baby.types import ThreadMetadata


# Helper to coerce any value to a flat list of strings
def to_flat_str_list(val) -> list[str]:
    if isinstance(val, list):
        flat = []
        for v in val:
            if isinstance(v, str):
                flat.append(v)
            elif isinstance(v, list):
                flat.extend([x for x in v if isinstance(x, str)])
        return flat
    elif isinstance(val, str):
        return [val]
    else:
        return []


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------


@pytest.fixture
def isolated_test_env():
    """
    Creates a completely isolated test environment with proper cleanup.
    This fixture ensures no test pollution and proper cleanup of all files.
    """
    # Create a temporary directory that will be automatically cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Store original working directory
        original_cwd = Path.cwd()

        # Change to temporary directory
        os.chdir(temp_path)

        # Create necessary directory structure
        memories_dir = temp_path / "memories"
        baby_dir = temp_path / "baby"

        # Create subdirectories
        (memories_dir / "public" / "masks").mkdir(parents=True, exist_ok=True)
        (memories_dir / "public" / "formats").mkdir(parents=True, exist_ok=True)
        (memories_dir / "public" / "threads").mkdir(parents=True, exist_ok=True)
        (memories_dir / "public" / "keys").mkdir(parents=True, exist_ok=True)
        (memories_dir / "private" / "agents").mkdir(parents=True, exist_ok=True)
        baby_dir.mkdir(exist_ok=True)

        # Create memory_preferences.json
        mem_prefs = {
            "sharding": {"width": 2, "max_files": 30000, "second_level": True},
            "storage_config": {"max_thread_size_mb": 64, "encryption_algorithm": "AES-256-GCM"},
            "format_config": {"default_cgm_version": "1.0.0", "max_character_label_length": 128},
        }
        with open(memories_dir / "memory_preferences.json", "w") as f:
            json.dump(mem_prefs, f, indent=2)

        try:
            yield temp_path
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
            # temp_dir is automatically cleaned up by tempfile.TemporaryDirectory


@pytest.fixture
def test_masks(isolated_test_env):
    """Create the required mask files for InferenceEngine."""
    # Create canonical patterns
    patterns_array, _ = derive_canonical_patterns()
    patterns_array.tofile("memories/public/masks/epigenome.dat")

    # Create genome mask (identity mapping for predictability)
    genome_mask = np.arange(256, dtype=np.uint8)
    genome_mask.tofile("memories/public/masks/genome.dat")

    return isolated_test_env


@pytest.fixture
def initialized_intelligence_engine(test_masks):
    """
    Creates a fully initialized IntelligenceEngine in private mode with proper setup.
    """
    # Define predictable agent UUID and secret
    agent_uuid = "11111111-1111-1111-1111-111111111111"
    agent_secret = "test-secret-for-fixture"

    # Create baby preferences
    baby_prefs = {
        "agent_secret": agent_secret,
        "log_level": "info",
        "response_length": 100,
        "learning_rate": 1.0,
    }
    with open("baby/baby_preferences.json", "w") as f:
        json.dump(baby_prefs, f, indent=2)

    # CRITICAL FIX: Create agent directory structure BEFORE initializing engine
    assign_agent_uuid(agent_uuid)

    # Initialize engine
    engine = initialize_intelligence_engine(agent_uuid=agent_uuid, agent_secret=agent_secret)

    # Verify proper initialization
    assert engine.agent_uuid == agent_uuid
    assert engine.agent_secret == agent_secret

    # Verify directories exist
    private_dir = Path("memories/private/agents")
    agent_shard = shard_path(private_dir, agent_uuid)
    agent_dir = agent_shard / f"agent-{agent_uuid}"
    assert agent_dir.exists(), f"Agent directory was not created: {agent_dir}"
    assert (agent_dir / "threads").exists(), f"Threads directory missing: {agent_dir / 'threads'}"
    assert (agent_dir / "keys").exists(), f"Keys directory missing: {agent_dir / 'keys'}"

    return engine


@pytest.fixture
def public_intelligence_engine(test_masks):
    """Creates an IntelligenceEngine in public/curation mode."""
    # Initialize in public mode (no agent_uuid or agent_secret)
    engine = initialize_intelligence_engine(agent_uuid=None, agent_secret=None)

    # Verify public mode
    assert engine.agent_uuid is None
    assert engine.agent_secret is None

    return engine


# ------------------------------------------------------------------------------
# Intelligence (S4) Tests
# ------------------------------------------------------------------------------


class TestIntelligence:
    """Tests for the Intelligence layer (orchestration)"""

    def test_intelligence_engine_init(self, initialized_intelligence_engine):
        """Test IntelligenceEngine initialization"""
        engine = initialized_intelligence_engine

        assert isinstance(engine.agent_uuid, str)
        assert isinstance(engine.format_uuid, str)
        assert engine.thread_uuid is None
        assert engine.thread_file_key is None
        assert isinstance(engine.M, dict)
        assert isinstance(engine.memory_prefs, dict)
        assert hasattr(engine, "pattern_index")
        assert engine.pattern_index is not None  # Should exist in private mode

    def test_load_or_init_formats_existing(self, initialized_intelligence_engine):
        """Test loading existing format metadata"""
        engine = initialized_intelligence_engine

        # Create format directory and file
        formats_dir = Path("memories/public/formats")
        format_shard = shard_path(formats_dir, engine.format_uuid)
        format_shard.mkdir(parents=True, exist_ok=True)

        format_path = format_shard / f"format-{engine.format_uuid}.json"
        mock_format = {
            "format_uuid": engine.format_uuid,
            "format_name": "test_format",
            "patterns": [],
        }

        with open(format_path, "w") as f:
            json.dump(mock_format, f)

        # Test loading
        result = engine._load_or_init_formats()

        # Verify loaded format
        assert result.get("format_uuid") == engine.format_uuid
        assert result.get("format_name") == "test_format"

    def test_load_or_init_formats_new(self, initialized_intelligence_engine):
        """Test initializing new format metadata"""
        engine = initialized_intelligence_engine

        # Ensure format file doesn't exist
        formats_dir = Path("memories/public/formats")
        format_shard = shard_path(formats_dir, engine.format_uuid)
        format_path = format_shard / f"format-{engine.format_uuid}.json"
        if format_path.exists():
            format_path.unlink()

        # Test initialization
        with patch.object(
            engine,
            "_initialize_format_metadata",
            return_value={"format_uuid": engine.format_uuid, "test": True},
        ):
            result = engine._load_or_init_formats()

            # Verify initialized format
            assert result.get("format_uuid") == engine.format_uuid
            assert result.get("test") is True

    def test_initialize_format_metadata(self, initialized_intelligence_engine):
        """Test format metadata initialization"""
        engine = initialized_intelligence_engine

        # Initialize new format metadata
        result = engine._initialize_format_metadata()

        # Verify structure
        assert result.get("format_uuid") == engine.format_uuid
        assert "patterns" in result
        assert len(result.get("patterns")) == 256
        assert "cgm_policies" in result

        # Check all required policies
        required_policies = ["governance", "information", "inference", "intelligence"]
        for policy in required_policies:
            assert policy in result.get("cgm_policies")

    def test_start_new_thread(self, initialized_intelligence_engine):
        """Test starting a new thread"""
        engine = initialized_intelligence_engine

        with patch.object(engine, "_derive_file_key", return_value=b"test_key" * 32):
            # Start a new thread
            thread_uuid = engine.start_new_thread()

            # Verify thread initialization
            assert thread_uuid == engine.thread_uuid
            assert engine.thread_file_key == b"test_key" * 32
            assert engine.current_thread_keys == []

            # Check that thread metadata file exists
            private_dir = Path("memories/private/agents")
            agent_shard = shard_path(private_dir, engine.agent_uuid)
            agent_dir = agent_shard / f"agent-{engine.agent_uuid}"
            threads_dir = agent_dir / "threads"
            thread_shard = shard_path(threads_dir, thread_uuid)
            thread_meta_path = thread_shard / f"thread-{thread_uuid}.json"
            assert thread_meta_path.exists()

            # Check that key file exists
            keys_dir = agent_dir / "keys"
            key_shard = shard_path(keys_dir, thread_uuid)
            key_path = key_shard / f"key-{thread_uuid}.bin.enc"
            assert key_path.exists()

    def test_process_input_stream(self, initialized_intelligence_engine):
        """Test processing an input stream"""
        engine = initialized_intelligence_engine
        test_input = b"Test input stream"
        import base64

        expected_event = {"type": "input", "data": base64.b64encode(test_input).decode("utf-8")}

        with (
            patch.object(
                engine.information_engine,
                "process_stream",
                return_value=(b"ciphertext", b"keystream"),
            ),
            patch.object(engine, "_append_to_thread") as mock_append,
        ):
            # Process input stream
            plaintext, ciphertext = engine.process_input_stream(test_input)

            # Verify results
            assert plaintext == test_input
            assert ciphertext == b"ciphertext"

            # Verify method calls
            engine.information_engine.process_stream.assert_called_once()
            mock_append.assert_called_once_with(expected_event)

    def test_end_current_thread(self, initialized_intelligence_engine):
        """Test ending a current thread"""
        engine = initialized_intelligence_engine

        # Set up thread state
        engine.thread_uuid = "test-thread-uuid"
        engine.thread_file_key = b"key" * 64  # 256 bytes
        engine.current_thread_keys = [{"cycle": 1, "pattern_index": 42}]

        # Ensure format metadata is present
        if "metadata" not in engine.M:
            engine.M["metadata"] = {}
        engine.M["metadata"].setdefault("last_updated", "2024-01-01T00:00:00")
        engine.M["metadata"].setdefault("usage_count", 0)

        with (
            patch("baby.intelligence.save_thread") as mock_save_thread,
            patch("baby.intelligence.store_gene_keys") as mock_store_gene_keys,
            patch("baby.intelligence.store_format") as mock_store_format,
        ):
            # End thread
            engine.end_current_thread(plaintext_to_save=b"Thread content to save")

            # Verify save calls
            mock_save_thread.assert_called_once()
            mock_store_gene_keys.assert_called_once()
            mock_store_format.assert_called_once()

    def test_generate_and_save_response(self, initialized_intelligence_engine):
        """Test generating and saving a response"""
        engine = initialized_intelligence_engine
        import base64

        with (
            patch.object(engine, "_generate_response_bytes", return_value=b"Generated response"),
            patch.object(engine, "_append_to_thread") as mock_append,
        ):
            # Generate response
            response = engine.generate_and_save_response(length=20)

            # Verify results
            assert response == b"Generated response"

            # Verify method calls
            engine._generate_response_bytes.assert_called_once_with(20)
            expected_event = {"type": "output", "data": base64.b64encode(b"Generated response").decode("utf-8")}
            mock_append.assert_called_once_with(expected_event)

    def test_generate_response_bytes(self, initialized_intelligence_engine):
        """Test generating response bytes"""
        engine = initialized_intelligence_engine

        with (
            patch.object(engine, "_generate_response_byte", side_effect=[(97, 0), (98, 1), (99, 2)]),
            patch.object(engine.inference_engine, "process_byte", return_value=(0, 0.5)),
            patch.object(engine, "update_learning_state"),
        ):
            # Generate 3 bytes
            response = engine._generate_response_bytes(3)

            # Verify the output
            assert response == b"abc"

            # Check method calls
            assert engine._generate_response_byte.call_count == 3
            assert engine.inference_engine.process_byte.call_count == 3
            assert engine.update_learning_state.call_count == 3

    def test_generate_response_byte(self, initialized_intelligence_engine):
        """Test generating a single response byte"""
        engine = initialized_intelligence_engine

        # Mock pattern index and resonances
        with (
            patch.object(
                engine.inference_engine,
                "compute_contextual_resonances",
                return_value=[0.5] * 256,
            ),
            patch("random.choice", return_value=42),
        ):
            # Generate response byte
            byte, pattern_idx = engine._generate_response_byte()

            # Should return output byte from G[pattern_idx]
            assert byte == engine.inference_engine.G[pattern_idx]

    def test_update_learning_state(self, initialized_intelligence_engine):
        """Test updating learning state with new signature"""
        engine = initialized_intelligence_engine

        # Initialize pattern data
        pattern_index = 42
        engine.M["patterns"][pattern_index]["count"] = 5
        engine.M["patterns"][pattern_index]["first_cycle"] = 1
        engine.M["patterns"][pattern_index]["last_cycle"] = 5

        # Update learning state with the new signature
        engine.update_learning_state(source_byte=100, key_index=pattern_index, resonance=0.5, event_type="INPUT")

        # Verify pattern updates
        assert engine.M["patterns"][pattern_index]["count"] == 6
        assert engine.M["patterns"][pattern_index]["first_cycle"] == 1
        assert engine.M["patterns"][pattern_index]["last_cycle"] == engine.inference_engine.cycle_counter

        # Verify key recording
        assert len(engine.current_thread_keys) == 1
        gene_key = engine.current_thread_keys[0]
        assert gene_key["cycle"] == engine.inference_engine.cycle_counter
        assert gene_key["pattern_index"] == pattern_index
        assert gene_key["event_type"] == "INPUT"
        assert gene_key["source_byte"] == 100

    def test_update_learning_state_first_occurrence(self, initialized_intelligence_engine):
        """Test updating learning state for first pattern occurrence"""
        engine = initialized_intelligence_engine

        # Initialize pattern data
        pattern_index = 42
        engine.M["patterns"][pattern_index]["count"] = 0
        engine.M["patterns"][pattern_index]["first_cycle"] = None
        engine.M["patterns"][pattern_index]["last_cycle"] = None

        # Update learning state
        engine.update_learning_state(source_byte=200, key_index=pattern_index, resonance=0.3, event_type="OUTPUT")

        # Verify pattern updates
        assert engine.M["patterns"][pattern_index]["count"] == 1
        assert engine.M["patterns"][pattern_index]["first_cycle"] == engine.inference_engine.cycle_counter
        assert engine.M["patterns"][pattern_index]["last_cycle"] == engine.inference_engine.cycle_counter

    def test_encode_decode(self, initialized_intelligence_engine):
        """Test character encoding and decoding"""
        engine = initialized_intelligence_engine

        # Setup pattern with character label
        pattern_index = 42
        character_label = "test_character"
        engine.M["patterns"][pattern_index]["character"] = character_label

        # Test encode
        assert engine.encode(character_label) == pattern_index
        assert engine.encode("nonexistent") is None

        # Test decode
        assert engine.decode(pattern_index) == character_label
        assert engine.decode(99) is None

    def test_load_thread_content(self, initialized_intelligence_engine):
        """Test loading a thread's decrypted content as NDJSON events"""
        engine = initialized_intelligence_engine
        thread_uuid = "test-thread-uuid"
        import base64, json

        # Prepare a fake NDJSON thread with two events
        event1 = {"type": "input", "data": base64.b64encode(b"abc").decode("utf-8")}
        event2 = {"type": "output", "data": base64.b64encode(b"xyz").decode("utf-8")}
        ndjson = f"{json.dumps(event1)}\n{json.dumps(event2)}\n".encode("utf-8")
        with (
            patch("baby.intelligence.load_thread", return_value=ndjson) as mock_load_thread,
            patch(
                "baby.intelligence.load_thread_key", return_value=bytes([i % 256 for i in range(256)])
            ) as mock_load_thread_key,
        ):
            # Load thread content
            content = engine.load_thread_content(thread_uuid)
            # Should return a list of event dicts with decoded data
            assert isinstance(content, list)
            assert len(content) == 2
            assert content[0]["type"] == "input"
            assert content[0]["data"] == b"abc"
            assert content[1]["type"] == "output"
            assert content[1]["data"] == b"xyz"
            # Check that the right functions were called
            mock_load_thread.assert_called_once_with(engine.agent_uuid, thread_uuid)
            mock_load_thread_key.assert_called_once_with(engine.agent_uuid, thread_uuid, engine.agent_secret)

    def test_get_thread_relationships(self, initialized_intelligence_engine):
        """Test getting thread relationships"""
        engine = initialized_intelligence_engine
        thread_uuid = "test-thread-uuid"
        parent_uuid = "parent-thread-uuid"
        child_uuid = "child-thread-uuid"

        with (
            patch("baby.intelligence.parent", return_value=parent_uuid) as mock_parent,
            patch("baby.intelligence.children", return_value=[child_uuid]) as mock_children,
        ):
            # Get relationships
            relationships = engine.get_thread_relationships(thread_uuid)

            # Verify result
            assert relationships["parent"] == parent_uuid
            assert relationships["children"] == [child_uuid]

            # Check function calls
            mock_parent.assert_called_once_with(engine.agent_uuid, thread_uuid)
            mock_children.assert_called_once_with(engine.agent_uuid, thread_uuid)

    def test_get_thread_chain(self, initialized_intelligence_engine):
        """Test getting a thread chain"""
        engine = initialized_intelligence_engine
        thread_uuid = "test-thread-uuid"
        parent_uuid = "parent-thread-uuid"
        grandparent_uuid = "grandparent-thread-uuid"

        # Mock parent function
        with patch("baby.intelligence.parent") as mock_parent:
            # Set up mock return values
            mock_parent.side_effect = lambda a, t: {
                "test-thread-uuid": parent_uuid,
                "parent-thread-uuid": grandparent_uuid,
                "grandparent-thread-uuid": None,
            }.get(t)

            # Get thread chain
            chain = engine.get_thread_chain(thread_uuid)

            # Verify chain order (root->leaf)
            assert grandparent_uuid in chain
            assert parent_uuid in chain
            assert thread_uuid in chain

            # Check parent calls
            assert mock_parent.call_count >= 2

    def test_get_thread_statistics(self, initialized_intelligence_engine):
        """Test getting thread statistics"""
        engine = initialized_intelligence_engine

        # Create test data in the actual directory structure
        agent_uuid = engine.agent_uuid
        private_dir = Path("memories/private/agents")
        agent_shard = shard_path(private_dir, agent_uuid)
        agent_dir = agent_shard / f"agent-{agent_uuid}"
        threads_dir = agent_dir / "threads"
        threads_dir.mkdir(parents=True, exist_ok=True)

        # Use valid UUIDs for thread IDs
        thread_ids = [str(uuid.uuid4()) for _ in range(3)]
        registry = {"count": 3, "uuids": thread_ids}
        with open(threads_dir / "registry.json", "w") as f:
            json.dump(registry, f)

        # Create thread metadata files in the correct shard paths
        for thread_id, meta in zip(
            thread_ids,
            [
                cast(
                    ThreadMetadata,
                    {
                        "thread_uuid": None,
                        "parent_uuid": None,
                        "child_uuids": [str(thread_ids[1])],
                        "size_bytes": 1000,
                        "privacy": "public",
                    },
                ),
                cast(
                    ThreadMetadata,
                    {
                        "thread_uuid": None,
                        "parent_uuid": thread_ids[0],
                        "child_uuids": [str(thread_ids[2])],
                        "size_bytes": 2000,
                        "privacy": "public",
                    },
                ),
                cast(
                    ThreadMetadata,
                    {
                        "thread_uuid": None,
                        "parent_uuid": thread_ids[1],
                        "child_uuids": [],
                        "size_bytes": 3000,
                        "privacy": "public",
                    },
                ),
            ],
        ):
            meta["thread_uuid"] = thread_id
            # Use the helper to guarantee a flat list of strings
            meta["child_uuids"] = to_flat_str_list(meta.get("child_uuids"))
            thread_shard = shard_path(threads_dir, thread_id)
            thread_shard.mkdir(parents=True, exist_ok=True)
            with open(thread_shard / f"thread-{thread_id}.json", "w") as f:
                json.dump(meta, f)

        # Get statistics
        stats = engine.get_thread_statistics()

        # Verify statistics
        assert stats["total_threads"] == 3
        assert stats["total_size_bytes"] == 6000
        assert stats["relationship_stats"]["threads_with_parents"] == 2
        assert stats["relationship_stats"]["threads_with_children"] == 2
        assert stats["relationship_stats"]["isolated_threads"] == 0
        assert len(stats["thread_details"]) == 3

    def test_thread_capacity_exceeded(self, isolated_test_env):
        """
        Test thread capacity handling with proper session resumption.

        This test verifies the complex edge case where:
        1. A thread is created and content is added
        2. The engine is "restarted" (simulating session resumption)
        3. More content is added that exceeds capacity
        4. A new thread should be created and properly linked
        """
        # Create test masks
        patterns_array, _ = derive_canonical_patterns()
        patterns_array.tofile("memories/public/masks/epigenome.dat")
        genome_mask = np.arange(256, dtype=np.uint8)
        genome_mask.tofile("memories/public/masks/genome.dat")

        # Create baby preferences
        agent_uuid = "22222222-2222-2222-2222-222222222222"
        agent_secret = "test-capacity-secret"
        baby_prefs = {
            "agent_secret": agent_secret,
            "log_level": "info",
            "response_length": 100,
            "learning_rate": 1.0,
        }
        with open("baby/baby_preferences.json", "w") as f:
            json.dump(baby_prefs, f, indent=2)

        # Set small capacity in memory preferences
        mem_prefs_path = "memories/memory_preferences.json"
        with open(mem_prefs_path, "r") as f:
            mem_prefs = json.load(f)
        original_max_size = mem_prefs["storage_config"]["max_thread_size_mb"]
        mem_prefs["storage_config"]["max_thread_size_mb"] = 0.0001  # 100 bytes
        with open(mem_prefs_path, "w") as f:
            json.dump(mem_prefs, f, indent=2)

        try:
            # === PHASE 1: Initial session ===
            # Create agent and initialize first engine
            assign_agent_uuid(agent_uuid)
            engine1 = initialize_intelligence_engine(agent_uuid=agent_uuid, agent_secret=agent_secret)

            # Process first input (creates first thread)
            first_input = b"First message content"
            engine1.process_input_stream(first_input)
            first_thread_uuid = engine1.thread_uuid
            assert first_thread_uuid is not None, "First thread was not created"

            # === PHASE 2: Session resumption ===
            # Create new engine instance (simulating restart)
            engine2 = initialize_intelligence_engine(agent_uuid=agent_uuid, agent_secret=agent_secret)

            # CRITICAL: Restore session state to continue working on the previous thread
            engine2.thread_uuid = first_thread_uuid

            # Load the thread key for the previous thread
            thread_key = load_thread_key(agent_uuid, first_thread_uuid, agent_secret)
            assert thread_key is not None, "Failed to load thread key for session resumption"
            engine2.thread_file_key = thread_key

            # Load existing thread content
            existing_content = load_thread(agent_uuid, first_thread_uuid)
            if existing_content:
                # Decrypt to get the actual content size
                decrypted_content = bytearray(len(existing_content))
                for i in range(len(existing_content)):
                    decrypted_content[i] = existing_content[i] ^ thread_key[i % 256]
                engine2.active_thread_content = decrypted_content
                engine2.current_thread_size = len(decrypted_content)
            else:
                engine2.active_thread_content = bytearray()
                engine2.current_thread_size = 0

            # Load gene keys for this thread
            engine2.current_thread_keys = load_gene_keys(first_thread_uuid, agent_uuid, agent_secret)

            # === PHASE 3: Capacity test ===
            # Process second input that exceeds capacity
            large_input = b"X" * 200  # This should exceed 100-byte capacity
            engine2.process_input_stream(large_input)
            second_thread_uuid = engine2.thread_uuid

            # === VERIFICATION ===
            # Should have created a new thread
            assert second_thread_uuid is not None, "Second thread was not created"
            assert first_thread_uuid != second_thread_uuid, "New thread was not created when capacity exceeded"

            # Verify thread relationship
            relationships = engine2.get_thread_relationships(second_thread_uuid)
            assert relationships["parent"] == first_thread_uuid, "New thread is not properly linked to parent"

        finally:
            # Restore original capacity
            mem_prefs["storage_config"]["max_thread_size_mb"] = original_max_size
            with open(mem_prefs_path, "w") as f:
                json.dump(mem_prefs, f, indent=2)


# ------------------------------------------------------------------------------
# Test Utilities
# ------------------------------------------------------------------------------


def test_weighted_choice():
    """Test the weighted choice utility function"""
    items = ["a", "b", "c"]
    weights = [0.1, 0.8, 0.1]

    # Mock random to return specific values
    with patch("random.random") as mock_random:
        # Test selecting first item
        mock_random.return_value = 0.05
        result = weighted_choice(items, weights)
        assert result == "a"

        # Test selecting second item
        mock_random.return_value = 0.5
        result = weighted_choice(items, weights)
        assert result == "b"

        # Test selecting third item
        mock_random.return_value = 0.95
        result = weighted_choice(items, weights)
        assert result == "c"


def test_public_mode_initialization(public_intelligence_engine):
    """Test that public mode works correctly"""
    engine = public_intelligence_engine

    assert engine.agent_uuid is None
    assert engine.agent_secret is None
    assert engine.pattern_index is None  # Should be None in public mode
    assert isinstance(engine.format_uuid, str)
    assert isinstance(engine.M, dict)


def test_confidence_updates_on_learning(initialized_intelligence_engine):
    """
    Verify that update_learning_state correctly calculates and updates
    a pattern's confidence as a moving average.
    """
    engine = initialized_intelligence_engine
    pattern_index = 10
    # Precondition: Confidence should be 0.0 initially.
    initial_confidence = engine.M["patterns"][pattern_index]["confidence"]
    assert initial_confidence == 0.0, "Initial confidence should be 0.0"
    # Action 1: A perfect match (resonance = 0.0)
    engine.update_learning_state(source_byte=0, key_index=pattern_index, resonance=0.0, event_type="INPUT")
    first_confidence = engine.M["patterns"][pattern_index]["confidence"]
    assert np.isclose(first_confidence, 0.01), f"Confidence after perfect match is wrong: {first_confidence}"
    # Action 2: A perfect mismatch (resonance = np.pi)
    engine.update_learning_state(source_byte=0, key_index=pattern_index, resonance=np.pi, event_type="INPUT")
    second_confidence = engine.M["patterns"][pattern_index]["confidence"]
    assert np.isclose(second_confidence, 0.0099), f"Confidence after mismatch is wrong: {second_confidence}"


def test_intelligent_encode_prefers_stronger_pattern(initialized_intelligence_engine):
    """
    Verify that intelligent_encode uses count and confidence to select the
    best pattern for a character when multiple options exist.
    """
    engine = initialized_intelligence_engine
    # Setup: Map "A" to two different patterns with different strengths.
    engine.M["patterns"][65]["character"] = "A"
    engine.M["patterns"][65]["count"] = 10
    engine.M["patterns"][65]["confidence"] = 0.5  # Score = 10 * 0.5 = 5
    engine.M["patterns"][150]["character"] = "A"
    engine.M["patterns"][150]["count"] = 100
    engine.M["patterns"][150]["confidence"] = 0.9  # Score = 100 * 0.9 = 90
    chosen_index = engine.intelligent_encode("A")
    assert chosen_index == 150, "intelligent_encode failed to pick the stronger pattern."


def test_intelligent_generation_prefers_meaningful_pattern(initialized_intelligence_engine, monkeypatch):
    """
    Verify that _generate_response_byte prefers a semantically meaningful pattern
    over a more physically resonant but meaningless one.
    """
    engine = initialized_intelligence_engine
    # Setup: Create a scenario with two candidates.
    engine.M["patterns"][10]["character"] = None
    engine.M["patterns"][10]["confidence"] = 0.0
    engine.M["patterns"][20]["character"] = "B"
    engine.M["patterns"][20]["confidence"] = 0.95  # High semantic score
    mock_resonances = np.full(256, np.pi)  # All are invalid by default
    mock_resonances[10] = 0.1 * np.pi  # physical_score = 0.9
    mock_resonances[20] = 0.4 * np.pi  # physical_score = 0.6
    monkeypatch.setattr(engine.inference_engine, "compute_pattern_resonances", lambda: mock_resonances)
    _output_byte, chosen_index = engine._generate_response_byte()
    assert chosen_index == 20, "Generation failed to prioritize the semantically meaningful pattern."


# ------------------------------------------------------------------------------
# Main test runner
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """Run tests when file is executed directly"""
    pytest.main(["-v", __file__])
