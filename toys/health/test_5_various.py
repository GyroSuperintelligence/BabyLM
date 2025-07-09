"""
Miscellaneous and integration tests for the BabyLM system.
Includes tests for public/private mode, pattern index, enhanced thread management, integration, and utility functions.
"""

import os
import uuid
import json
import numpy as np
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch
from typing import cast
from datetime import datetime

# Import modules from baby package
from baby.inference import InferenceEngine
from baby.information import (
    InformationEngine,
    ensure_agent_uuid,
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
    assign_agent_uuid,
)
from baby.intelligence import IntelligenceEngine, weighted_choice, initialize_intelligence_engine
from baby.governance import derive_canonical_patterns
from baby.types import GeneKeysMetadata, FormatMetadata

# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files"""
    return tmp_path


@pytest.fixture
def mock_memories_dir(temp_dir, monkeypatch):
    """Mock the memories directory structure"""
    memories_dir = temp_dir / "memories"
    memories_dir.mkdir()

    # Create subdirectories
    (memories_dir / "private").mkdir()
    (memories_dir / "private/agents").mkdir()
    (memories_dir / "public").mkdir()
    (memories_dir / "public/formats").mkdir()
    (memories_dir / "public/masks").mkdir()
    (memories_dir / "public/threads").mkdir()
    (memories_dir / "public/keys").mkdir()

    # Monkeypatch os.makedirs to use the temp directory
    original_makedirs = os.makedirs

    def mock_makedirs(path, exist_ok=False):
        if isinstance(path, str) and path.startswith("memories/"):
            path = str(temp_dir / path)
        return original_makedirs(path, exist_ok=exist_ok)

    monkeypatch.setattr(os, "makedirs", mock_makedirs)

    # Monkeypatch open function
    original_open = open

    def mock_open_wrapper(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith("memories/"):
            path = str(temp_dir / path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_wrapper)

    return memories_dir


@pytest.fixture
def mock_uuid():
    """Mock UUID generation for predictable test results"""
    return "00000000-0000-0000-0000-000000000000"


@pytest.fixture
def test_tensor():
    """Create a test tensor with standard shape"""
    return np.zeros((4, 2, 3, 2), dtype=np.float32)


@pytest.fixture
def inference_engine():
    """Create an inference engine for testing"""
    with (
        patch.object(InferenceEngine, "_load_patterns", return_value=(np.zeros((256, 48)), ["test"] * 256)),
        patch.object(InferenceEngine, "_load_genome_mask", return_value=np.arange(256, dtype=np.uint8)),
        patch.object(InferenceEngine, "_initialize_epigenome"),
    ):
        engine = InferenceEngine()
        engine.T = np.zeros((4, 2, 3, 2), dtype=np.float32)
        return engine


@pytest.fixture
def information_engine():
    """Create an information engine for testing"""
    return InformationEngine()


@pytest.fixture
def mock_env(tmp_path):
    """
    Creates a temporary, self-contained environment for the tests to run in.
    This fixture simulates the required 'memories' and 'baby' directories.
    It automatically cleans up after the tests are done.
    """
    # Create the base directories inside the temporary folder provided by pytest
    memories_dir = tmp_path / "memories"
    baby_dir = tmp_path / "baby"

    # Create necessary subdirectories
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

    # Change the current working directory to the temporary path
    # This makes file paths like "memories/..." work as expected
    original_cwd = Path.cwd()
    os.chdir(tmp_path)

    yield tmp_path  # The test runs now

    # Teardown: Change back to the original directory
    os.chdir(original_cwd)


@pytest.fixture
def initialized_intelligence_engine(mock_env):
    """
    This fixture populates the mock environment with all the
    necessary files (masks, preferences, formats) and returns a fully
    initialized IntelligenceEngine instance ready for testing in private-agent mode.
    """
    # 1. Define a specific agent UUID and secret for predictable testing
    agent_uuid = "11111111-1111-1111-1111-111111111111"
    agent_secret = "test-secret-for-fixture"

    # 2. Create default baby_preferences.json with the secret
    baby_prefs_path = Path("baby/baby_preferences.json")
    baby_prefs = {
        "agent_secret": agent_secret,
        "log_level": "info",
        "response_length": 100,
        "learning_rate": 1.0,
    }
    with open(baby_prefs_path, "w") as f:
        json.dump(baby_prefs, f, indent=2)

    # 3. Create the masks required for InferenceEngine
    patterns_array, _ = derive_canonical_patterns()
    patterns_array.tofile("memories/public/masks/epigenome.dat")

    genome_mask = np.arange(256, dtype=np.uint8)  # Identity mapping for predictability
    genome_mask.tofile("memories/public/masks/genome.dat")

    # 4. Ensure agent directory is created
    assign_agent_uuid(agent_uuid)

    # 5. Initialize the engine EXPLICITLY in private mode
    engine = initialize_intelligence_engine(agent_uuid=agent_uuid, agent_secret=agent_secret)

    # 6. Assert that the engine is correctly configured for private mode
    assert engine.agent_uuid == agent_uuid
    assert engine.agent_secret == agent_secret

    # Assert that the private agent directories exist
    private_dir = Path("memories/private/agents")
    agent_shard = shard_path(private_dir, engine.agent_uuid)
    agent_dir = agent_shard / f"agent-{engine.agent_uuid}"
    keys_dir = agent_dir / "keys"

    assert agent_dir.exists(), f"Agent directory was not created: {agent_dir}"
    assert keys_dir.exists(), f"Keys directory was not created: {keys_dir}"

    return engine


@pytest.fixture
def public_intelligence_engine(mock_env):
    """Initializes an IntelligenceEngine in public/curation mode."""
    # Create the masks required for InferenceEngine
    patterns_array, _ = derive_canonical_patterns()
    patterns_array.tofile("memories/public/masks/epigenome.dat")

    genome_mask = np.arange(256, dtype=np.uint8)
    genome_mask.tofile("memories/public/masks/genome.dat")

    # This call correctly defaults to public mode
    engine = initialize_intelligence_engine(agent_uuid=None, agent_secret=None)

    # Verify public mode
    assert engine.agent_uuid is None
    assert engine.agent_secret is None

    return engine


@pytest.fixture(scope="session", autouse=True)
def cleanup_htmlcov():
    yield
    if os.path.exists("htmlcov"):
        shutil.rmtree("htmlcov")


# ------------------------------------------------------------------------------
# Public Mode Tests
# ------------------------------------------------------------------------------


class TestPublicMode:
    """Tests for public/curation mode operation"""

    def test_public_engine_init(self, public_intelligence_engine):
        """Verify the engine is in public mode."""
        engine = public_intelligence_engine
        assert engine.agent_uuid is None
        assert engine.agent_secret is None
        assert engine.pattern_index is None  # PatternIndex requires an agent

    def test_public_thread_and_gene_keys_storage(self, public_intelligence_engine):
        """Verify that threads and gene_keys are saved unencrypted to public directories."""
        engine = public_intelligence_engine
        engine.process_input_stream(b"public data")

        assert engine.thread_uuid is not None

        # Check for public thread file
        thread_shard = shard_path(Path("memories/public/threads"), engine.thread_uuid)
        assert (thread_shard / f"thread-{engine.thread_uuid}.dat").exists()  # Note: .dat, not .enc
        assert not (thread_shard / f"thread-{engine.thread_uuid}.enc").exists()

        # Check for public gene keys file
        keys_shard = shard_path(Path("memories/public/keys"), engine.thread_uuid)
        assert (keys_shard / f"gene-{engine.thread_uuid}.ndjson").exists()
        assert not (keys_shard / f"gene-{engine.thread_uuid}.ndjson.enc").exists()

    def test_private_operations_fail_in_public_mode(self, public_intelligence_engine):
        """Verify that operations requiring a key fail gracefully."""
        # Loading encrypted content should fail or return None
        assert public_intelligence_engine.load_thread_content("some-uuid") is None
        # Trying to derive a key should return None
        assert public_intelligence_engine._derive_file_key(np.zeros(1), None, "some-uuid") is None

    def test_public_format_sharing(self, public_intelligence_engine):
        """Test that formats are shared in public directories"""
        engine = public_intelligence_engine
        # Process some data to create pattern usage
        engine.process_input_stream(b"test data for format")

        # Format should be stored in public directory
        format_uuid = engine.format_uuid
        formats_dir = Path("memories/public/formats")
        format_shard = shard_path(formats_dir, format_uuid)
        format_path = format_shard / f"format-{format_uuid}.json"
        assert format_path.exists()


# ------------------------------------------------------------------------------
# Pattern Index and Contextual Generation Tests
# ------------------------------------------------------------------------------


class TestPatternIndexAndContext:
    """Tests for the PatternIndex and contextual generation features"""

    def test_pattern_index_updates(self, initialized_intelligence_engine):
        """Test that the PatternIndex is updated correctly after processing."""
        engine = initialized_intelligence_engine

        # Manually create gene_keys with known patterns:
        gene_keys: list[GeneKeysMetadata] = [
            {
                "cycle": 1,
                "pattern_index": 10,
                "thread_uuid": "test-uuid",
                "agent_uuid": engine.agent_uuid,
                "format_uuid": engine.format_uuid,
                "event_type": "INPUT",
                "source_byte": 65,  # 'A'
                "resonance": 0.1,
                "created_at": datetime.now().isoformat(),
            },
            {
                "cycle": 2,
                "pattern_index": 11,
                "thread_uuid": "test-uuid",
                "agent_uuid": engine.agent_uuid,
                "format_uuid": engine.format_uuid,
                "event_type": "INPUT",
                "source_byte": 66,  # 'B'
                "resonance": 0.2,
                "created_at": datetime.now().isoformat(),
            },
            {
                "cycle": 3,
                "pattern_index": 10,
                "thread_uuid": "test-uuid",
                "agent_uuid": engine.agent_uuid,
                "format_uuid": engine.format_uuid,
                "event_type": "INPUT",
                "source_byte": 65,  # 'A'
                "resonance": 0.1,
                "created_at": datetime.now().isoformat(),
            },
        ]

        engine.pattern_index.update_from_thread("test-uuid", gene_keys)

        # Assertions
        next_patterns = engine.pattern_index.get_likely_next_patterns(current_pattern=10, top_k=1)
        assert len(next_patterns) == 1
        assert next_patterns[0][0] == 11  # The most likely pattern after 10 (A) is 11 (B)

    def test_degenerate_patterns_and_deterministic_choice(self, initialized_intelligence_engine):
        """
        Verify that a tensor state matching multiple canonical patterns (degeneracy)
        still leads to a deterministic choice based on the lowest index,
        confirming a fundamental physical property of the system.
        """
        engine = initialized_intelligence_engine

        # 1. Set the engine's tensor T to be a copy of a known pattern, F[100].
        # Our analysis shows F[100] is one of several patterns identical to -gene_add.
        engine.inference_engine.T = engine.inference_engine.F[100].reshape(4, 2, 3, 2)

        # 2. Calculate the "natural" resonances against all 256 canonical patterns.
        natural_resonances = engine.inference_engine.compute_pattern_resonances()
        min_resonance_val = np.min(natural_resonances)

        # 3. Acknowledge and identify the degeneracy. Find ALL indices that have
        # the absolute minimum resonance. This is the set of degenerate patterns.
        indices_with_min_resonance = [i for i, v in enumerate(natural_resonances) if np.isclose(v, min_resonance_val)]
        print(f"Minimum resonance value found: {min_resonance_val}")
        print(
            f"Found {len(indices_with_min_resonance)} degenerate patterns with this resonance: {indices_with_min_resonance}"
        )

        # 4. Verify our core understanding of the system's physics.
        # The minimum resonance should be effectively zero, indicating a perfect match.
        assert np.isclose(min_resonance_val, 0.0)

        # Both 2 and 100 should be in the list of perfect matches, proving they are degenerate.
        assert 2 in indices_with_min_resonance
        assert 100 in indices_with_min_resonance

        # 5. Generate a byte using the pure, spec-compliant generation logic.
        _byte, pattern_idx = engine._generate_response_byte()

        # 6. Assert that the chosen pattern is the FIRST one in the list of
        # equally good candidates. This validates the deterministic behavior of the
        # selection logic in the face of physical ambiguity.
        assert pattern_idx == min(indices_with_min_resonance)

        # In this specific case, based on the `argmin` behavior, we know the first
        # minimum index in the full list of resonances is 2.
        assert pattern_idx == 2

    def test_pattern_sequence_tracking(self, initialized_intelligence_engine):
        """Test that pattern sequences are tracked correctly"""
        engine = initialized_intelligence_engine
        pattern_index = engine.pattern_index

        # Create a simple sequence: 1 -> 2 -> 3 -> 1
        test_gene_keys = [
            {
                "cycle": 1,
                "pattern_index": 1,
                "thread_uuid": "test",
                "agent_uuid": engine.agent_uuid,
                "format_uuid": engine.format_uuid,
                "event_type": "INPUT",
                "source_byte": 1,
                "resonance": 0.1,
                "created_at": datetime.now().isoformat(),
            },
            {
                "cycle": 2,
                "pattern_index": 2,
                "thread_uuid": "test",
                "agent_uuid": engine.agent_uuid,
                "format_uuid": engine.format_uuid,
                "event_type": "INPUT",
                "source_byte": 2,
                "resonance": 0.1,
                "created_at": datetime.now().isoformat(),
            },
            {
                "cycle": 3,
                "pattern_index": 3,
                "thread_uuid": "test",
                "agent_uuid": engine.agent_uuid,
                "format_uuid": engine.format_uuid,
                "event_type": "INPUT",
                "source_byte": 3,
                "resonance": 0.1,
                "created_at": datetime.now().isoformat(),
            },
            {
                "cycle": 4,
                "pattern_index": 1,
                "thread_uuid": "test",
                "agent_uuid": engine.agent_uuid,
                "format_uuid": engine.format_uuid,
                "event_type": "INPUT",
                "source_byte": 1,
                "resonance": 0.1,
                "created_at": datetime.now().isoformat(),
            },
        ]

        pattern_index.update_from_thread("test-thread", test_gene_keys)

        # Test sequence predictions
        likely_after_1 = pattern_index.get_likely_next_patterns(1, top_k=1)
        likely_after_2 = pattern_index.get_likely_next_patterns(2, top_k=1)

        # After pattern 1, we should expect pattern 2
        assert len(likely_after_1) > 0
        assert likely_after_1[0][0] == 2

        # After pattern 2, we should expect pattern 3
        assert len(likely_after_2) > 0
        assert likely_after_2[0][0] == 3


# ------------------------------------------------------------------------------
# Enhanced Thread Management Tests
# ------------------------------------------------------------------------------


class TestEnhancedThreadManagement:
    """Tests for enhanced thread management features"""

    def test_thread_chaining_on_rollover(self, initialized_intelligence_engine):
        """Verify parent-child links are correctly created across multiple rollovers."""
        engine = initialized_intelligence_engine
        engine.memory_prefs["storage_config"]["max_thread_size_mb"] = 0.00002  # ~20 bytes

        # Process three streams, each forcing a new thread
        engine.process_input_stream(b"1" * 15)
        thread1_uuid = engine.thread_uuid

        engine.process_input_stream(b"2" * 15)
        thread2_uuid = engine.thread_uuid

        engine.process_input_stream(b"3" * 15)
        thread3_uuid = engine.thread_uuid

        # Verify the chain is not a single thread
        assert thread1_uuid != thread2_uuid
        assert thread2_uuid != thread3_uuid

        # Verify the relationships using the information helpers
        assert parent(engine.agent_uuid, thread3_uuid) == thread2_uuid
        assert parent(engine.agent_uuid, thread2_uuid) == thread1_uuid
        assert parent(engine.agent_uuid, thread1_uuid) is None

        assert children(engine.agent_uuid, thread1_uuid) == [thread2_uuid]
        assert children(engine.agent_uuid, thread2_uuid) == [thread3_uuid]

    def test_thread_content_accumulation(self, initialized_intelligence_engine):
        """Test that thread content properly accumulates before rollover"""
        engine = initialized_intelligence_engine

        # Process multiple small inputs
        engine.process_input_stream(b"First part")
        engine.process_input_stream(b"Second part")
        engine.process_input_stream(b"Third part")

        # Should still be in the same thread
        thread_uuid = engine.thread_uuid

        # Load the thread content and verify accumulation
        content = engine.load_thread_content(thread_uuid)
        assert b"First part" in content
        assert b"Second part" in content
        assert b"Third part" in content

    def test_thread_metadata_consistency(self, initialized_intelligence_engine):
        """Test that thread metadata remains consistent across operations"""
        engine = initialized_intelligence_engine

        # Create a thread with metadata
        engine.process_input_stream(b"test content")
        thread_uuid = engine.thread_uuid

        # Verify metadata exists and is consistent
        stats = engine.get_thread_statistics()
        assert stats["total_threads"] >= 1

        # Find our thread in the details
        our_thread = None
        for thread_detail in stats["thread_details"]:
            if thread_detail["thread_uuid"] == thread_uuid:
                our_thread = thread_detail
                break

        assert our_thread is not None
        assert our_thread["size_bytes"] > 0


# ------------------------------------------------------------------------------
# Integration Tests
# ------------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for end-to-end functionality"""

    def test_end_to_end_processing(self, initialized_intelligence_engine):
        """Test end-to-end processing of input data."""
        # The 'initialized_intelligence_engine' fixture has already done all the setup.
        intelligence_engine = initialized_intelligence_engine

        # Process input data
        test_input = b"Test input for end-to-end processing"
        plaintext, ciphertext = intelligence_engine.process_input_stream(test_input)

        # Verify results
        assert plaintext == test_input
        assert len(ciphertext) == len(test_input)
        assert ciphertext != test_input  # Ensure it was actually encrypted

        # Generate response
        response = intelligence_engine.generate_and_save_response(length=20)

        # Verify response
        assert len(response) == 20

        # Verify that thread files were created
        assert intelligence_engine.thread_uuid is not None

        # Check that thread file exists
        agent_uuid = intelligence_engine.agent_uuid
        thread_uuid = intelligence_engine.thread_uuid
        thread_shard = shard_path(
            Path(f"memories/private/agents/{agent_uuid[:2]}/agent-{agent_uuid}/threads"), thread_uuid
        )
        thread_path = thread_shard / f"thread-{thread_uuid}.enc"
        assert thread_path.exists()

        # Check that key file exists
        key_shard = shard_path(Path(f"memories/private/agents/{agent_uuid[:2]}/agent-{agent_uuid}/keys"), thread_uuid)
        key_path = key_shard / f"key-{thread_uuid}.bin.enc"
        assert key_path.exists()

    def test_load_thread_round_trip(self, initialized_intelligence_engine):
        """Test saving and then loading a thread to verify the full cycle."""
        intelligence_engine = initialized_intelligence_engine

        # 1. Process some input to create a thread
        original_content = b"This content will be saved and then reloaded."
        intelligence_engine.process_input_stream(original_content)
        thread_uuid = intelligence_engine.thread_uuid

        # 2. Load the thread content back
        loaded_content = intelligence_engine.load_thread_content(thread_uuid)

        # 3. Verify the thread content matches
        assert loaded_content == original_content

    def test_select_stable_format(self, initialized_intelligence_engine):
        """Test selecting a format based on domain and stability."""
        engine = initialized_intelligence_engine

        # Create test formats
        formats = [
            {
                "format_uuid": "english-stable-1",
                "format_name": "Eng Stable",
                "stability": "stable",
                "metadata": {"tags": ["english"], "description": "", "usage_count": 10},
                "patterns": [{"index": i} for i in range(256)],
            },
            {
                "format_uuid": "english-stable-2",
                "format_name": "Eng Stable More Used",
                "stability": "stable",
                "metadata": {"tags": ["english"], "description": "", "usage_count": 20},
                "patterns": [{"index": i} for i in range(256)],
            },
            {
                "format_uuid": "code-stable-1",
                "format_name": "Code Stable",
                "stability": "stable",
                "metadata": {"tags": ["code"], "description": "", "usage_count": 5},
                "patterns": [{"index": i} for i in range(256)],
            },
        ]

        # Store formats
        for fmt in formats:
            store_format(cast(FormatMetadata, fmt))

        # Select format
        with patch(
            "baby.intelligence.list_formats", return_value=["english-stable-1", "english-stable-2", "code-stable-1"]
        ):
            # Should select English format with highest usage
            selected = engine.select_stable_format("english", "stable")
            assert selected == "english-stable-2"

            # Should select code format
            selected = engine.select_stable_format("code", "stable")
            assert selected == "code-stable-1"

            # Should return None for nonexistent domain
            selected = engine.select_stable_format("nonexistent", "stable")
            assert selected is None

    def test_public_private_interoperability(self, public_intelligence_engine, initialized_intelligence_engine):
        """Test that public and private systems can coexist and share formats"""
        public_engine = public_intelligence_engine
        private_engine = initialized_intelligence_engine

        # Both engines should be able to access formats
        public_formats = list_formats()
        private_formats = list_formats()

        # Formats should be shared (both can see the same public formats)
        assert len(public_formats) > 0
        assert len(private_formats) > 0

        # Should have at least some overlap
        common_formats = set(public_formats) & set(private_formats)
        assert len(common_formats) > 0

    def test_gene_keys_learning_cycle(self, initialized_intelligence_engine):
        """Test complete learning cycle with gene keys storage and retrieval"""
        engine = initialized_intelligence_engine

        # Process input to generate gene keys
        test_input = b"Learning cycle test"
        engine.process_input_stream(test_input)

        # Generate response to create more gene keys
        engine.generate_and_save_response(length=10)

        thread_uuid = engine.thread_uuid

        # Verify gene keys were stored
        stored_gene_keys = load_gene_keys(
            thread_uuid=thread_uuid, agent_uuid=engine.agent_uuid, agent_secret=engine.agent_secret
        )

        # Should have gene keys for both input and output
        assert len(stored_gene_keys) > 0

        # Check that we have both INPUT and OUTPUT events
        event_types = {gk["event_type"] for gk in stored_gene_keys}
        assert "INPUT" in event_types
        assert "OUTPUT" in event_types


# ------------------------------------------------------------------------------
# Utility Function Tests
# ------------------------------------------------------------------------------


class TestUtilityFunctions:
    """Tests for utility functions"""

    def test_weighted_choice(self):
        """Test weighted choice selection"""
        items = ["A", "B", "C", "D"]

        # Test with equal weights
        with patch("random.random", return_value=0.4):
            result = weighted_choice(items, [1, 1, 1, 1])
            assert result == "B"  # 0.4 falls in the second item's range

        # Test with unequal weights
        with patch("random.random", return_value=0.7):
            result = weighted_choice(items, [1, 2, 4, 8])
            # 0.7 * (1+2+4+8) = 10.5, which falls in range for "D"
            assert result == "D"

    def test_initialize_intelligence_engine_modes(self, mock_env):
        """Test intelligence engine initialization in different modes"""
        # Create masks for all tests
        patterns_array, _ = derive_canonical_patterns()
        patterns_array.tofile("memories/public/masks/epigenome.dat")
        genome_mask = np.arange(256, dtype=np.uint8)
        genome_mask.tofile("memories/public/masks/genome.dat")

        # Test public mode
        public_engine = initialize_intelligence_engine(agent_uuid=None, agent_secret=None)
        assert public_engine.agent_uuid is None
        assert public_engine.agent_secret is None

        # Test private mode with explicit parameters
        test_agent_uuid = "test-agent-uuid"
        test_agent_secret = "test-agent-secret"
        private_engine = initialize_intelligence_engine(agent_uuid=test_agent_uuid, agent_secret=test_agent_secret)
        assert private_engine.agent_uuid == test_agent_uuid
        assert private_engine.agent_secret == test_agent_secret

        # Test auto-initialization mode
        with patch("uuid.uuid4", return_value=uuid.UUID("00000000-0000-0000-0000-000000000000")):
            # Create baby_preferences.json with a secret
            prefs_dir = Path("baby")
            prefs_dir.mkdir(exist_ok=True)
            with open(prefs_dir / "baby_preferences.json", "w") as f:
                json.dump({"agent_secret": "auto_secret"}, f)

            # Initialize engine without parameters (should use auto-agent)
            auto_engine = initialize_intelligence_engine()
            assert isinstance(auto_engine, IntelligenceEngine)
            assert auto_engine.agent_uuid is not None
            assert auto_engine.agent_secret == "auto_secret"


# ------------------------------------------------------------------------------
# Main test runner
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """Run tests when file is executed directly"""
    pytest.main(["-v", __file__])
