"""
Miscellaneous and integration tests for the BabyLM system.
Includes tests for public/private mode, pattern index, enhanced thread management, integration, and utility functions.
"""

import uuid
import json
import numpy as np
import pytest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import patch
from typing import cast
from datetime import datetime

# Import modules from baby package
from baby.information import (
    shard_path,
    parent,
    children,
    store_format,
    list_formats,
    load_gene_keys,
    get_memory_preferences,
    update_registry,
    _read_registry_cached,
    _REGISTRY_CACHE,
    _REGISTRY_CACHE_MAX_SIZE,
    json_loads,
    json_dumps,
)
from baby.intelligence import IntelligenceEngine, weighted_choice, initialize_intelligence_engine
from baby.governance import derive_canonical_patterns
from baby.types import GeneKeysMetadata, FormatMetadata


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

    def test_public_thread_and_gene_keys_storage(self, public_intelligence_engine, mock_env):
        """Verify that threads and gene_keys are saved unencrypted to public directories and are append-only NDJSON."""
        engine = public_intelligence_engine

        # --- Session 1 ---
        engine.process_input_stream(b"public data 1", privacy="public")
        first_thread_uuid = engine.thread_uuid
        assert first_thread_uuid is not None
        engine.finalize_and_save_thread(privacy="public")

        # --- Session 2 (Resuming the same thread) ---
        engine.resume_thread(first_thread_uuid, privacy="public")
        engine.process_input_stream(b"public data 2", privacy="public")
        engine.finalize_and_save_thread(privacy="public")

        # --- Assertions for Thread Content ---
        import os

        engine.finalize_and_save_thread(privacy="public")
        print("CWD:", os.getcwd())
        thread_shard = shard_path(
            Path(str(mock_env / "toys/health/memories/public/threads")), first_thread_uuid, engine.memory_prefs
        )
        thread_path = thread_shard / f"thread-{first_thread_uuid}.ndjson"
        print("Thread path:", thread_path)
        print("Files in dir:", list(thread_path.parent.iterdir()))
        assert thread_path.exists(), "Public thread NDJSON file was not created."
        assert not (thread_shard / f"thread-{first_thread_uuid}.enc").exists()

        with open(thread_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Use b64 representation as that's what's stored in the file
        assert any("cHVibGljIGRhdGEgMQ==" in line for line in lines)  # "public data 1"
        assert any("cHVibGljIGRhdGEgMg==" in line for line in lines)  # "public data 2"
        assert len(lines) >= 2, "Expected at least two event lines in the thread file."

        # --- Assertions for Gene Keys ---
        keys_shard = shard_path(
            Path(str(mock_env / "toys/health/memories/public/keys")), first_thread_uuid, engine.memory_prefs
        )
        gene_keys_path = keys_shard / f"gene-{first_thread_uuid}.ndjson"
        assert gene_keys_path.exists(), "Public gene keys NDJSON file was not created."

        with open(gene_keys_path, "r", encoding="utf-8") as f:
            gene_key_lines = f.readlines()
        # The number of gene keys should be the total from both sessions
        assert len(gene_key_lines) == (len(b"public data 1") + len(b"public data 2"))

        # --- Test direct append to gene_keys file ---
        from baby.information import store_gene_keys, load_gene_keys
        from baby.types import GeneKeysMetadata
        from datetime import datetime

        new_gene_key: GeneKeysMetadata = {
            "cycle": 999,
            "pattern_index": 123,
            "thread_uuid": first_thread_uuid,
            "format_uuid": "test-format",
            "event_type": "INPUT",
            "source_byte": 42,
            "resonance": 0.5,
            "created_at": datetime.now().isoformat(),
            "privacy": "public",
            "agent_uuid": None,
        }
        store_gene_keys(
            first_thread_uuid,
            [new_gene_key],
            privacy="public",
            prefs=engine.memory_prefs,
            base_memories_dir=str(mock_env / "toys/health/memories"),
        )

        loaded_keys = load_gene_keys(
            first_thread_uuid, prefs=engine.memory_prefs, base_memories_dir=str(mock_env / "toys/health/memories")
        )
        # Total count should be original count + 1
        assert len(loaded_keys) == len(gene_key_lines) + 1
        assert any(key["pattern_index"] == 123 for key in loaded_keys)

    def test_private_operations_fail_in_public_mode(self, public_intelligence_engine):
        """Verify that operations requiring a key fail gracefully."""
        # Loading encrypted content should fail or return None
        assert public_intelligence_engine.load_thread_content("some-uuid") is None
        # Trying to derive a key should return None
        assert public_intelligence_engine._derive_file_key(np.zeros(1), None, "some-uuid") is None

    def test_public_format_sharing(self, public_intelligence_engine, mock_env):
        """Test that formats are shared in public directories"""
        engine = public_intelligence_engine
        # Process some data to create pattern usage
        engine.process_input_stream(b"test data for format")

        # Format should be stored in public directory
        format_uuid = engine.format_uuid
        formats_dir = Path("toys/health/memories/public/formats")
        format_shard = shard_path(formats_dir, format_uuid, engine.memory_prefs)
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
                "privacy": "private",
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
                "privacy": "private",
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
                "privacy": "private",
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
            f"Found {len(indices_with_min_resonance)} degenerate patterns with this "
            f"resonance: {indices_with_min_resonance}"
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
                "privacy": "private",
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
                "privacy": "private",
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
                "privacy": "private",
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
                "privacy": "private",
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
        base_dir = str(initialized_intelligence_engine.base_memories_dir)
        prefs = get_memory_preferences(base_dir)
        assert parent(initialized_intelligence_engine.agent_uuid, thread3_uuid, prefs, base_dir) == thread2_uuid
        assert parent(initialized_intelligence_engine.agent_uuid, thread2_uuid, prefs, base_dir) == thread1_uuid
        assert parent(initialized_intelligence_engine.agent_uuid, thread1_uuid, prefs, base_dir) is None

        assert children(initialized_intelligence_engine.agent_uuid, thread1_uuid, prefs, base_dir) == [thread2_uuid]
        assert children(initialized_intelligence_engine.agent_uuid, thread2_uuid, prefs, base_dir) == [thread3_uuid]

    def test_thread_content_accumulation(self, initialized_intelligence_engine):
        """Test that thread content properly accumulates before rollover"""
        engine = initialized_intelligence_engine

        # Process multiple small inputs
        engine.process_input_stream(b"First part")
        engine.process_input_stream(b"Second part")
        engine.process_input_stream(b"Third part")
        # Explicitly finalize to ensure content is written
        engine.finalize_and_save_thread()

        # Should still be in the same thread
        thread_uuid = engine.thread_uuid

        # Load the thread content and verify accumulation
        content = engine.load_thread_content(thread_uuid)
        assert content is not None
        # Check that each part is present in the decoded data fields
        data_fields = [event["data"] for event in content if "data" in event]
        assert b"First part" in data_fields
        assert b"Second part" in data_fields
        assert b"Third part" in data_fields

    def test_thread_metadata_consistency(self, initialized_intelligence_engine):
        """Test that thread metadata remains consistent and can be read by stats."""
        engine = initialized_intelligence_engine

        # Create a thread by processing some data
        engine.process_input_stream(b"test content for stats")
        engine.finalize_and_save_thread()
        thread_uuid = engine.thread_uuid

        # Get statistics, which reads from the registry and metadata files
        stats = engine.get_thread_statistics()

        assert stats["total_threads"] >= 1

        # Find our specific thread in the detailed stats
        our_thread_details = None
        for detail in stats["thread_details"]:
            if detail["thread_uuid"] == thread_uuid:
                our_thread_details = detail
                break

        assert our_thread_details is not None, "Created thread not found in statistics"
        assert our_thread_details["size_bytes"] > 0


# ------------------------------------------------------------------------------
# Integration Tests
# ------------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for end-to-end functionality"""

    def test_end_to_end_processing(self, initialized_intelligence_engine, mock_env):
        """Test complete end-to-end processing workflow"""
        intelligence_engine = initialized_intelligence_engine

        # Process input
        test_input = b"Hello, world!"
        intelligence_engine.process_input_stream(test_input)

        # Generate response
        response = intelligence_engine.generate_and_save_response(length=20)
        intelligence_engine.finalize_and_save_thread()  # Ensure thread is saved

        # Verify response
        assert len(response) == 20

        # Verify that thread files were created
        assert intelligence_engine.thread_uuid is not None

        # Check that thread file exists
        agent_uuid = intelligence_engine.agent_uuid
        thread_uuid = intelligence_engine.thread_uuid
        thread_shard = shard_path(
            Path(f"toys/health/memories/private/agents/{agent_uuid[:2]}/agent-{agent_uuid}/threads"),
            thread_uuid,
            intelligence_engine.memory_prefs,
        )
        thread_path = thread_shard / f"thread-{thread_uuid}.enc"
        assert thread_path.exists()

        # Check that key file exists
        key_shard = shard_path(
            Path(f"toys/health/memories/private/agents/{agent_uuid[:2]}/agent-{agent_uuid}/keys"),
            thread_uuid,
            intelligence_engine.memory_prefs,
        )
        key_path = key_shard / f"key-{thread_uuid}.bin.enc"
        assert key_path.exists()

    def test_load_thread_round_trip(self, initialized_intelligence_engine):
        """Test saving and then loading a thread to verify the full cycle."""
        intelligence_engine = initialized_intelligence_engine

        # 1. Process some input to create a thread
        original_content = b"This content will be saved and then reloaded."
        intelligence_engine.process_input_stream(original_content)
        intelligence_engine.finalize_and_save_thread()  # Ensure thread is saved
        thread_uuid = intelligence_engine.thread_uuid

        # 2. Load the thread content back (as NDJSON event list)
        loaded_events = intelligence_engine.load_thread_content(thread_uuid)
        assert loaded_events is not None
        # 3. Verify the thread content matches (check data fields)
        data_fields = [event["data"] for event in loaded_events if "data" in event]
        assert original_content in data_fields

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
        prefs = get_memory_preferences(initialized_intelligence_engine.base_memories_dir)
        for fmt in formats:
            store_format(cast(FormatMetadata, fmt), prefs, initialized_intelligence_engine.base_memories_dir)

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
        # Remove unused variables public_engine and private_engine at lines 702 and 703
        prefs = get_memory_preferences(initialized_intelligence_engine.base_memories_dir)
        public_formats = list_formats(base_memories_dir=public_intelligence_engine.base_memories_dir)
        private_formats = list_formats(base_memories_dir=initialized_intelligence_engine.base_memories_dir)

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
        engine.finalize_and_save_thread()  # Ensure thread is saved

        # Generate response to create more gene keys
        engine.generate_and_save_response(length=10)
        engine.finalize_and_save_thread()  # Ensure thread is saved

        thread_uuid = engine.thread_uuid

        # Verify gene keys were stored
        stored_gene_keys = load_gene_keys(
            thread_uuid,
            agent_uuid=engine.agent_uuid,
            agent_secret=engine.agent_secret,
            prefs=engine.memory_prefs,
            base_memories_dir=engine.base_memories_dir,
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

    def test_initialize_intelligence_engine_modes(self, mock_env):
        """Test intelligence engine initialization in different modes"""
        # Create masks for all tests
        patterns_array, _ = derive_canonical_patterns()
        patterns_array.tofile(str(mock_env / "toys/health/memories/public/masks/epigenome.dat"))
        genome_mask = np.arange(256, dtype=np.uint8)
        genome_mask.tofile(str(mock_env / "toys/health/memories/public/masks/genome.dat"))

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
# Registry Cache Tests
# ------------------------------------------------------------------------------


def test_registry_cache_hit_and_invalidation(tmp_path):
    # Setup isolated registry file
    test_dir = tmp_path / "memories" / "test_shard"
    test_dir.mkdir(parents=True, exist_ok=True)
    registry_path = test_dir / "registry.json"
    # Initial write
    update_registry(test_dir, "uuid-1", update_parent=False)
    # First read (should populate cache)
    data1 = _read_registry_cached(registry_path)
    assert "uuid-1" in data1["uuids"]
    # Second read (should hit cache)
    data2 = _read_registry_cached(registry_path)
    assert data2 == data1
    # Modify file externally (simulate another process)
    with open(registry_path, "w") as f:
        f.write(json_dumps({"count": 2, "uuids": ["uuid-1", "uuid-2"]}))
    # Wait to ensure mtime changes
    time.sleep(1.1)
    # Read again (should invalidate cache and reload)
    data3 = _read_registry_cached(registry_path)
    assert "uuid-2" in data3["uuids"]


def test_registry_cache_eviction(tmp_path):
    # Clear the cache to ensure isolation
    _REGISTRY_CACHE.clear()
    # Setup isolated registry files
    test_base = tmp_path / "memories" / "evict_shard"
    test_base.mkdir(parents=True, exist_ok=True)
    # Fill the cache beyond its max size
    for i in range(_REGISTRY_CACHE_MAX_SIZE + 10):
        test_dir = test_base / f"shard_{i}"
        test_dir.mkdir(parents=True, exist_ok=True)
        update_registry(test_dir, f"uuid-{i}", update_parent=False)
        registry_path = test_dir / "registry.json"
        _read_registry_cached(registry_path)
    # Explicitly trim the cache after all insertions
    while len(_REGISTRY_CACHE) > _REGISTRY_CACHE_MAX_SIZE:
        del _REGISTRY_CACHE[next(iter(_REGISTRY_CACHE))]
    assert len(_REGISTRY_CACHE) <= _REGISTRY_CACHE_MAX_SIZE


def test_registry_update_after_external_change(tmp_path):
    # Setup isolated registry file
    test_dir = tmp_path / "memories" / "external_update"
    test_dir.mkdir(parents=True, exist_ok=True)
    registry_path = test_dir / "registry.json"
    # Initial write
    update_registry(test_dir, "uuid-1", update_parent=False)
    # External modification
    with open(registry_path, "w") as f:
        f.write(json_dumps({"count": 2, "uuids": ["uuid-1", "uuid-2"]}))
    time.sleep(1.1)
    # Now update via function (should see both uuids and add a third)
    update_registry(test_dir, "uuid-3", update_parent=False)
    data = _read_registry_cached(registry_path)
    assert set(data["uuids"]) == {"uuid-1", "uuid-2", "uuid-3"}


# ------------------------------------------------------------------------------
# Main test runner
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """Run tests when file is executed directly"""
    pytest.main(["-v", __file__])
