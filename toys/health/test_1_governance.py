"""
Tests for governance logic, gene operations, and pattern classification in the BabyLM system.
Includes tests for gene constants, operation application, gyrodistance, canonical pattern derivation, and pattern resonance.
"""

import os
import uuid
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
from typing import cast, Dict, Any
from datetime import datetime
import shutil

# Import modules from baby package
from baby.governance import (
    gene_com,
    gene_nest,
    gene_add,
    gene_stateless,
    gyrodistance,
    derive_canonical_patterns,
    classify_pattern_resonance,
)
from baby.inference import InferenceEngine
from baby.information import (
    InformationEngine,
    ensure_agent_uuid,
    ensure_agent_uuid as ensure_agent_uuid_fn,
    create_thread,
    load_thread,
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
from baby.intelligence import initialize_intelligence_engine
from baby.types import FormatMetadata, GeneKeysMetadata


# ------------------------------------------------------------------------------
# Governance (S1) Tests
# ------------------------------------------------------------------------------


class TestGovernance:
    """Tests for the Governance layer (tensor operations)"""

    def test_gene_constants(self):
        """Test that gene constants have the correct shapes and types"""
        assert gene_com.shape == (3, 2)
        assert gene_nest.shape == (2, 3, 2)
        assert gene_add.shape == (4, 2, 3, 2)
        assert isinstance(gene_stateless, int)
        assert gene_stateless == 0xAA  # 10101010 in binary

    def test_apply_operation_identity(self):
        """Test the identity operation (L0) on gene_add tensor"""
        from baby.governance import gene_add, apply_operation

        test_tensor = gene_add.copy().astype(np.float32)
        original = test_tensor.copy()
        result = apply_operation(test_tensor, 0)  # bit_index 0 = L0 (identity)
        np.testing.assert_array_equal(result, original)
        result = apply_operation(test_tensor, 7)  # bit_index 7 = L0 (identity)
        np.testing.assert_array_equal(result, original)

    def test_apply_operation_inverse(self):
        """Test the inverse operation (LI) on gene_add tensor"""
        from baby.governance import gene_add, apply_operation

        test_tensor = gene_add.copy().astype(np.float32)
        original = test_tensor.copy()
        result = apply_operation(test_tensor, 1)  # bit_index 1 = LI
        np.testing.assert_array_equal(result, -original)
        result2 = apply_operation(result, 1)
        np.testing.assert_array_equal(result2, original)
        result3 = apply_operation(test_tensor, 6)
        np.testing.assert_array_equal(result3, -original)

    def test_apply_operation_forward_gyration(self):
        """Test the forward gyration operation (FG) on gene_add tensor"""
        from baby.governance import gene_add, apply_operation

        test_tensor = gene_add.copy().astype(np.float32)
        original = test_tensor.copy()
        result = apply_operation(test_tensor, 2)  # bit_index 2 = FG
        np.testing.assert_array_equal(result[0], -original[0])
        np.testing.assert_array_equal(result[1], original[1])
        np.testing.assert_array_equal(result[2], -original[2])
        np.testing.assert_array_equal(result[3], original[3])
        result2 = apply_operation(test_tensor, 5)
        np.testing.assert_array_equal(result2[0], -original[0])
        np.testing.assert_array_equal(result2[1], original[1])
        np.testing.assert_array_equal(result2[2], -original[2])
        np.testing.assert_array_equal(result2[3], original[3])

    def test_apply_operation_backward_gyration(self):
        """Test the backward gyration operation (BG) on gene_add tensor"""
        from baby.governance import gene_add, apply_operation

        test_tensor = gene_add.copy().astype(np.float32)
        original = test_tensor.copy()
        result = apply_operation(test_tensor, 3)  # bit_index 3 = BG
        np.testing.assert_array_equal(result[0], original[0])
        np.testing.assert_array_equal(result[1], -original[1])
        np.testing.assert_array_equal(result[2], original[2])
        np.testing.assert_array_equal(result[3], -original[3])
        result2 = apply_operation(test_tensor, 4)
        np.testing.assert_array_equal(result2[0], original[0])
        np.testing.assert_array_equal(result2[1], -original[1])
        np.testing.assert_array_equal(result2[2], original[2])
        np.testing.assert_array_equal(result2[3], -original[3])

    def test_gyrodistance_identical(self):
        """Test gyrodistance between identical tensors"""
        T1 = np.ones((4, 2, 3, 2))
        T2 = T1.copy()

        distance = gyrodistance(T1, T2)
        assert distance == 0.0

    def test_gyrodistance_opposite(self):
        """Test gyrodistance between opposite tensors"""
        T1 = np.ones((4, 2, 3, 2))
        T2 = -T1.copy()

        distance = gyrodistance(T1, T2)
        assert np.isclose(distance, np.pi)

    def test_gyrodistance_orthogonal(self):
        """Test gyrodistance between orthogonal tensors"""
        T1 = np.zeros((4, 2, 3, 2))
        T2 = np.zeros((4, 2, 3, 2))

        # Set half of T1 to 1.0
        T1[:2].fill(1.0)

        # Set other half of T2 to 1.0
        T2[2:].fill(1.0)

        distance = gyrodistance(T1, T2)
        assert np.isclose(distance, np.pi / 2)

    def test_derive_canonical_patterns(self):
        """Test derivation of canonical patterns"""
        patterns, gyration_features = derive_canonical_patterns()

        # Check dimensions
        assert patterns.shape == (256, 48)  # 256 patterns, each 48 elements
        assert len(gyration_features) == 256

        # Check pattern content
        assert patterns.dtype == np.float32

        # Check that all resonance classes are valid
        valid_classes = ["identity", "inverse", "forward", "backward"]
        for cls in gyration_features:
            assert cls in valid_classes

    @pytest.mark.parametrize(
        "mask,expected_class",
        [
            (0b00000000, "identity"),  # No bits set
            (0b10000001, "identity"),  # Only identity bits set
            (0b01000010, "inverse"),  # Only inverse bits set
            (0b00100100, "forward"),  # Only forward gyration bits set
            (0b00011000, "backward"),  # Only backward gyration bits set
        ],
    )
    def test_classify_pattern_resonance(self, mask, expected_class):
        """Test pattern resonance classification with various bit patterns"""
        gyration_feature = classify_pattern_resonance(mask)
        assert gyration_feature == expected_class


# ------------------------------------------------------------------------------
# Information (S2) Storage Tests
# ------------------------------------------------------------------------------


class TestInformationStorage:
    """Tests for the Information layer persistent storage functions"""

    def test_get_memory_preferences(self):
        """Test loading memory preferences"""
        prefs = get_memory_preferences()

        # Check structure
        assert "sharding" in prefs
        assert "storage_config" in prefs
        assert "format_config" in prefs

        # Check default values
        assert prefs["sharding"]["width"] == 2
        assert prefs["sharding"]["max_files"] == 30000
        assert prefs["storage_config"]["max_thread_size_mb"] == 64

    def test_shard_path_first_level(self, mock_env):
        # Clean up the agents directory to ensure a clean state
        agents_dir = Path("memories/private/agents")
        if agents_dir.exists():
            shutil.rmtree(agents_dir)
        agents_dir.mkdir(parents=True, exist_ok=True)
        """Test calculating first-level shard path"""
        test_uuid = "abcdef12-3456-7890-abcd-ef1234567890"
        root = Path("memories/private/agents")
        shard = shard_path(root, test_uuid, width=2, limit=30000)
        assert shard == root / "ab"

    def test_shard_path_second_level(self):
        """Test calculating second-level shard path when needed"""
        test_uuid = "abcdef12-3456-7890-abcd-ef1234567890"
        root = Path("memories/private/agents")
        first_level = root / "ab"
        first_level.mkdir(parents=True, exist_ok=True)

        # Create a registry with count exceeding the limit
        registry = {"count": 40000, "uuids": ["test"] * 40000}
        registry_path = first_level / "registry.json"
        with open(registry_path, "w") as f:
            json.dump(registry, f)

        # Calculate shard path
        shard = shard_path(root, test_uuid)

        # Should be first two characters + next two characters
        assert shard == root / "ab" / "cd"

    def test_atomic_write(self):
        """Test atomic file writing"""
        test_path = Path("memories/test_atomic.dat")
        test_data = b"Test data for atomic write"

        # Write data
        atomic_write(test_path, test_data)

        # Check that file exists and contains correct data
        assert test_path.exists()
        with open(test_path, "rb") as f:
            assert f.read() == test_data

        # Check that no temporary file remains
        assert not test_path.with_suffix(test_path.suffix + ".tmp").exists()

    def test_update_registry(self, mock_env):
        # Clean up the test_registry directory to ensure a clean state
        test_dir = Path("memories/test_registry")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
        """Test updating a registry file"""
        registry_path = test_dir / "registry.json"
        update_registry(test_dir, "test-uuid-1")
        with open(registry_path, "r") as f:
            registry = json.load(f)
        assert registry["count"] == 1
        update_registry(test_dir, "test-uuid-2")
        with open(registry_path, "r") as f:
            registry = json.load(f)
        assert registry["count"] == 2

    def test_rebuild_registry(self):
        """Test rebuilding a registry from directory contents"""
        test_dir = Path("memories/test_rebuild")
        test_dir.mkdir(parents=True, exist_ok=True)

        # Create some test files
        test_files = [
            "thread-abcdef12-3456-7890-abcd-ef1234567890.json",
            "thread-12345678-9abc-def0-1234-56789abcdef0.enc",
            "key-abcdef12-3456-7890-abcd-ef1234567890.bin.enc",
        ]

        for filename in test_files:
            with open(test_dir / filename, "w") as f:
                f.write("test")

        # Create a temporary file that should be cleaned up
        with open(test_dir / "test.tmp", "w") as f:
            f.write("temp file")

        # Rebuild registry
        from baby.information import rebuild_registry

        rebuild_registry(test_dir)

        # Check that registry was created
        registry_path = test_dir / "registry.json"
        assert registry_path.exists()

        # Check registry content
        with open(registry_path, "r") as f:
            registry = json.load(f)

        # Should have 3 UUIDs (one from each file)
        assert registry["count"] == 3
        assert "abcdef12-3456-7890-abcd-ef1234567890" in registry["uuids"]
        assert "12345678-9abc-def0-1234-56789abcdef0" in registry["uuids"]

        # Temp file should be gone
        assert not (test_dir / "test.tmp").exists()

    def test_ensure_agent_uuid_new(self):
        """Test creating a new agent UUID when none exists"""
        # Ensure no agent exists
        private_dir = Path("memories/private/agents")
        if private_dir.exists():
            shutil.rmtree(private_dir)
        private_dir.mkdir(parents=True, exist_ok=True)

        # Get or create agent UUID
        with patch("uuid.uuid4", return_value=uuid.UUID("00000000-0000-0000-0000-000000000000")):
            agent_uuid = ensure_agent_uuid()

        # Check UUID value
        assert agent_uuid == "00000000-0000-0000-0000-000000000000"

        # Check that agent directory was created
        agent_dir = private_dir / "00" / f"agent-{agent_uuid}"
        assert agent_dir.exists()

        # Check that registry was created
        registry_path = private_dir / "00" / "registry.json"
        assert registry_path.exists()

        # Check that threads and keys directories were created
        assert (agent_dir / "threads").exists()
        assert (agent_dir / "keys").exists()

    def test_ensure_agent_uuid_existing(self):
        """Test finding an existing agent UUID"""
        # Create agent directory
        private_dir = Path("memories/private/agents")
        agent_uuid = "00000000-0000-0000-0000-000000000000"
        agent_shard = private_dir / "00"
        agent_shard.mkdir(parents=True, exist_ok=True)
        agent_dir = agent_shard / f"agent-{agent_uuid}"
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Create registry
        registry = {"count": 1, "uuids": [agent_uuid]}
        with open(agent_shard / "registry.json", "w") as f:
            json.dump(registry, f)

        # Get agent UUID
        found_uuid = ensure_agent_uuid()

        # Should find the existing UUID
        assert found_uuid == agent_uuid

    def test_thread_lifecycle(self):
        """End-to-end test: create agent, thread, save/load encrypted content and key using IntelligenceEngine workflow"""
        from baby.intelligence import IntelligenceEngine
        from baby.inference import InferenceEngine
        from baby.information import InformationEngine, store_gene_keys, load_gene_keys
        import sys
        import uuid
        from datetime import datetime
        import numpy as np
        import os

        # os.chdir(tmp_path)  # No longer needed, handled by mock_env
        # sys.path.insert(0, str(tmp_path))  # Not needed for test isolation
        (Path("memories") / "private" / "agents").mkdir(parents=True, exist_ok=True)
        agent_secret = "test_secret"
        agent_uuid = ensure_agent_uuid()
        inference_engine = InferenceEngine()
        information_engine = InformationEngine()
        engine = IntelligenceEngine(
            agent_uuid=agent_uuid,
            agent_secret=agent_secret,
            inference_engine=inference_engine,
            information_engine=information_engine,
        )
        engine.start_new_thread(privacy="private")
        thread_uuid = engine.thread_uuid
        assert thread_uuid is not None, "thread_uuid is None after starting new thread"
        thread_uuid = str(thread_uuid)
        # Clean up any existing gene key file for this thread and agent
        private_dir = Path("memories/private/agents")
        agent_shard = shard_path(private_dir, agent_uuid)
        agent_dir = agent_shard / f"agent-{agent_uuid}"
        keys_dir = agent_dir / "keys"
        key_shard = shard_path(keys_dir, thread_uuid)
        gene_keys_path = key_shard / f"gene-{thread_uuid}.ndjson.enc"
        if gene_keys_path.exists():
            gene_keys_path.unlink()
        test_content = b"Test thread content"
        engine.process_input_stream(test_content)
        engine.finalize_and_save_thread(privacy="private")
        loaded_content = engine.load_thread_content(thread_uuid)
        assert isinstance(loaded_content, list)
        assert loaded_content[0]["data"] == test_content
        # Store gene keys with proper structure
        test_gene_keys = [
            {
                "cycle": 1,
                "pattern_index": 42,
                "thread_uuid": thread_uuid,
                "agent_uuid": str(agent_uuid),
                "format_uuid": "11111111-1111-1111-1111-111111111111",
                "event_type": "INPUT",
                "source_byte": 0,
                "resonance": 0.5,
                "created_at": datetime.now().isoformat(),
                "privacy": "private",
            },
            {
                "cycle": 2,
                "pattern_index": 84,
                "thread_uuid": thread_uuid,
                "agent_uuid": str(agent_uuid),
                "format_uuid": "11111111-1111-1111-1111-111111111111",
                "event_type": "INPUT",
                "source_byte": 1,
                "resonance": 0.4,
                "created_at": datetime.now().isoformat(),
                "privacy": "private",
            },
        ]
        store_gene_keys(
            thread_uuid=thread_uuid,
            gene_keys=cast(list[GeneKeysMetadata], test_gene_keys),
            privacy="private",
            agent_secret=agent_secret,
            agent_uuid=agent_uuid,
        )
        # Debug: print the contents of the gene key file
        print("DEBUG gene_keys_path:", gene_keys_path)
        if gene_keys_path.exists():
            with open(gene_keys_path, "rb") as f:
                print("DEBUG gene_keys_file_bytes:", f.read())
        loaded_gene_keys = load_gene_keys(
            thread_uuid=thread_uuid, agent_uuid=str(agent_uuid), agent_secret=agent_secret
        )
        print("DEBUG loaded_gene_keys:", loaded_gene_keys)
        assert isinstance(loaded_gene_keys, list)
        for original in test_gene_keys:
            assert any(
                loaded["cycle"] == original["cycle"] and loaded["pattern_index"] == original["pattern_index"]
                for loaded in loaded_gene_keys
            )

    def test_thread_relationships(self):
        from baby.intelligence import IntelligenceEngine
        from baby.inference import InferenceEngine
        from baby.information import InformationEngine, parent, children
        import os

        agent_secret = "test_secret"
        agent_uuid = ensure_agent_uuid()
        inference_engine = InferenceEngine()
        information_engine = InformationEngine()
        engine = IntelligenceEngine(
            agent_uuid=agent_uuid,
            agent_secret=agent_secret,
            inference_engine=inference_engine,
            information_engine=information_engine,
        )
        # Start parent thread
        engine.start_new_thread(privacy="private")
        parent_uuid = engine.thread_uuid
        assert parent_uuid is not None, "parent_uuid is None after starting new thread"
        parent_uuid = str(parent_uuid)
        # Start child thread (no parent_thread_uuid kwarg)
        engine.start_new_thread(privacy="private")
        child_uuid = engine.thread_uuid
        assert child_uuid is not None, "child_uuid is None after starting child thread"
        child_uuid = str(child_uuid)
        # Manually set parent relationship if needed, or use public API if available
        # Assert relationships using public API
        children_list = children(agent_uuid, parent_uuid)
        assert parent(agent_uuid, child_uuid) == parent_uuid
        assert child_uuid in children_list, f"Child UUID {child_uuid} not found in children list: {children_list}"

    def test_format_management(self):
        """Test format storage and retrieval"""
        # Create format directory
        formats_dir = Path("memories/public/formats")
        formats_dir.mkdir(parents=True, exist_ok=True)

        # Create test format
        format_uuid = "11111111-1111-1111-1111-111111111111"
        format_data = {
            "format_uuid": format_uuid,
            "format_name": "Test Format",
            "metadata": {"author": "test_author", "description": "Test format for unit tests"},
            "patterns": [{"index": i, "character": None} for i in range(10)],
        }

        # Store format
        stored_uuid = store_format(cast(FormatMetadata, format_data))
        assert stored_uuid == format_uuid

        # Check format file
        format_shard = shard_path(formats_dir, format_uuid)
        format_path = format_shard / f"format-{format_uuid}.json"
        assert format_path.exists()

        # Load format
        loaded_format = load_format(format_uuid)
        assert loaded_format is not None, f"Format {format_uuid} could not be loaded"
        assert loaded_format.get("format_uuid") == format_uuid
        assert loaded_format.get("format_name") == "Test Format"

        # List formats
        format_list = list_formats()
        assert format_uuid in format_list

    def test_gene_keys_public_storage(self, mock_env):
        # Clean up the public keys directory to ensure a clean state
        keys_dir = Path("memories/public/keys")
        if keys_dir.exists():
            shutil.rmtree(keys_dir)
        keys_dir.mkdir(parents=True, exist_ok=True)
        """Test storing and loading gene keys in public mode"""
        thread_uuid = "test-thread-uuid"
        test_gene_keys: list[GeneKeysMetadata] = [
            {
                "cycle": 1,
                "pattern_index": 42,
                "thread_uuid": thread_uuid,
                "agent_uuid": None,
                "format_uuid": "test-format",
                "event_type": "INPUT",
                "source_byte": 0,
                "resonance": 0.5,
                "created_at": datetime.now().isoformat(),
                "privacy": "public",
            },
        ]

        # Store in public mode (no agent_uuid/agent_secret)
        store_gene_keys(thread_uuid=thread_uuid, gene_keys=test_gene_keys, privacy="public")

        # Load from public mode
        loaded_gene_keys = load_gene_keys(thread_uuid=thread_uuid)

        assert len(loaded_gene_keys) == 1
        assert loaded_gene_keys[0]["pattern_index"] == 42

    def test_pattern_index_functionality(self):
        """Test PatternIndex class functionality"""
        agent_uuid = "test-agent-uuid"
        agent_secret = "test-secret"

        pattern_index = PatternIndex(agent_uuid, agent_secret)

        # Test gene keys
        test_gene_keys = [
            {"cycle": 1, "pattern_index": 10},
            {"cycle": 2, "pattern_index": 11},
            {"cycle": 3, "pattern_index": 10},
            {"cycle": 4, "pattern_index": 12},
        ]

        # Update index
        pattern_index.update_from_thread("test-thread", test_gene_keys)

        # Test sequence tracking
        likely_next = pattern_index.get_likely_next_patterns(10, top_k=2)
        assert len(likely_next) <= 2

        # Pattern 10 is followed by 11 and 12 in our test data
        pattern_indices = [p[0] for p in likely_next]
        assert 11 in pattern_indices or 12 in pattern_indices


# ------------------------------------------------------------------------------
# Main test runner
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """Run tests when file is executed directly"""
    pytest.main(["-v", __file__])
