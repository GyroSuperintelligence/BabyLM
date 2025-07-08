"""
Comprehensive test suite for GyroSI Baby LM

This test suite covers all major components of the GyroSI Baby LM:
- Governance (S1): Core tensor operations
- Inference (S3): Pattern recognition
- Information (S2): Stream processing and storage
- Intelligence (S4): Orchestration and thread lifecycle

Run with: pytest toys/tests/baby_tests.py -v
"""

import os
import uuid
import json
import numpy as np
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import cast

# Import modules from baby package
from baby.governance import (
    gene_com,
    gene_nest,
    gene_add,
    gene_stateless,
    apply_operation,
    gyrodistance,
    derive_canonical_patterns,
    classify_pattern_resonance,
)
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
)
from baby.intelligence import IntelligenceEngine, weighted_choice, initialize_intelligence_engine
from baby.types import FormatMetadata

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
    initialized IntelligenceEngine instance ready for testing.
    """
    # 1. Create default baby_preferences.json
    baby_prefs_path = Path("baby/baby_preferences.json")
    baby_prefs = {
        "agent_secret": str(uuid.uuid4()),
        "log_level": "info",
        "response_length": 100,
        "learning_rate": 1.0,
    }
    with open(baby_prefs_path, "w") as f:
        json.dump(baby_prefs, f, indent=2)

    # 2. Create the masks required for InferenceEngine
    patterns_array, _ = derive_canonical_patterns()
    patterns_array.tofile("memories/public/masks/epigenome.dat")

    genome_mask = np.arange(256, dtype=np.uint8)  # Identity mapping for predictability
    genome_mask.tofile("memories/public/masks/genome.dat")

    # 3. Initialize the engine. This will create the agent directory and format files.
    engine = initialize_intelligence_engine()

    # Assert that the keys directory exists
    private_dir = Path("memories/private/agents")
    agent_shard = shard_path(private_dir, engine.agent_uuid)
    agent_dir = agent_shard / f"agent-{engine.agent_uuid}"
    keys_dir = agent_dir / "keys"
    assert keys_dir.exists() and keys_dir.is_dir(), f"Keys directory was not created: {keys_dir}"

    return engine


@pytest.fixture(scope="session", autouse=True)
def cleanup_htmlcov():
    yield
    if os.path.exists("htmlcov"):
        shutil.rmtree("htmlcov")


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

    def test_apply_operation_identity(self, test_tensor):
        """Test the identity operation (L0)"""
        original = test_tensor.copy()
        apply_operation(test_tensor, 0)  # bit_index 0 = L0 (identity)
        np.testing.assert_array_equal(test_tensor, original)

        apply_operation(test_tensor, 7)  # bit_index 7 = L0 (identity)
        np.testing.assert_array_equal(test_tensor, original)

    def test_apply_operation_inverse(self, test_tensor):
        """Test the inverse operation (LI)"""
        # Initialize with non-zero values
        test_tensor.fill(1.0)
        original = test_tensor.copy()

        # Apply LI (Left Inverse)
        apply_operation(test_tensor, 1)  # bit_index 1 = LI
        np.testing.assert_array_equal(test_tensor, -original)

        # Apply LI again (should return to original)
        apply_operation(test_tensor, 1)
        np.testing.assert_array_equal(test_tensor, original)

        # Test bit_index 6 (also LI)
        apply_operation(test_tensor, 6)
        np.testing.assert_array_equal(test_tensor, -original)

    def test_apply_operation_forward_gyration(self, test_tensor):
        """Test the forward gyration operation (FG)"""
        # Initialize with different values for each row
        for i in range(4):
            test_tensor[i].fill(float(i + 1))

        original = test_tensor.copy()

        # Apply FG (Forward Gyration)
        apply_operation(test_tensor, 2)  # bit_index 2 = FG

        # Check that rows 0 and 2 are negated
        np.testing.assert_array_equal(test_tensor[0], -original[0])
        np.testing.assert_array_equal(test_tensor[1], original[1])
        np.testing.assert_array_equal(test_tensor[2], -original[2])
        np.testing.assert_array_equal(test_tensor[3], original[3])

        # Test bit_index 5 (also FG)
        test_tensor = original.copy()
        apply_operation(test_tensor, 5)
        np.testing.assert_array_equal(test_tensor[0], -original[0])
        np.testing.assert_array_equal(test_tensor[1], original[1])
        np.testing.assert_array_equal(test_tensor[2], -original[2])
        np.testing.assert_array_equal(test_tensor[3], original[3])

    def test_apply_operation_backward_gyration(self, test_tensor):
        """Test the backward gyration operation (BG)"""
        # Initialize with different values for each row
        for i in range(4):
            test_tensor[i].fill(float(i + 1))

        original = test_tensor.copy()

        # Apply BG (Backward Gyration)
        apply_operation(test_tensor, 3)  # bit_index 3 = BG

        # Check that rows 1 and 3 are negated
        np.testing.assert_array_equal(test_tensor[0], original[0])
        np.testing.assert_array_equal(test_tensor[1], -original[1])
        np.testing.assert_array_equal(test_tensor[2], original[2])
        np.testing.assert_array_equal(test_tensor[3], -original[3])

        # Test bit_index 4 (also BG)
        test_tensor = original.copy()
        apply_operation(test_tensor, 4)
        np.testing.assert_array_equal(test_tensor[0], original[0])
        np.testing.assert_array_equal(test_tensor[1], -original[1])
        np.testing.assert_array_equal(test_tensor[2], original[2])
        np.testing.assert_array_equal(test_tensor[3], -original[3])

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
        patterns, gyration_featurees = derive_canonical_patterns()

        # Check dimensions
        assert patterns.shape == (256, 48)  # 256 patterns, each 48 elements
        assert len(gyration_featurees) == 256

        # Check pattern content
        assert patterns.dtype == np.float32

        # Check that all resonance classes are valid
        valid_classes = ["identity", "inverse", "forward", "backward"]
        for cls in gyration_featurees:
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
# Inference (S3) Tests
# ------------------------------------------------------------------------------


class TestInference:
    """Tests for the Inference layer (pattern recognition)"""

    def test_inference_engine_init(self):
        """Test InferenceEngine initialization"""
        with (
            patch.object(InferenceEngine, "_load_patterns", return_value=(np.zeros((256, 48)), ["test"] * 256)),
            patch.object(InferenceEngine, "_load_genome_mask", return_value=np.arange(256, dtype=np.uint8)),
            patch.object(InferenceEngine, "_initialize_epigenome"),
        ):

            engine = InferenceEngine()

            assert engine.T.shape == (4, 2, 3, 2)
            assert engine.cycle_counter == 0
            assert len(engine.gyration_featurees) == 256
            assert engine.F.shape == (256, 48)
            assert engine.G.shape == (256,)

    def test_load_patterns_file_exists(self, mock_memories_dir):
        """Test loading patterns when file exists"""
        # Create a mock pattern file
        pattern_file = mock_memories_dir / "public/masks/epigenome.dat"
        pattern_file.parent.mkdir(parents=True, exist_ok=True)

        # Create test patterns
        test_patterns = np.ones((256, 48), dtype=np.float32)
        with open(pattern_file, "wb") as f:
            test_patterns.tofile(f)

        # Test loading
        engine = InferenceEngine()

        # Verify loaded patterns
        assert engine.F.shape == (256, 48)
        assert len(engine.gyration_featurees) == 256

        # Clean up
        pattern_file.unlink()

    def test_load_patterns_generate_new(self, mock_memories_dir):
        """Test pattern generation when file doesn't exist"""
        # Ensure pattern file doesn't exist
        pattern_file = mock_memories_dir / "public/masks/epigenome.dat"
        if pattern_file.exists():
            pattern_file.unlink()

        # Use actual implementation to load/generate patterns
        with patch("baby.inference.derive_canonical_patterns", return_value=(np.ones((256, 48)), ["test"] * 256)):
            engine = InferenceEngine()

            # Verify generated patterns
            assert engine.F.shape == (256, 48)
            assert len(engine.gyration_featurees) == 256

    def test_load_genome_mask_file_exists(self, mock_memories_dir):
        """Test loading genome mask when file exists"""
        # Create a mock genome file
        genome_file = mock_memories_dir / "public/masks/genome.dat"
        genome_file.parent.mkdir(parents=True, exist_ok=True)

        # Create test genome mask
        test_genome = np.arange(256, dtype=np.uint8)
        with open(genome_file, "wb") as f:
            test_genome.tofile(f)

        # Test loading
        with (
            patch.object(InferenceEngine, "_load_patterns", return_value=(np.zeros((256, 48)), ["test"] * 256)),
            patch.object(InferenceEngine, "_initialize_epigenome"),
        ):
            engine = InferenceEngine()

            # Verify loaded genome mask
            assert engine.G.shape == (256,)
            np.testing.assert_array_equal(engine.G, test_genome)

        # Clean up
        genome_file.unlink()

    def test_load_genome_mask_generate_new(self, mock_memories_dir):
        """Test genome mask generation when file doesn't exist"""
        # Ensure genome file doesn't exist
        genome_file = mock_memories_dir / "public/masks/genome.dat"
        if genome_file.exists():
            genome_file.unlink()

        # Test loading/generation
        with (
            patch.object(InferenceEngine, "_load_patterns", return_value=(np.zeros((256, 48)), ["test"] * 256)),
            patch.object(InferenceEngine, "_initialize_epigenome"),
        ):
            engine = InferenceEngine()

            # Verify generated genome mask
            assert engine.G.shape == (256,)
            np.testing.assert_array_equal(engine.G, np.arange(256, dtype=np.uint8))

    def test_initialize_epigenome(self):
        """Test Epigenome initialization"""
        with (
            patch.object(InferenceEngine, "_load_patterns", return_value=(np.zeros((256, 48)), ["test"] * 256)),
            patch.object(InferenceEngine, "_load_genome_mask", return_value=np.arange(256, dtype=np.uint8)),
        ):

            engine = InferenceEngine()

            # Capture initial tensor state after initialization
            initial_T = engine.T.copy()

            # Reset tensor to zeros
            engine.T.fill(0.0)

            # Manually initialize epigenome
            engine._initialize_epigenome()

            # Verify initialization effect
            np.testing.assert_array_equal(engine.T, initial_T)
            assert engine.cycle_counter == 0

    def test_process_byte(self, inference_engine):
        """Test processing a byte with explicit mutation logic"""
        # Set up a non-uniform tensor
        inference_engine.T = np.arange(4 * 2 * 3 * 2, dtype=np.float32).reshape(4, 2, 3, 2)
        initial_T = inference_engine.T.copy()
        input_byte = 0x55
        gene_mutated = input_byte ^ 0xAA

        # Manually apply the expected operations
        expected_T = initial_T.copy()
        for i in range(8):
            if gene_mutated & (1 << i):
                apply_operation(expected_T, i)

        # Call the method under test
        result = inference_engine.process_byte(input_byte)

        # Assert the tensor matches the expected result
        np.testing.assert_array_equal(inference_engine.T, expected_T)

        # Check that cycle counter incremented
        assert inference_engine.cycle_counter == 1

        # Check return value is a valid index
        assert isinstance(result, int)
        assert 0 <= result < 256

    def test_find_closest_pattern_index(self, inference_engine):
        """Test finding the closest pattern"""
        # Set tensor to all ones
        inference_engine.T.fill(1.0)
        # Set all patterns to zeros except index 42, which matches the tensor
        inference_engine.F = np.zeros((256, 48))
        inference_engine.F[42] = np.ones(48)  # Make pattern 42 closest

        result = inference_engine.find_closest_pattern_index()
        assert result == 42

    def test_canonical_pattern_matching(self):
        # Generate canonical patterns
        patterns, _ = derive_canonical_patterns()
        # Test several masks for robustness
        for mask in [0, 1, 37, 128, 255]:
            # Generate the tensor by applying the mask to the base tensor
            base_tensor = gene_add.copy().astype(np.float32)
            T = base_tensor.copy()
            for i in range(8):
                if mask & (1 << i):
                    apply_operation(T, i)
            flat_T = T.flatten().astype(np.float32)
            # The pattern at index 'mask' should match the tensor we generated
            assert np.allclose(flat_T, patterns[mask]), f"Pattern mismatch for mask {mask}"

    def test_compute_pattern_resonances(self, inference_engine):
        """Test computing pattern resonances"""
        # Create a controlled set of patterns for testing
        inference_engine.F = np.zeros((256, 48))
        inference_engine.F[0] = np.ones(48)  # First pattern has all 1s
        inference_engine.F[1] = -np.ones(48)  # Second pattern has all -1s

        # Set current tensor to all 1s
        inference_engine.T.fill(1.0)

        # Compute resonances
        resonances = inference_engine.compute_pattern_resonances()

        # Check that first pattern has zero distance (perfect match)
        assert len(resonances) == 256
        assert resonances[0] == 0.0

        # Check that second pattern has maximum distance (opposite)
        assert np.isclose(resonances[1], np.pi)

        # Check that other patterns have intermediate distances
        for i in range(2, 256):
            assert 0.0 < resonances[i] <= np.pi

    def test_compute_contextual_resonances(self, inference_engine):
        """Test computing contextual pattern resonances"""
        # Set up recent patterns
        inference_engine.recent_patterns = [1, 2, 3, 4, 5]

        # Create pattern contexts
        pattern_contexts = {
            5: {
                "after": {10: 5, 20: 3, 30: 1},  # Pattern 5 is followed by 10 most often
                "before": {4: 4, 3: 2},  # Pattern 5 is preceded by 4 most often
            }
        }

        # Mock base resonances
        base_resonances = [0.5] * 256
        with patch.object(inference_engine, "compute_pattern_resonances", return_value=base_resonances):
            # Compute contextual resonances
            resonances = inference_engine.compute_contextual_resonances(pattern_contexts)

            # Pattern 10 should have reduced distance (higher likelihood)
            assert resonances[10] < base_resonances[10]

            # Check that total list length is still 256
            assert len(resonances) == 256


# ------------------------------------------------------------------------------
# Information (S2) Storage Tests
# ------------------------------------------------------------------------------


class TestInformationStorage:
    """Tests for the Information layer persistent storage functions"""

    def test_get_memory_preferences(self, mock_env):
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
        """Test calculating first-level shard path"""
        test_uuid = "abcdef12-3456-7890-abcd-ef1234567890"
        root = Path("memories/private/agents")

        # Calculate shard path
        shard = shard_path(root, test_uuid)

        # Should be first two characters of UUID
        assert shard == root / "ab"

    def test_shard_path_second_level(self, mock_env):
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

    def test_atomic_write(self, mock_env):
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
        """Test updating a registry file"""
        test_dir = Path("memories/test_registry")
        test_dir.mkdir(parents=True, exist_ok=True)

        # Update registry with a new UUID
        test_uuid = "abcdef12-3456-7890-abcd-ef1234567890"
        update_registry(test_dir, test_uuid)

        # Check that registry file exists
        registry_path = test_dir / "registry.json"
        assert registry_path.exists()

        # Check registry content
        with open(registry_path, "r") as f:
            registry = json.load(f)

        assert "count" in registry
        assert "uuids" in registry
        assert test_uuid in registry["uuids"]
        assert registry["count"] == 1

        # Update registry with another UUID
        test_uuid2 = "12345678-9abc-def0-1234-56789abcdef0"
        update_registry(test_dir, test_uuid2)

        # Check updated registry
        with open(registry_path, "r") as f:
            registry = json.load(f)

        assert test_uuid in registry["uuids"]
        assert test_uuid2 in registry["uuids"]
        assert registry["count"] == 2

    def test_rebuild_registry(self, mock_env):
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

    def test_ensure_agent_uuid_new(self, mock_env):
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

    def test_ensure_agent_uuid_existing(self, mock_env):
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

    def test_thread_lifecycle(self, tmp_path):
        """End-to-end test: create agent, thread, save/load encrypted content and key"""
        import os

        os.chdir(tmp_path)
        agent_secret = "test_secret"
        # Create agent and directories
        agent_uuid = ensure_agent_uuid()
        format_uuid = "11111111-1111-1111-1111-111111111111"
        # Create thread
        thread_uuid = create_thread(agent_uuid, None, format_uuid)
        # Save thread content
        test_content = b"Test thread content"
        save_thread(agent_uuid, thread_uuid, test_content, len(test_content))
        # Store thread key
        test_key = bytes([i % 256 for i in range(256)])
        store_thread_key(agent_uuid, thread_uuid, test_key, agent_secret)
        # Store gene keys
        test_gene_keys = [{"cycle": 1, "pattern_index": 42}, {"cycle": 2, "pattern_index": 84}]
        store_gene_keys(agent_uuid, thread_uuid, test_gene_keys, agent_secret)
        # Load thread content
        loaded_content = load_thread(agent_uuid, thread_uuid)
        assert loaded_content == test_content
        # Load thread key
        loaded_key = load_thread_key(agent_uuid, thread_uuid, agent_secret)
        assert loaded_key == test_key
        # Load gene keys
        loaded_gene_keys = load_gene_keys(agent_uuid, thread_uuid, agent_secret)
        assert loaded_gene_keys == test_gene_keys

    def test_thread_relationships(self, mock_env):
        """Test parent-child thread relationships"""
        # Create agent
        agent_uuid = "00000000-0000-0000-0000-000000000000"
        format_uuid = "11111111-1111-1111-1111-111111111111"

        # Ensure agent directory exists
        private_dir = Path("memories/private/agents")
        agent_shard = shard_path(private_dir, agent_uuid)
        agent_shard.mkdir(parents=True, exist_ok=True)
        agent_dir = agent_shard / f"agent-{agent_uuid}"
        agent_dir.mkdir(exist_ok=True)
        (agent_dir / "threads").mkdir(exist_ok=True)

        # Create registry
        update_registry(agent_shard, agent_uuid)
        update_registry(agent_dir / "threads", "")

        # Create parent thread
        with patch("uuid.uuid4", return_value=uuid.UUID("00000000-0000-0000-0000-000000000001")):
            parent_uuid = create_thread(agent_uuid, None, format_uuid)

        # Create child thread
        with patch("uuid.uuid4", return_value=uuid.UUID("00000000-0000-0000-0000-000000000002")):
            child_uuid = create_thread(agent_uuid, parent_uuid, format_uuid)

        # Check parent-child relationship
        assert parent(agent_uuid, child_uuid) == parent_uuid
        assert child_uuid in children(agent_uuid, parent_uuid)

    def test_format_management(self, mock_env):
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


# ------------------------------------------------------------------------------
# Information (S2) Processing Tests
# ------------------------------------------------------------------------------


class TestInformationProcessing:
    """Tests for the Information layer (stream processing)"""

    def test_information_engine_init(self):
        """Test InformationEngine initialization"""
        engine = InformationEngine()

        assert engine.stream_pointer == 0
        assert isinstance(engine.output_buffer, bytearray)
        assert len(engine.output_buffer) == 0

    def test_process_stream(self, information_engine, inference_engine):
        """Test processing an input stream"""
        # Create mock update callback
        update_callback = MagicMock()

        # Create test input stream
        test_input = b"Hello, world!"

        # Process the stream
        ciphertext, keystream = information_engine.process_stream(inference_engine, update_callback, test_input)

        # Check output properties
        assert len(ciphertext) == len(test_input)
        assert len(keystream) == len(test_input)

        # Check that update_callback was called for each byte
        assert update_callback.call_count == len(test_input)

        # Check stream pointer was advanced
        assert information_engine.stream_pointer == len(test_input)

    def test_process_generated_bytes(self, information_engine, inference_engine):
        """Test processing generated bytes"""
        # Create mock update callback
        update_callback = MagicMock()

        # Create test bytes
        test_bytes = b"Generated data"

        # Process the bytes
        information_engine.process_generated_bytes(inference_engine, update_callback, test_bytes)

        # Check that update_callback was called for each byte
        assert update_callback.call_count == len(test_bytes)

        # Check stream pointer was advanced
        assert information_engine.stream_pointer == len(test_bytes)

    def test_tensor_to_output_byte(self):
        """Test canonical tensor-to-byte conversion using pattern matching"""
        engine = InformationEngine()
        # Create canonical patterns and genome mask
        patterns = np.zeros((256, 48), dtype=np.float32)
        patterns[42] = np.arange(48, dtype=np.float32)  # Make pattern 42 unique
        genome_mask = np.arange(256, dtype=np.uint8)
        # Set tensor to match pattern 42
        T = patterns[42].reshape(4, 2, 3, 2)
        # Should return genome_mask[42]
        output_byte = engine.tensor_to_output_byte(T, patterns, genome_mask)
        assert output_byte == 42


# ------------------------------------------------------------------------------
# Intelligence (S4) Tests
# ------------------------------------------------------------------------------


class TestIntelligence:
    """Tests for the Intelligence layer (orchestration)"""

    def test_intelligence_engine_init(self, initialized_intelligence_engine):
        """Test IntelligenceEngine initialization"""
        # Use the actual UUIDs from the initialized engine, not hardcoded values
        assert isinstance(initialized_intelligence_engine.agent_uuid, str)
        assert isinstance(initialized_intelligence_engine.format_uuid, str)
        assert initialized_intelligence_engine.thread_uuid is None
        assert initialized_intelligence_engine.thread_file_key is None
        assert isinstance(initialized_intelligence_engine.M, dict)
        assert isinstance(initialized_intelligence_engine.memory_prefs, dict)
        assert hasattr(initialized_intelligence_engine, "pattern_index")

    def test_load_or_init_formats_existing(self, mock_env, initialized_intelligence_engine):
        """Test loading existing format metadata"""
        # Create format directory for the UUID
        formats_dir = Path("memories/public/formats")
        format_shard = shard_path(formats_dir, initialized_intelligence_engine.format_uuid)
        format_shard.mkdir(parents=True, exist_ok=True)

        # Create mock format file
        format_path = format_shard / f"format-{initialized_intelligence_engine.format_uuid}.json"

        mock_format = {
            "format_uuid": initialized_intelligence_engine.format_uuid,
            "format_name": "test_format",
            "patterns": [],
        }

        with open(format_path, "w") as f:
            json.dump(mock_format, f)

        # Test loading
        result = initialized_intelligence_engine._load_or_init_formats()

        # Verify loaded format
        assert result.get("format_uuid") == initialized_intelligence_engine.format_uuid
        assert result.get("format_name") == "test_format"

    def test_load_or_init_formats_new(self, mock_env, initialized_intelligence_engine):
        """Test initializing new format metadata"""
        # Ensure format file doesn't exist
        formats_dir = Path("memories/public/formats")
        format_shard = shard_path(formats_dir, initialized_intelligence_engine.format_uuid)
        format_path = format_shard / f"format-{initialized_intelligence_engine.format_uuid}.json"
        if format_path.exists():
            format_path.unlink()

        # Test initialization
        with patch.object(
            initialized_intelligence_engine,
            "_initialize_format_metadata",
            return_value={"format_uuid": initialized_intelligence_engine.format_uuid, "test": True},
        ):
            result = initialized_intelligence_engine._load_or_init_formats()

            # Verify initialized format
            assert result.get("format_uuid") == initialized_intelligence_engine.format_uuid
            assert result.get("test") is True

    def test_initialize_format_metadata(self, initialized_intelligence_engine):
        """Test format metadata initialization"""
        # Initialize new format metadata
        result = initialized_intelligence_engine._initialize_format_metadata()

        # Verify structure
        assert result.get("format_uuid") == initialized_intelligence_engine.format_uuid
        assert "patterns" in result
        assert len(result.get("patterns")) == 256
        assert "cgm_policies" in result

        # Check all required policies
        required_policies = ["governance", "information", "inference", "intelligence"]
        for policy in required_policies:
            assert policy in result.get("cgm_policies")

    def test_start_new_thread(self, initialized_intelligence_engine):
        """Test starting a new thread"""
        with patch.object(initialized_intelligence_engine, "_derive_file_key", return_value=b"test_key" * 32):
            # Start a new thread
            thread_uuid = initialized_intelligence_engine.start_new_thread()

            # Verify thread initialization
            assert thread_uuid == initialized_intelligence_engine.thread_uuid
            assert initialized_intelligence_engine.thread_file_key == b"test_key" * 32
            assert initialized_intelligence_engine.current_thread_keys == []

            # Check that thread file exists
            thread_shard = shard_path(
                Path(
                    f"memories/private/agents/{initialized_intelligence_engine.agent_uuid[:2]}/"
                    f"agent-{initialized_intelligence_engine.agent_uuid}/threads"
                ),
                thread_uuid,
            )
            thread_meta_path = thread_shard / f"thread-{thread_uuid}.json"
            assert thread_meta_path.exists()

            # Check that key file exists
            key_shard = shard_path(
                Path(
                    f"memories/private/agents/{initialized_intelligence_engine.agent_uuid[:2]}/"
                    f"agent-{initialized_intelligence_engine.agent_uuid}/keys"
                ),
                thread_uuid,
            )
            key_path = key_shard / f"key-{thread_uuid}.bin.enc"
            assert key_path.exists()

    def test_process_input_stream(self, initialized_intelligence_engine):
        """Test processing an input stream"""
        test_input = b"Test input stream"

        with (
            patch.object(
                initialized_intelligence_engine.information_engine,
                "process_stream",
                return_value=(b"ciphertext", b"keystream"),
            ),
            patch.object(initialized_intelligence_engine, "_append_to_thread"),
        ):
            # Process input stream
            plaintext, ciphertext = initialized_intelligence_engine.process_input_stream(test_input)

            # Verify results
            assert plaintext == test_input
            assert ciphertext == b"ciphertext"

            # Verify method calls
            initialized_intelligence_engine.information_engine.process_stream.assert_called_once()
            initialized_intelligence_engine._append_to_thread.assert_called_once_with(test_input)

    def test_end_current_thread(self, initialized_intelligence_engine):
        """Test ending a current thread"""
        # Set up thread state
        initialized_intelligence_engine.thread_uuid = "test-thread-uuid"
        initialized_intelligence_engine.thread_file_key = b"key" * 64  # 256 bytes
        initialized_intelligence_engine.current_thread_keys = [{"cycle": 1, "pattern_index": 42}]
        initialized_intelligence_engine.active_thread_content = b"Thread content to save"
        # Ensure format metadata is present
        if "metadata" not in initialized_intelligence_engine.M:
            initialized_intelligence_engine.M["metadata"] = {}
        initialized_intelligence_engine.M["metadata"].setdefault("last_updated", "2024-01-01T00:00:00")
        initialized_intelligence_engine.M["metadata"].setdefault("usage_count", 0)

        with (
            patch("baby.intelligence.save_thread") as mock_save_thread,
            patch("baby.intelligence.store_gene_keys") as mock_store_gene_keys,
            patch("baby.intelligence.store_format") as mock_store_format,
        ):
            # End thread
            initialized_intelligence_engine.end_current_thread(plaintext_to_save=b"Thread content to save")

            # Verify save calls
            mock_save_thread.assert_called_once()
            mock_store_gene_keys.assert_called_once()
            mock_store_format.assert_called_once()

    def test_generate_and_save_response(self, initialized_intelligence_engine):
        """Test generating and saving a response"""
        with (
            patch.object(
                initialized_intelligence_engine, "_generate_response_bytes", return_value=b"Generated response"
            ),
            patch.object(initialized_intelligence_engine, "_append_to_thread"),
        ):
            # Generate response
            response = initialized_intelligence_engine.generate_and_save_response(length=20)

            # Verify results
            assert response == b"Generated response"

            # Verify method calls
            initialized_intelligence_engine._generate_response_bytes.assert_called_once_with(20)
            initialized_intelligence_engine._append_to_thread.assert_called_once_with(b"Generated response")

    def test_generate_response_bytes(self, initialized_intelligence_engine):
        """Test generating response bytes"""
        with (
            patch.object(
                initialized_intelligence_engine, "_generate_response_byte", side_effect=[(97, 0), (98, 1), (99, 2)]
            ),
            patch.object(initialized_intelligence_engine.inference_engine, "process_byte"),
            patch.object(initialized_intelligence_engine, "update_learning_state"),
        ):
            # Generate 3 bytes
            response = initialized_intelligence_engine._generate_response_bytes(3)

            # Verify the output
            assert response == b"abc"

            # Check method calls
            assert initialized_intelligence_engine._generate_response_byte.call_count == 3
            assert initialized_intelligence_engine.inference_engine.process_byte.call_count == 3
            assert initialized_intelligence_engine.update_learning_state.call_count == 3

    def test_generate_response_byte(self, initialized_intelligence_engine):
        """Test generating a single response byte"""
        # Mock pattern index and resonances
        with (
            patch.object(
                initialized_intelligence_engine.inference_engine,
                "compute_contextual_resonances",
                return_value=[0.5] * 256,
            ),
            patch("random.choice", return_value=42),
        ):
            # Generate response byte
            byte, pattern_idx = initialized_intelligence_engine._generate_response_byte()

            # Should return output byte from G[pattern_idx]
            assert byte == initialized_intelligence_engine.inference_engine.G[pattern_idx]

    def test_update_learning_state(self, initialized_intelligence_engine):
        """Test updating learning state"""
        # Initialize pattern data
        pattern_index = 42
        initialized_intelligence_engine.M["patterns"][pattern_index]["count"] = 5
        initialized_intelligence_engine.M["patterns"][pattern_index]["first_cycle"] = 1
        initialized_intelligence_engine.M["patterns"][pattern_index]["last_cycle"] = 5

        # Setup inference engine
        inference_engine = MagicMock()
        inference_engine.cycle_counter = 10

        # Update learning state
        initialized_intelligence_engine.update_learning_state(pattern_index, inference_engine)

        # Verify pattern updates
        assert initialized_intelligence_engine.M["patterns"][pattern_index]["count"] == 6
        assert initialized_intelligence_engine.M["patterns"][pattern_index]["first_cycle"] == 1
        assert initialized_intelligence_engine.M["patterns"][pattern_index]["last_cycle"] == 10

        # Verify key recording
        assert len(initialized_intelligence_engine.current_thread_keys) == 1
        assert initialized_intelligence_engine.current_thread_keys[0]["cycle"] == 10
        assert initialized_intelligence_engine.current_thread_keys[0]["pattern_index"] == pattern_index

    def test_update_learning_state_first_occurrence(self, initialized_intelligence_engine):
        """Test updating learning state for first pattern occurrence"""
        # Initialize pattern data
        pattern_index = 42
        initialized_intelligence_engine.M["patterns"][pattern_index]["count"] = 0
        initialized_intelligence_engine.M["patterns"][pattern_index]["first_cycle"] = None
        initialized_intelligence_engine.M["patterns"][pattern_index]["last_cycle"] = None

        # Setup inference engine
        inference_engine = MagicMock()
        inference_engine.cycle_counter = 10

        # Update learning state
        initialized_intelligence_engine.update_learning_state(pattern_index, inference_engine)

        # Verify pattern updates
        assert initialized_intelligence_engine.M["patterns"][pattern_index]["count"] == 1
        assert initialized_intelligence_engine.M["patterns"][pattern_index]["first_cycle"] == 10
        assert initialized_intelligence_engine.M["patterns"][pattern_index]["last_cycle"] == 10

    def test_encode_decode(self, initialized_intelligence_engine):
        """Test character encoding and decoding"""
        # Setup pattern with character label
        pattern_index = 42
        character_label = "test_character"
        initialized_intelligence_engine.M["patterns"][pattern_index]["character"] = [character_label]

        # Test encode
        assert initialized_intelligence_engine.encode(character_label) == pattern_index
        assert initialized_intelligence_engine.encode("nonexistent") is None

        # Test decode
        assert initialized_intelligence_engine.decode(pattern_index) == character_label
        assert initialized_intelligence_engine.decode(99) is None

    def test_load_thread_content(self, initialized_intelligence_engine):
        """Test loading a thread's decrypted content"""
        thread_uuid = "test-thread-uuid"

        with (
            patch("baby.intelligence.load_thread", return_value=b"encrypted_content") as mock_load_thread,
            patch(
                "baby.intelligence.load_thread_key", return_value=bytes([i % 256 for i in range(256)])
            ) as mock_load_thread_key,
        ):
            # Load thread content
            content = initialized_intelligence_engine.load_thread_content(thread_uuid)

            # Verify decryption
            assert content is not None

            # Check that the right functions were called
            mock_load_thread.assert_called_once_with(initialized_intelligence_engine.agent_uuid, thread_uuid)
            mock_load_thread_key.assert_called_once_with(
                initialized_intelligence_engine.agent_uuid, thread_uuid, initialized_intelligence_engine.agent_secret
            )

    def test_get_thread_relationships(self, initialized_intelligence_engine):
        """Test getting thread relationships"""
        thread_uuid = "test-thread-uuid"
        parent_uuid = "parent-thread-uuid"
        child_uuid = "child-thread-uuid"

        with (
            patch("baby.intelligence.parent", return_value=parent_uuid) as mock_parent,
            patch("baby.intelligence.children", return_value=[child_uuid]) as mock_children,
        ):
            # Get relationships
            relationships = initialized_intelligence_engine.get_thread_relationships(thread_uuid)

            # Verify result
            assert relationships["parent"] == parent_uuid
            assert relationships["children"] == [child_uuid]

            # Check function calls
            mock_parent.assert_called_once_with(initialized_intelligence_engine.agent_uuid, thread_uuid)
            mock_children.assert_called_once_with(initialized_intelligence_engine.agent_uuid, thread_uuid)

    def test_get_thread_chain(self, initialized_intelligence_engine):
        """Test getting a thread chain"""
        thread_uuid = "test-thread-uuid"
        parent_uuid = "parent-thread-uuid"
        grandparent_uuid = "grandparent-thread-uuid"
        child_uuid = "child-thread-uuid"

        # Mock parent and children functions
        with patch("baby.intelligence.parent") as mock_parent, patch("baby.intelligence.children") as mock_children:

            # Set up mock return values
            mock_parent.side_effect = lambda a, t: {
                "test-thread-uuid": parent_uuid,
                "parent-thread-uuid": grandparent_uuid,
                "grandparent-thread-uuid": None,
            }.get(t)

            mock_children.side_effect = lambda a, t: {
                "test-thread-uuid": [child_uuid],
                "parent-thread-uuid": [],
                "grandparent-thread-uuid": [parent_uuid],
            }.get(t, [])

            # Get thread chain
            chain = initialized_intelligence_engine.get_thread_chain(thread_uuid)

            # Verify chain order (root->leaf)
            assert grandparent_uuid in chain
            assert parent_uuid in chain
            assert thread_uuid in chain
            assert child_uuid in chain

            # Check parent/children calls
            # The number of calls may depend on the chain logic; check for at least 2 calls each
            assert mock_parent.call_count >= 2
            assert mock_children.call_count >= 2

    def test_get_thread_statistics(self, initialized_intelligence_engine):
        """Test getting thread statistics"""
        import uuid
        from baby.information import shard_path

        agent_uuid = initialized_intelligence_engine.agent_uuid
        private_dir = Path("memories/private/agents")
        agent_shard = shard_path(private_dir, agent_uuid)
        agent_dir = agent_shard / f"agent-{agent_uuid}"
        threads_dir = agent_dir / "threads"
        threads_dir.mkdir(parents=True, exist_ok=True)

        # Use valid UUIDs for thread IDs
        thread_ids: list[str] = [str(uuid.uuid4()) for _ in range(3)]
        registry = {"count": 3, "uuids": thread_ids}
        with open(threads_dir / "registry.json", "w") as f:
            json.dump(registry, f)

        # Mock thread metadata files in the correct shard path
        for thread_id, meta in zip(
            thread_ids,
            [
                {"thread_uuid": None, "parent_uuid": None, "child_uuids": [thread_ids[1]], "size_bytes": 1000},
                {"thread_uuid": None, "parent_uuid": thread_ids[0], "child_uuids": [thread_ids[2]], "size_bytes": 2000},
                {"thread_uuid": None, "parent_uuid": thread_ids[1], "child_uuids": [], "size_bytes": 3000},
            ],
        ):
            meta["thread_uuid"] = thread_id
            thread_shard = shard_path(threads_dir, thread_id)
            thread_shard.mkdir(parents=True, exist_ok=True)
            with open(thread_shard / f"thread-{thread_id}.json", "w") as f:
                json.dump(meta, f)

        # Get statistics
        stats = initialized_intelligence_engine.get_thread_statistics()

        # Verify statistics
        assert stats["total_threads"] == 3
        assert stats["total_size_bytes"] == 6000
        assert stats["relationship_stats"]["threads_with_parents"] == 2
        assert stats["relationship_stats"]["threads_with_children"] == 2
        assert stats["relationship_stats"]["isolated_threads"] == 0
        assert len(stats["thread_details"]) == 3

    def test_thread_capacity_exceeded(self, mock_env):
        """Test that new threads are created when capacity is exceeded"""
        from baby.intelligence import IntelligenceEngine
        import json

        # Initialize engine in the mock environment
        engine = initialize_intelligence_engine()

        # Set a small capacity for testing
        original_max_size = engine.memory_prefs["storage_config"]["max_thread_size_mb"]
        engine.memory_prefs["storage_config"]["max_thread_size_mb"] = 0.0001  # 100 bytes
        mem_prefs_path = "memories/memory_preferences.json"
        with open(mem_prefs_path, "r") as f:
            mem_prefs = json.load(f)
        original_disk_max_size = mem_prefs["storage_config"]["max_thread_size_mb"]
        mem_prefs["storage_config"]["max_thread_size_mb"] = 0.0001
        with open(mem_prefs_path, "w") as f:
            json.dump(mem_prefs, f, indent=2)

        try:
            # Process first input
            engine.process_input_stream(b"First message")
            engine._save_current_thread()
            first_thread_uuid = engine.thread_uuid

            # Re-instantiate the engine to pick up the new cap from disk
            engine = IntelligenceEngine(
                agent_uuid=engine.agent_uuid,
                agent_secret=engine.agent_secret,
                inference_engine=engine.inference_engine,
                information_engine=engine.information_engine,
                format_uuid=engine.format_uuid,
                formats=engine.M,
            )
            # Restore only the thread UUID and reload state
            engine.thread_uuid = first_thread_uuid
            assert engine.thread_uuid is not None, "thread_uuid must not be None"
            from baby.information import load_thread_key, load_thread

            engine.thread_file_key = load_thread_key(engine.agent_uuid, engine.thread_uuid, engine.agent_secret)
            engine.active_thread_content = bytearray(load_thread(engine.agent_uuid, engine.thread_uuid) or b"")
            engine.current_thread_size = len(engine.active_thread_content)

            # Debug: print thread state before second input
            max_thread_size_bytes = engine.memory_prefs["storage_config"]["max_thread_size_mb"] * 1024 * 1024
            print(f"[DEBUG] max_thread_size_bytes={max_thread_size_bytes}")
            print(f"[DEBUG] second input size={len(b'X' * 200)}")

            # Process second input that exceeds capacity
            engine.process_input_stream(b"X" * 200)  # 200 bytes
            second_thread_uuid = engine.thread_uuid

            # Should be different threads
            assert first_thread_uuid != second_thread_uuid
            assert second_thread_uuid is not None, "second_thread_uuid is None"
            relationships = engine.get_thread_relationships(second_thread_uuid)
            assert relationships["parent"] == first_thread_uuid

        finally:
            # Restore original capacity in memory and on disk
            engine.memory_prefs["storage_config"]["max_thread_size_mb"] = original_max_size
            mem_prefs["storage_config"]["max_thread_size_mb"] = original_disk_max_size
            with open(mem_prefs_path, "w") as f:
                json.dump(mem_prefs, f, indent=2)


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

    def test_initialize_intelligence_engine(self, mock_env):
        """Test intelligence engine initialization"""
        with (
            patch("uuid.uuid4", return_value=uuid.UUID("00000000-0000-0000-0000-000000000000")),
            patch.object(InferenceEngine, "_load_patterns", return_value=(np.zeros((256, 48)), ["test"] * 256)),
            patch.object(InferenceEngine, "_load_genome_mask", return_value=np.arange(256, dtype=np.uint8)),
            patch.object(InferenceEngine, "_initialize_epigenome"),
        ):
            # Create baby_preferences.json
            prefs_dir = Path("baby")
            prefs_dir.mkdir(exist_ok=True)
            with open(prefs_dir / "baby_preferences.json", "w") as f:
                json.dump({"agent_secret": "test_secret"}, f)

            # Initialize engine
            engine = initialize_intelligence_engine()

            # Verify initialization
            assert isinstance(engine, IntelligenceEngine)
            assert isinstance(engine.inference_engine, InferenceEngine)
            assert isinstance(engine.information_engine, InformationEngine)


# ------------------------------------------------------------------------------
# Main test runner
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """Run tests when file is executed directly"""
    pytest.main(["-v", __file__])
