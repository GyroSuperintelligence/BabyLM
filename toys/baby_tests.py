"""
Comprehensive test suite for GyroSI Baby LM

This test suite covers all major components of the GyroSI Baby LM:
- Governance (S1): Core tensor operations
- Inference (S3): Pattern recognition
- Information (S2): Stream processing
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
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, List, Tuple, Any, Optional

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
from baby.information import InformationEngine
from baby.intelligence import IntelligenceEngine, weighted_choice, ensure_uuid_registry, initialize_intelligence_engine

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
    (memories_dir / "public").mkdir()
    (memories_dir / "public/formats").mkdir()
    (memories_dir / "public/masks").mkdir()

    # Monkeypatch os.makedirs to use the temp directory
    original_makedirs = os.makedirs

    def mock_makedirs(path, exist_ok=False):
        if path.startswith("memories/"):
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
def intelligence_engine(inference_engine, information_engine):
    """Create an intelligence engine for testing"""
    mock_format_metadata = {
        "format_uuid": "test-format-uuid",
        "format_name": "test_format",
        "cgm_version": "1.0.0",
        "format_version": "1.0.0",
        "stability": "experimental",
        "compatibility": {
            "min_cgm_version": "0.9.0",
            "max_cgm_version": "1.0.0",
            "depends_on": [],
            "conflicts_with": [],
        },
        "metadata": {
            "author": "test_author",
            "description": "Test format",
            "tags": ["test"],
            "created_at": "2023-01-01T00:00:00",
            "last_updated": "2023-01-01T00:00:00",
            "usage_count": 0,
            "validation_status": "unverified",
        },
        "cgm_policies": {
            "governance": {"operation": "L0", "bits": [0, 7], "policy": "traceability"},
            "information": {"operation": "LI", "bits": [1, 6], "policy": "variety"},
            "inference": {"operation": "FG", "bits": [2, 5], "policy": "accountability"},
            "intelligence": {"operation": "BG", "bits": [3, 4], "policy": "integrity"},
        },
        "patterns": [
            {
                "index": i,
                "semantic": None,
                "count": 0,
                "first_cycle": None,
                "last_cycle": None,
                "resonance_class": "test",
                "confidence": 0.0,
            }
            for i in range(256)
        ],
    }

    mock_memory_prefs = {
        "uuid_registry": {"agent_uuid": "test-agent-uuid", "format_uuid": "test-format-uuid", "thread_uuids": []},
        "storage_config": {"max_thread_size_mb": 64, "shard_prefix_length": 2, "encryption_algorithm": "AES-256-GCM"},
        "format_config": {
            "default_cgm_version": "1.0.0",
            "resonance_threshold": float(np.pi / 2),
            "max_semantic_label_length": 128,
        },
    }

    with (
        patch.object(IntelligenceEngine, "_load_or_init_formats", return_value=mock_format_metadata),
        patch.object(IntelligenceEngine, "_load_memory_preferences", return_value=mock_memory_prefs),
        patch.object(IntelligenceEngine, "_validate_format_compatibility"),
    ):
        engine = IntelligenceEngine(
            agent_uuid="test-agent-uuid",
            agent_secret="test-agent-secret",
            format_uuid="test-format-uuid",
            inference_engine=inference_engine,
            information_engine=information_engine,
        )
        return engine


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
    (memories_dir / "private").mkdir(parents=True, exist_ok=True)
    baby_dir.mkdir(exist_ok=True)

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
        "default_resonance_threshold": float(np.pi / 2),
    }
    with open(baby_prefs_path, "w") as f:
        json.dump(baby_prefs, f, indent=2)

    # 2. Create the UUID registry and memory_preferences.json
    uuid_registry = ensure_uuid_registry()
    agent_uuid = uuid_registry["agent_uuid"]
    format_uuid = uuid_registry["format_uuid"]

    # 3. Create the Epigenome and Genome masks
    # These are required for the InferenceEngine to initialize
    patterns_array, _ = derive_canonical_patterns()
    patterns_array.tofile("memories/public/masks/epigenome.dat")

    genome_mask = np.arange(256, dtype=np.uint8)  # Identity mapping for predictability
    genome_mask.tofile("memories/public/masks/genome.dat")

    # 4. Initialize the engine. This will now find all necessary files.
    # The IntelligenceEngine's __init__ will automatically create a default format file.
    engine = initialize_intelligence_engine()

    return engine


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
        patterns, resonance_classes = derive_canonical_patterns()

        # Check dimensions
        assert patterns.shape == (256, 48)  # 256 patterns, each 48 elements
        assert len(resonance_classes) == 256

        # Check pattern content
        assert patterns.dtype == np.float32

        # Check that all resonance classes are valid
        valid_classes = ["identity", "inverse", "forward", "backward"]
        for cls in resonance_classes:
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
        resonance_class = classify_pattern_resonance(mask)
        assert resonance_class == expected_class


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
            assert len(engine.resonance_classes) == 256
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
        assert len(engine.resonance_classes) == 256

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
            assert len(engine.resonance_classes) == 256

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


# ------------------------------------------------------------------------------
# Information (S2) Tests
# ------------------------------------------------------------------------------


class TestInformation:
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

    def test_intelligence_engine_init(self, intelligence_engine):
        """Test IntelligenceEngine initialization"""
        assert intelligence_engine.agent_uuid == "test-agent-uuid"
        assert intelligence_engine.format_uuid == "test-format-uuid"
        assert intelligence_engine.thread_uuid is None
        assert intelligence_engine.thread_file_key is None
        assert isinstance(intelligence_engine.M, dict)
        assert isinstance(intelligence_engine.memory_prefs, dict)

    def test_load_or_init_formats_existing(self, mock_memories_dir, intelligence_engine):
        """Test loading existing format metadata"""
        # Create mock format file
        format_path = mock_memories_dir / f"public/formats/formats-{intelligence_engine.format_uuid}.json"
        format_path.parent.mkdir(parents=True, exist_ok=True)

        mock_format = {"format_uuid": intelligence_engine.format_uuid, "format_name": "test_format", "patterns": []}

        with open(format_path, "w") as f:
            json.dump(mock_format, f)

        # Test loading
        result = intelligence_engine._load_or_init_formats()

        # Verify loaded format
        assert result["format_uuid"] == intelligence_engine.format_uuid
        assert result["format_name"] == "test_format"

    def test_load_or_init_formats_new(self, mock_memories_dir, intelligence_engine):
        """Test initializing new format metadata"""
        # Ensure format file doesn't exist
        format_path = mock_memories_dir / f"public/formats/formats-{intelligence_engine.format_uuid}.json"
        if format_path.exists():
            format_path.unlink()

        # Test initialization
        with patch.object(
            intelligence_engine,
            "_initialize_format_metadata",
            return_value={"format_uuid": intelligence_engine.format_uuid, "test": True},
        ):
            result = intelligence_engine._load_or_init_formats()

            # Verify initialized format
            assert result["format_uuid"] == intelligence_engine.format_uuid
            assert result["test"] is True

    def test_initialize_format_metadata(self, intelligence_engine):
        """Test format metadata initialization"""
        # Initialize new format metadata
        result = intelligence_engine._initialize_format_metadata()

        # Verify structure
        assert result["format_uuid"] == intelligence_engine.format_uuid
        assert "patterns" in result
        assert len(result["patterns"]) == 256
        assert "cgm_policies" in result

        # Check all required policies
        required_policies = ["governance", "information", "inference", "intelligence"]
        for policy in required_policies:
            assert policy in result["cgm_policies"]

    def test_start_new_thread(self, intelligence_engine):
        """Test starting a new thread"""
        with (
            patch.object(intelligence_engine, "_update_uuid_registry"),
            patch.object(intelligence_engine, "_derive_file_key", return_value=b"test_key"),
        ):

            # Start a new thread
            thread_uuid = intelligence_engine.start_new_thread()

            # Verify thread initialization
            assert thread_uuid == intelligence_engine.thread_uuid
            assert intelligence_engine.thread_file_key == b"test_key"
            assert intelligence_engine.current_thread_keys == []

    def test_process_input_stream(self, intelligence_engine):
        """Test processing an input stream"""
        test_input = b"Test input stream"

        with (
            patch.object(intelligence_engine, "start_new_thread", return_value="test-thread-uuid"),
            patch.object(
                intelligence_engine.information_engine, "process_stream", return_value=(b"ciphertext", b"keystream")
            ),
            patch.object(intelligence_engine, "end_current_thread"),
        ):

            # Process input stream
            plaintext, ciphertext = intelligence_engine.process_input_stream(test_input)

            # Verify results
            assert plaintext == test_input
            assert ciphertext == b"ciphertext"

            # Verify method calls
            intelligence_engine.start_new_thread.assert_called_once()
            intelligence_engine.information_engine.process_stream.assert_called_once()
            intelligence_engine.end_current_thread.assert_called_once_with(plaintext_to_save=test_input)

    def test_end_current_thread(self, intelligence_engine, mock_memories_dir):
        """Test ending a current thread"""
        # Setup thread state
        intelligence_engine.thread_uuid = "test-thread-uuid"
        intelligence_engine.thread_file_key = bytes([i % 256 for i in range(256)])  # Test key
        intelligence_engine.current_thread_keys = [{"cycle": 1, "pattern_index": 42}]

        # Test data to save
        test_data = b"Thread content to save"

        with (
            patch.object(intelligence_engine, "_derive_agent_key", return_value=b"agent_key"),
            patch.object(intelligence_engine, "_encrypt_data", return_value=b"encrypted_keys"),
            patch.object(intelligence_engine, "_decrypt_data", return_value=b"{}"),
        ):

            # End the thread
            intelligence_engine.end_current_thread(plaintext_to_save=test_data)

            # Check that thread file was created
            thread_path = (
                mock_memories_dir / f"private/{intelligence_engine.agent_uuid}/threads/"
                f"{intelligence_engine.thread_uuid[:2]}/thread-{intelligence_engine.thread_uuid}.enc"
            )
            assert thread_path.parent.exists()

    def test_generate_and_save_response(self, intelligence_engine):
        """Test generating and saving a response"""
        with (
            patch.object(intelligence_engine, "start_new_thread", return_value="test-thread-uuid"),
            patch.object(intelligence_engine, "_generate_response_bytes", return_value=b"Generated response"),
            patch.object(intelligence_engine, "end_current_thread"),
        ):

            # Generate response
            response = intelligence_engine.generate_and_save_response(length=20)

            # Verify results
            assert response == b"Generated response"

            # Verify method calls
            intelligence_engine.start_new_thread.assert_called_once()
            intelligence_engine._generate_response_bytes.assert_called_once_with(20)
            intelligence_engine.end_current_thread.assert_called_once_with(plaintext_to_save=b"Generated response")

    def test_generate_response_bytes(self, intelligence_engine):
        """Test generating response bytes"""
        with (
            patch.object(intelligence_engine, "_generate_response_byte", side_effect=[(97, 0), (98, 1), (99, 2)]),
            patch.object(intelligence_engine.inference_engine, "process_byte"),
            patch.object(intelligence_engine, "update_learning_state"),
        ):

            # Generate 3 bytes
            response = intelligence_engine._generate_response_bytes(3)

            # Verify the output
            assert response == b"abc"

            # Check method calls
            assert intelligence_engine._generate_response_byte.call_count == 3
            assert intelligence_engine.inference_engine.process_byte.call_count == 3
            assert intelligence_engine.update_learning_state.call_count == 3

    def test_update_learning_state(self, intelligence_engine):
        """Test updating learning state"""
        # Initialize pattern data
        pattern_index = 42
        intelligence_engine.M["patterns"][pattern_index]["count"] = 5
        intelligence_engine.M["patterns"][pattern_index]["first_cycle"] = 1
        intelligence_engine.M["patterns"][pattern_index]["last_cycle"] = 5

        # Setup inference engine
        inference_engine = MagicMock()
        inference_engine.cycle_counter = 10

        # Update learning state
        intelligence_engine.update_learning_state(pattern_index, inference_engine)

        # Verify pattern updates
        assert intelligence_engine.M["patterns"][pattern_index]["count"] == 6
        assert intelligence_engine.M["patterns"][pattern_index]["first_cycle"] == 1
        assert intelligence_engine.M["patterns"][pattern_index]["last_cycle"] == 10

        # Verify key recording
        assert len(intelligence_engine.current_thread_keys) == 1
        assert intelligence_engine.current_thread_keys[0]["cycle"] == 10
        assert intelligence_engine.current_thread_keys[0]["pattern_index"] == pattern_index

    def test_update_learning_state_first_occurrence(self, intelligence_engine):
        """Test updating learning state for first pattern occurrence"""
        # Initialize pattern data
        pattern_index = 42
        intelligence_engine.M["patterns"][pattern_index]["count"] = 0
        intelligence_engine.M["patterns"][pattern_index]["first_cycle"] = None
        intelligence_engine.M["patterns"][pattern_index]["last_cycle"] = None

        # Setup inference engine
        inference_engine = MagicMock()
        inference_engine.cycle_counter = 10

        # Update learning state
        intelligence_engine.update_learning_state(pattern_index, inference_engine)

        # Verify pattern updates
        assert intelligence_engine.M["patterns"][pattern_index]["count"] == 1
        assert intelligence_engine.M["patterns"][pattern_index]["first_cycle"] == 10
        assert intelligence_engine.M["patterns"][pattern_index]["last_cycle"] == 10

    def test_encode_decode(self, intelligence_engine):
        """Test semantic encoding and decoding"""
        # Setup pattern with semantic label
        pattern_index = 42
        semantic_label = "test_semantic"
        intelligence_engine.M["patterns"][pattern_index]["semantic"] = semantic_label

        # Test encode
        assert intelligence_engine.encode(semantic_label) == pattern_index
        assert intelligence_engine.encode("nonexistent") is None

        # Test decode
        assert intelligence_engine.decode(pattern_index) == semantic_label
        assert intelligence_engine.decode(99) is None

    def test_load_thread(self, intelligence_engine, mock_memories_dir):
        """Test loading a thread"""
        # Setup thread data
        thread_uuid = "test-thread-uuid"
        intelligence_engine.memory_prefs["uuid_registry"]["thread_uuids"].append(thread_uuid)

        # Create mock thread file
        thread_dir = mock_memories_dir / f"private/{intelligence_engine.agent_uuid}/threads/{thread_uuid[:2]}"
        thread_dir.mkdir(parents=True, exist_ok=True)

        thread_path = thread_dir / f"thread-{thread_uuid}.enc"
        with open(thread_path, "wb") as f:
            f.write(b"encrypted_content")

        # Create mock keys file
        keys_dir = mock_memories_dir / f"private/{intelligence_engine.agent_uuid}/keys"
        keys_dir.mkdir(parents=True, exist_ok=True)

        keys_path = keys_dir / f"keys-{intelligence_engine.agent_uuid}.json.enc"
        with open(keys_path, "wb") as f:
            f.write(b"encrypted_keys")

        # Mock key derivation and decryption
        mock_thread_keys = json.dumps({thread_uuid: [{"cycle": 1, "pattern_index": 0}]})

        with (
            patch.object(intelligence_engine, "_derive_agent_key", return_value=b"agent_key"),
            patch.object(
                intelligence_engine,
                "_decrypt_data",
                side_effect=[
                    mock_thread_keys.encode(),  # For keys file
                ],
            ),
            patch.object(intelligence_engine, "_derive_file_key", return_value=bytes([i ^ 0xFF for i in range(256)])),
        ):

            # Load thread
            content = intelligence_engine.load_thread(thread_uuid)

            # Verify thread was loaded and decrypted
            assert content is not None

    def test_load_thread_nonexistent(self, intelligence_engine):
        """Test loading a nonexistent thread"""
        # Try to load nonexistent thread
        content = intelligence_engine.load_thread("nonexistent-uuid")

        # Should return None
        assert content is None

    def test_discover_formats_from_agent(self, intelligence_engine, mock_memories_dir):
        """Test discovering formats from another agent"""
        # Create mock memory preferences
        memory_path = mock_memories_dir / "memory_preferences.json"
        with open(memory_path, "w") as f:
            json.dump(
                {"uuid_registry": {"agent_uuid": "test-agent-2", "format_uuid": "test-format-2", "thread_uuids": []}}, f
            )

        # Create test format files
        formats_dir = mock_memories_dir / "public/formats"
        formats_dir.mkdir(parents=True, exist_ok=True)

        # Create a format authored by the target agent
        format_data = {
            "format_uuid": "test-format-2",
            "format_name": "Test Format 2",
            "metadata": {"author": "agent_test-agent-2", "description": "Format by test agent 2"},
        }

        with open(formats_dir / "formats-test-format-2.json", "w") as f:
            json.dump(format_data, f)

        # Discover formats
        discovered = intelligence_engine.discover_formats_from_agent("test-agent-2")

        # Should find the format
        assert "test-format-2" in discovered

    def test_compose_formats(self, intelligence_engine, mock_memories_dir):
        """Test composing multiple formats"""
        # Create test format files
        formats_dir = mock_memories_dir / "public/formats"
        formats_dir.mkdir(parents=True, exist_ok=True)

        # Create primary format
        primary_format = {
            "format_uuid": "primary-format",
            "format_name": "Primary Format",
            "compatibility": {"depends_on": [], "conflicts_with": []},
            "metadata": {"tags": ["primary"], "description": "Primary format for testing", "author": "test_author"},
            "patterns": [
                {"index": i, "semantic": None if i != 1 else "test", "count": 5, "confidence": 0.5} for i in range(256)
            ],
        }

        with open(formats_dir / "formats-primary-format.json", "w") as f:
            json.dump(primary_format, f)

        # Create secondary format
        secondary_format = {
            "format_uuid": "secondary-format",
            "format_name": "Secondary Format",
            "metadata": {"tags": ["secondary"], "description": "Secondary format for testing"},
            "patterns": [
                {"index": i, "semantic": f"secondary_{i}" if i % 10 == 0 else None, "count": 10, "confidence": 0.8}
                for i in range(256)
            ],
        }

        with open(formats_dir / "formats-secondary-format.json", "w") as f:
            json.dump(secondary_format, f)

        # Mock UUID generation for composed format
        with patch("uuid.uuid4", return_value=uuid.UUID("00000000-0000-0000-0000-000000000000")):
            # Compose formats
            composed_uuid = intelligence_engine.compose_formats("primary-format", ["secondary-format"])

            # Verify composed format was created
            assert composed_uuid == "00000000-0000-0000-0000-000000000000"

            # Check that composed format file exists
            composed_path = formats_dir / f"formats-{composed_uuid}.json"
            assert composed_path.exists()

            # Load and verify composed format
            with open(composed_path, "r") as f:
                composed_data = json.load(f)

            # Check basic properties
            assert composed_data["format_uuid"] == composed_uuid
            assert composed_data["format_name"].startswith("composed_")
            assert composed_data["stability"] == "experimental"

            # Check dependencies
            assert composed_data["compatibility"]["depends_on"] == ["primary-format", "secondary-format"]

            # Check merged tags
            assert "primary" in composed_data["metadata"]["tags"]
            assert "secondary" in composed_data["metadata"]["tags"]

            # Check pattern merging (primary semantic retained, secondary added where primary was None)
            assert composed_data["patterns"][1]["semantic"] == "test"  # From primary
            assert composed_data["patterns"][10]["semantic"] == "secondary_10"  # From secondary


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

        # Verify that thread files were created and registered
        thread_uuids = intelligence_engine.memory_prefs["uuid_registry"]["thread_uuids"]
        assert len(thread_uuids) == 2  # One for the input, one for the response

        # Verify the actual thread files exist on disk
        for thread_uuid in thread_uuids:
            shard = thread_uuid[:2]
            thread_path = Path(
                f"memories/private/{intelligence_engine.agent_uuid}/threads/{shard}/thread-{thread_uuid}.enc"
            )
            assert thread_path.exists()
            assert thread_path.stat().st_size > 0

    def test_generate_response_byte(self, initialized_intelligence_engine):
        """Test generating a single response byte with specific state."""
        intelligence_engine = initialized_intelligence_engine

        # We don't need to mock as much, we can test the real logic.
        # Let's set the tensor state to be identical to a known pattern.
        # This guarantees that pattern will be the closest match.
        target_pattern_index = 2  # Use pattern 2 which is unique (not identical to pattern 0)
        target_output_byte = 65  # ASCII 'A'

        # Set the current tensor to exactly match the canonical pattern for index 2
        intelligence_engine.inference_engine.T = intelligence_engine.inference_engine.F[target_pattern_index].reshape(
            4, 2, 3, 2
        )

        # Ensure the genome mask maps this pattern to our expected byte
        intelligence_engine.inference_engine.G[target_pattern_index] = target_output_byte

        # Generate the byte using the real, canonical method
        byte, pattern_index = intelligence_engine._generate_response_byte()

        # Verify output
        assert pattern_index == target_pattern_index
        assert byte == target_output_byte

    def test_load_thread(self, initialized_intelligence_engine):
        """Test saving and then loading a thread to verify the full cycle."""
        intelligence_engine = initialized_intelligence_engine

        # 1. First, create a thread by processing some input. This is the most reliable
        #    way to get correctly formatted and encrypted thread/key files.
        original_content = b"This content will be saved and then reloaded."
        intelligence_engine.process_input_stream(original_content)

        # 2. Get the UUID of the thread we just created.
        thread_uuid = intelligence_engine.memory_prefs["uuid_registry"]["thread_uuids"][0]

        # 3. Simulate an application restart by creating a NEW engine instance.
        #    This new instance will have a fresh, zeroed-out state.
        new_engine = initialize_intelligence_engine()

        # 4. Use the new engine to load the thread created by the first one.
        loaded_content = new_engine.load_thread(thread_uuid)

        # 5. Verify the thread was loaded and decrypted correctly.
        assert loaded_content is not None
        assert loaded_content == original_content

    def test_load_thread_nonexistent(self, initialized_intelligence_engine):
        """Test loading a nonexistent thread returns None."""
        intelligence_engine = initialized_intelligence_engine

        # Try to load a random, non-existent thread UUID
        content = intelligence_engine.load_thread(str(uuid.uuid4()))

        assert content is None

    def test_select_stable_format(self, initialized_intelligence_engine):
        """Test selecting a format based on domain and stability."""
        intelligence_engine = initialized_intelligence_engine
        formats_dir = Path("memories/public/formats")

        # Create a few test format files with different properties
        formats_data = [
            {
                "format_uuid": "english-stable-1",
                "format_name": "Eng Stable",
                "stability": "stable",
                "metadata": {"tags": ["english"], "description": "", "usage_count": 10},
            },
            {
                "format_uuid": "english-stable-2",
                "format_name": "Eng Stable More Used",
                "stability": "stable",
                "metadata": {"tags": ["english"], "description": "", "usage_count": 20},  # Higher usage
            },
            {
                "format_uuid": "code-stable-1",
                "format_name": "Code Stable",
                "stability": "stable",
                "metadata": {"tags": ["code"], "description": "", "usage_count": 5},
            },
            {
                "format_uuid": "english-beta-1",
                "format_name": "Eng Beta",
                "stability": "beta",
                "metadata": {"tags": ["english"], "description": "", "usage_count": 100},
            },
        ]

        # Add required boilerplate to each format and save it
        for fmt in formats_data:
            fmt["cgm_version"] = "1.0.0"
            fmt["format_version"] = "1.0.0"
            fmt["compatibility"] = {}
            fmt["cgm_policies"] = {}
            fmt["metadata"]["author"] = "test"
            with open(formats_dir / f"formats-{fmt['format_uuid']}.json", "w") as f:
                json.dump(fmt, f)

        # Should select the stable English format with the highest usage
        selected = intelligence_engine.select_stable_format(domain="english", stability="stable")
        assert selected == "english-stable-2"

        # Should select the only stable code format
        selected = intelligence_engine.select_stable_format(domain="code", stability="stable")
        assert selected == "code-stable-1"

        # Should select the only beta English format
        selected = intelligence_engine.select_stable_format(domain="english", stability="beta")
        assert selected == "english-beta-1"

        # Should return None for a domain that doesn't exist
        selected = intelligence_engine.select_stable_format(domain="nonexistent", stability="stable")
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

    def test_ensure_uuid_registry(self, mock_memories_dir):
        """Test UUID registry initialization"""
        # Ensure registry doesn't exist
        registry_path = mock_memories_dir / "memory_preferences.json"
        if registry_path.exists():
            registry_path.unlink()

        # Create registry
        with patch("uuid.uuid4", return_value=uuid.UUID("00000000-0000-0000-0000-000000000000")):
            registry = ensure_uuid_registry()

        # Verify registry was created
        assert "agent_uuid" in registry
        assert "format_uuid" in registry
        assert "thread_uuids" in registry

        # Verify registry was saved
        assert registry_path.exists()

        # Load saved registry
        with open(registry_path, "r") as f:
            saved_data = json.load(f)

        assert "uuid_registry" in saved_data
        assert saved_data["uuid_registry"]["agent_uuid"] == "00000000-0000-0000-0000-000000000000"

    def test_initialize_intelligence_engine(self, mock_memories_dir):
        """Test intelligence engine initialization"""
        with (
            patch("uuid.uuid4", return_value=uuid.UUID("00000000-0000-0000-0000-000000000000")),
            patch.object(IntelligenceEngine, "_validate_format_compatibility"),
            patch.object(InferenceEngine, "_load_patterns", return_value=(np.zeros((256, 48)), ["test"] * 256)),
            patch.object(InferenceEngine, "_load_genome_mask", return_value=np.arange(256, dtype=np.uint8)),
            patch.object(InferenceEngine, "_initialize_epigenome"),
        ):

            # Initialize engine
            engine = initialize_intelligence_engine()

            # Verify initialization
            assert isinstance(engine, IntelligenceEngine)
            assert isinstance(engine.inference_engine, InferenceEngine)
            assert isinstance(engine.information_engine, InformationEngine)

            # Verify agent UUID and format UUID were set
            assert engine.agent_uuid is not None
            assert engine.format_uuid is not None


# ------------------------------------------------------------------------------
# Main test runner
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """Run tests when file is executed directly"""
    pytest.main(["-v", __file__])
