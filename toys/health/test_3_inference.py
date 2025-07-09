"""
Tests for the inference engine and related tensor operations in the BabyLM system.
Includes pattern loading, genome mask handling, epigenome initialization, and pattern matching logic.
"""

import os
import json
import numpy as np
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch

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
)
from baby.intelligence import IntelligenceEngine, weighted_choice, initialize_intelligence_engine
from baby.governance import derive_canonical_patterns, apply_operation, gene_add

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

    # 4. Initialize the engine EXPLICITLY in private mode
    # This is the key change: we pass the agent_uuid and secret.
    engine = initialize_intelligence_engine(agent_uuid=agent_uuid, agent_secret=agent_secret)

    # 5. Assert that the engine is correctly configured for private mode
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

        # Call the method under test and UNPACK the tuple
        key_index, resonance = inference_engine.process_byte(input_byte)

        # Assert the tensor matches the expected result
        np.testing.assert_array_equal(inference_engine.T, expected_T)

        # Check that cycle counter incremented
        assert inference_engine.cycle_counter == 1

        # Check return values are valid
        assert isinstance(key_index, int)
        assert 0 <= key_index < 256
        assert isinstance(resonance, float)

    def test_find_closest_pattern_index(self, inference_engine):
        """Test finding the closest pattern"""
        # Set tensor to all ones
        inference_engine.T.fill(1.0)
        # Set all patterns to zeros except index 42, which matches the tensor
        inference_engine.F = np.zeros((256, 48))
        inference_engine.F[42] = np.ones(48)  # Make pattern 42 closest

        closest_index, min_distance = inference_engine.find_closest_pattern_index()
        assert closest_index == 42
        assert min_distance == 0.0

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
# Main test runner
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """Run tests when file is executed directly"""
    pytest.main(["-v", __file__])
