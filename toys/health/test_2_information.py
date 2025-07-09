"""
Tests for information storage, thread management, and registry operations in the BabyLM system.
Covers memory preferences, sharding, atomic writes, registry updates, and thread relationships.
"""

import os
import json
import numpy as np
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

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
from baby.governance import derive_canonical_patterns

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
# Main test runner
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    """Run tests when file is executed directly"""
    pytest.main(["-v", __file__])
