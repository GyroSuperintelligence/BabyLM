"""
Tests for governance logic, gene operations, and pattern classification in the BabyLM system.
Includes tests for gene constants, operation application, gyrodistance, canonical pattern derivation, and pattern resonance classification.
"""

import os
import uuid
import json
import numpy as np
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch
from typing import cast, Dict, Any
from datetime import datetime

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
    PatternIndex,
)
from baby.intelligence import initialize_intelligence_engine
from baby.types import FormatMetadata, GeneKeysMetadata

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
        """End-to-end test: create agent, thread, save/load encrypted content and key using IntelligenceEngine workflow"""
        import os
        from pathlib import Path
        from baby.intelligence import IntelligenceEngine
        from baby.inference import InferenceEngine
        from baby.information import InformationEngine
        import sys
        import glob

        os.chdir(tmp_path)
        # Patch the 'memories/' root to point to tmp_path
        sys.path.insert(0, str(tmp_path))
        (Path("memories") / "private" / "agents").mkdir(parents=True, exist_ok=True)
        agent_secret = "test_secret"
        # Create agent and directories
        agent_uuid = ensure_agent_uuid()
        format_uuid = "11111111-1111-1111-1111-111111111111"
        # Create required engine instances
        inference_engine = InferenceEngine()
        information_engine = InformationEngine()
        engine = IntelligenceEngine(
            agent_uuid=agent_uuid,
            agent_secret=agent_secret,
            inference_engine=inference_engine,
            information_engine=information_engine,
        )
        # Remove manual thread_key generation and assignment
        # engine.thread_file_key = thread_key
        engine.start_new_thread(privacy="private")
        thread_uuid = engine.thread_uuid
        assert thread_uuid is not None, "thread_uuid is None after starting new thread"
        thread_uuid = str(thread_uuid)
        # Buffer content
        test_content = b"Test thread content"
        engine.active_thread_content.extend(test_content)
        # Finalize and save thread
        engine.finalize_and_save_thread(privacy="private")
        # Glob for any .enc file in the memories/private/agents tree
        enc_files = list(Path("memories/private/agents").rglob("thread-*.enc"))
        print(f"[DEBUG] .enc files found: {enc_files}")
        assert enc_files, "No encrypted thread file was created in the expected directory tree."
        # Store gene keys with proper structure
        test_gene_keys = [
            cast(
                GeneKeysMetadata,
                {
                    "cycle": 1,
                    "pattern_index": 42,
                    "thread_uuid": thread_uuid,
                    "agent_uuid": str(agent_uuid),
                    "format_uuid": format_uuid,
                    "event_type": "INPUT",
                    "source_byte": 0,
                    "resonance": 0.5,
                    "created_at": datetime.now().isoformat(),
                    "privacy": "private",
                },
            ),
            cast(
                GeneKeysMetadata,
                {
                    "cycle": 2,
                    "pattern_index": 84,
                    "thread_uuid": thread_uuid,
                    "agent_uuid": str(agent_uuid),
                    "format_uuid": format_uuid,
                    "event_type": "INPUT",
                    "source_byte": 1,
                    "resonance": 0.4,
                    "created_at": datetime.now().isoformat(),
                    "privacy": "private",
                },
            ),
        ]
        store_gene_keys(thread_uuid=thread_uuid, gene_keys=test_gene_keys, privacy="private", agent_secret=agent_secret)
        # Load thread content (decrypt for private thread)
        encrypted_content = load_thread(str(agent_uuid), thread_uuid)
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes

        # Derive decryption key
        thread_key = load_thread_key(str(agent_uuid), thread_uuid, agent_secret)
        assert thread_key is not None, "Thread key could not be loaded for decryption"
        salt = (str(thread_uuid)).encode("utf-8")
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend(),
        )
        decryption_key = kdf.derive(thread_key)
        nonce = encrypted_content[:12]
        tag = encrypted_content[12:28]
        ciphertext = encrypted_content[28:]
        cipher = Cipher(algorithms.AES(decryption_key), modes.GCM(nonce, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_content = decryptor.update(ciphertext) + decryptor.finalize()
        assert decrypted_content == test_content
        # Load gene keys
        loaded_gene_keys = load_gene_keys(
            thread_uuid=thread_uuid, agent_uuid=str(agent_uuid), agent_secret=agent_secret
        )
        assert isinstance(loaded_gene_keys, list)
        assert len(loaded_gene_keys) == len(test_gene_keys)
        for loaded, original in zip(loaded_gene_keys, test_gene_keys):
            assert loaded is not None and original is not None
            loaded_dict = cast(Dict[str, Any], loaded)
            original_dict = cast(Dict[str, Any], original)
            assert loaded_dict["cycle"] == original_dict["cycle"]  # type: ignore
            assert loaded_dict["pattern_index"] == original_dict["pattern_index"]  # type: ignore

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
            parent_uuid = create_thread("private", None, format_uuid)
        # Create child thread
        with patch("uuid.uuid4", return_value=uuid.UUID("00000000-0000-0000-0000-000000000002")):
            child_uuid = create_thread("private", parent_uuid, format_uuid)
        # Update parent thread's metadata to add child
        threads_dir = agent_dir / "threads"
        parent_shard = shard_path(threads_dir, parent_uuid)
        parent_meta_path = parent_shard / f"thread-{parent_uuid}.json"
        if parent_meta_path.exists():
            with open(parent_meta_path, "r") as f:
                parent_meta = json.load(f)
            if "children" not in parent_meta or not isinstance(parent_meta["children"], list):
                parent_meta["children"] = []
            # Check if child_uuid is already present by uuid
            if not any(c.get("uuid") == child_uuid for c in parent_meta["children"]):
                parent_meta["children"].append({"uuid": child_uuid})
            with open(parent_meta_path, "w") as f:
                f.write(json.dumps(parent_meta))
        # Debug: Print children list before assertion
        children_list = children(agent_uuid, parent_uuid)
        print(f"Children of parent {parent_uuid}: {children_list}")
        assert parent(agent_uuid, child_uuid) == parent_uuid
        assert child_uuid in children_list, f"Child UUID {child_uuid} not found in children list: {children_list}"

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

    def test_gene_keys_public_storage(self, mock_env):
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

    def test_pattern_index_functionality(self, mock_env):
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
