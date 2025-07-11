# toys/health/conftest.py

import os
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from baby.governance import derive_canonical_patterns
from baby.information import InformationEngine, assign_agent_uuid, shard_path, get_memory_preferences
from baby.intelligence import initialize_intelligence_engine


# --- Golden Environment Fixture ---
@pytest.fixture
def mock_env(tmp_path):
    """
    Creates a temporary, self-contained 'toys/health/memories' and 'baby' environment,
    changes the current working directory into it, and cleans up afterward.
    This is the foundational fixture for all tests needing file I/O.
    """
    # Create the base directories
    test_memories_dir = tmp_path / "toys/health/memories"
    baby_dir = tmp_path / "baby"
    (test_memories_dir / "public" / "masks").mkdir(parents=True, exist_ok=True)
    (test_memories_dir / "public" / "formats").mkdir(parents=True, exist_ok=True)
    (test_memories_dir / "public" / "threads").mkdir(parents=True, exist_ok=True)
    (test_memories_dir / "public" / "keys").mkdir(parents=True, exist_ok=True)
    (test_memories_dir / "private" / "agents").mkdir(parents=True, exist_ok=True)
    baby_dir.mkdir(exist_ok=True)

    # Create default memory_preferences.json
    mem_prefs = {
        "sharding": {"width": 2, "max_files": 30000, "second_level": True},
        "storage_config": {"max_thread_size_mb": 64, "encryption_algorithm": "AES-256-GCM"},
        "format_config": {"default_cgm_version": "1.0.0", "max_character_label_length": 128},
    }
    with open(test_memories_dir / "memory_preferences.json", "w") as f:
        json.dump(mem_prefs, f, indent=2)

    # Create default mask files needed by InferenceEngine
    patterns_array, _ = derive_canonical_patterns()
    patterns_array.tofile(test_memories_dir / "public" / "masks" / "epigenome.dat")
    genome_mask = np.arange(256, dtype=np.uint8)
    genome_mask.tofile(test_memories_dir / "public" / "masks" / "genome.dat")

    # Change the CWD to the temp path so 'toys/health/memories/...' paths work
    original_cwd = Path.cwd()
    os.chdir(tmp_path)

    yield tmp_path  # The test runs here

    # Teardown: Change back to the original directory
    os.chdir(original_cwd)


# --- Engine Fixtures (built on top of mock_env) ---


@pytest.fixture
def public_intelligence_engine(mock_env):
    """Initializes an IntelligenceEngine in public/curation mode."""
    base_memories_dir = str(mock_env / "toys/health/memories")
    engine = initialize_intelligence_engine(agent_uuid=None, agent_secret=None, base_memories_dir=base_memories_dir)
    assert engine.agent_uuid is None
    assert engine.agent_secret is None
    return engine


@pytest.fixture
def initialized_intelligence_engine(mock_env):
    """Initializes an IntelligenceEngine in a predictable private-agent mode."""
    agent_uuid = "11111111-1111-1111-1111-111111111111"
    agent_secret = "test-secret-for-fixture"
    base_memories_dir = str(mock_env / "toys/health/memories")
    # Create baby_preferences.json with the secret to simulate a user's setup
    with open("baby/baby_preferences.json", "w") as f:
        json.dump({"agent_secret": agent_secret}, f)
    # Ensure the agent's directory structure exists *before* initializing the engine
    prefs = get_memory_preferences(base_memories_dir)
    assign_agent_uuid(agent_uuid, base_memories_dir=base_memories_dir, prefs=prefs)
    # Initialize the engine explicitly for this agent
    engine = initialize_intelligence_engine(
        agent_uuid=agent_uuid,
        agent_secret=agent_secret,
        base_memories_dir=base_memories_dir)
    # Assertions to ensure the fixture is set up correctly
    assert engine.agent_uuid == agent_uuid
    assert engine.agent_secret == agent_secret
    private_dir = Path(base_memories_dir) / "private/agents"
    prefs = get_memory_preferences(base_memories_dir)
    agent_shard = shard_path(private_dir, engine.agent_uuid, prefs)
    agent_dir = agent_shard / f"agent-{engine.agent_uuid}"
    assert agent_dir.exists()
    return engine


# --- Simple Mocking Fixtures ---


@pytest.fixture
def inference_engine():
    """Mocks an InferenceEngine to isolate its behavior for S2/S4 tests."""
    # This is useful when you want to test InformationEngine without running real tensor math
    with patch("baby.inference.InferenceEngine") as MockInferenceEngine:
        mock_engine = MockInferenceEngine.return_value
        mock_engine.process_byte.return_value = (42, 0.1)  # Default mock return
        yield mock_engine


@pytest.fixture
def information_engine(mock_env):
    """Provides a standard InformationEngine instance."""
    base_memories_dir = str(mock_env / "toys/health/memories")
    return InformationEngine(base_memories_dir=base_memories_dir)
