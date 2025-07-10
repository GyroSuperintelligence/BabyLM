# toys/health/conftest.py

import os
import json
import numpy as np
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch

from baby.governance import derive_canonical_patterns
from baby.information import InformationEngine, assign_agent_uuid, shard_path
from baby.intelligence import initialize_intelligence_engine


# --- Golden Environment Fixture ---
@pytest.fixture
def mock_env(tmp_path):
    """
    Creates a temporary, self-contained 'memories' and 'baby' environment,
    changes the current working directory into it, and cleans up afterward.
    This is the foundational fixture for all tests needing file I/O.
    """
    # Create the base directories
    memories_dir = tmp_path / "memories"
    baby_dir = tmp_path / "baby"
    (memories_dir / "public" / "masks").mkdir(parents=True, exist_ok=True)
    (memories_dir / "public" / "formats").mkdir(parents=True, exist_ok=True)
    (memories_dir / "public" / "threads").mkdir(parents=True, exist_ok=True)
    (memories_dir / "public" / "keys").mkdir(parents=True, exist_ok=True)
    (memories_dir / "private" / "agents").mkdir(parents=True, exist_ok=True)
    baby_dir.mkdir(exist_ok=True)

    # Create default memory_preferences.json
    mem_prefs = {
        "sharding": {"width": 2, "max_files": 30000, "second_level": True},
        "storage_config": {"max_thread_size_mb": 64, "encryption_algorithm": "AES-256-GCM"},
        "format_config": {"default_cgm_version": "1.0.0", "max_character_label_length": 128},
    }
    with open(memories_dir / "memory_preferences.json", "w") as f:
        json.dump(mem_prefs, f, indent=2)

    # Create default mask files needed by InferenceEngine
    patterns_array, _ = derive_canonical_patterns()
    patterns_array.tofile(memories_dir / "public" / "masks" / "epigenome.dat")
    genome_mask = np.arange(256, dtype=np.uint8)
    genome_mask.tofile(memories_dir / "public" / "masks" / "genome.dat")

    # Change the CWD to the temp path so "memories/..." paths work
    original_cwd = Path.cwd()
    os.chdir(tmp_path)

    yield tmp_path  # The test runs here

    # Teardown: Change back to the original directory
    os.chdir(original_cwd)


# --- Engine Fixtures (built on top of mock_env) ---


@pytest.fixture
def public_intelligence_engine(mock_env):
    """Initializes an IntelligenceEngine in public/curation mode."""
    engine = initialize_intelligence_engine(agent_uuid=None, agent_secret=None)
    assert engine.agent_uuid is None
    assert engine.agent_secret is None
    return engine


@pytest.fixture
def initialized_intelligence_engine(mock_env):
    """Initializes an IntelligenceEngine in a predictable private-agent mode."""
    agent_uuid = "11111111-1111-1111-1111-111111111111"
    agent_secret = "test-secret-for-fixture"

    # Create baby_preferences.json with the secret to simulate a user's setup
    with open("baby/baby_preferences.json", "w") as f:
        json.dump({"agent_secret": agent_secret}, f)

    # Ensure the agent's directory structure exists *before* initializing the engine
    assign_agent_uuid(agent_uuid)

    # Initialize the engine explicitly for this agent
    engine = initialize_intelligence_engine(agent_uuid=agent_uuid, agent_secret=agent_secret)

    # Assertions to ensure the fixture is set up correctly
    assert engine.agent_uuid == agent_uuid
    assert engine.agent_secret == agent_secret
    private_dir = Path("memories/private/agents")
    agent_shard = shard_path(private_dir, engine.agent_uuid)
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
def information_engine():
    """Provides a standard InformationEngine instance."""
    return InformationEngine()


# This is optional, but good practice to keep it in conftest.py
@pytest.fixture(scope="session", autouse=True)
def cleanup_htmlcov():
    """Removes the htmlcov directory after the test session finishes."""
    yield
    if os.path.exists("htmlcov"):
        shutil.rmtree("htmlcov")
