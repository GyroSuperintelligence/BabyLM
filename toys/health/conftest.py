"""
Shared pytest fixtures and configuration for GyroSI test suite.
"""

import os
import shutil
import json
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any

# Add the baby module to the Python path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from baby import (
    discover_and_save_manifold,
    build_phenomenology_map,
    GyroSI,
    AgentPool,
)
from baby.types import ManifoldData, AgentConfig, PreferencesConfig
from baby.information import OrbitStore


# Base temp directory for all tests
BASE_TEMP_DIR = Path(__file__).parent / "memories"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup and teardown test environment."""
    # Create base temp directory
    BASE_TEMP_DIR.mkdir(exist_ok=True)

    yield

    # Cleanup after all tests (optional - can be disabled for debugging)
    # shutil.rmtree(BASE_TEMP_DIR, ignore_errors=True)


@pytest.fixture
def temp_dir():
    """Create a unique temporary directory for each test."""
    test_dir = BASE_TEMP_DIR / f"test_{os.getpid()}_{id(object())}"
    test_dir.mkdir(parents=True, exist_ok=True)

    yield str(test_dir)

    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def manifold_data(temp_dir):
    """Create and return test manifold data."""
    manifold_path = os.path.join(temp_dir, "manifold", "ontology_map.json")
    os.makedirs(os.path.dirname(manifold_path), exist_ok=True)

    # For testing, create a smaller mock manifold
    # In real tests, you'd use discover_and_save_manifold
    mock_manifold: ManifoldData = {
        "schema_version": "1.0.0",
        "ontology_map": {i: i for i in range(1000)},  # Mock 1000 states
        "endogenous_modulus": 788_986,  # Keep the real constant
        "manifold_diameter": 6,
        "total_states": 788_986,
        "build_timestamp": 1234567890.0,
    }

    with open(manifold_path, "w") as f:
        json.dump(mock_manifold, f)

    return manifold_path, mock_manifold


@pytest.fixture
def real_manifold(temp_dir):
    """Create the real manifold (expensive - use sparingly)."""
    manifold_path = os.path.join(temp_dir, "manifold", "ontology_map.json")
    os.makedirs(os.path.dirname(manifold_path), exist_ok=True)

    # This is expensive but necessary for integration tests
    discover_and_save_manifold(manifold_path)

    # Also build canonical map
    canonical_path = os.path.join(temp_dir, "manifold", "phenomenology_map.json")
    build_phenomenology_map(manifold_path, canonical_path)

    with open(manifold_path, "r") as f:
        manifold_data = json.load(f)

    return manifold_path, canonical_path, manifold_data


@pytest.fixture
def orbit_store(temp_dir):
    """Create an OrbitStore instance."""
    store_path = os.path.join(temp_dir, "knowledge.pkl.gz")
    store = OrbitStore(store_path)
    yield store
    store.close()


@pytest.fixture
def multi_agent_store(temp_dir):
    """Create an OrbitStore overlay instance."""
    public_path = os.path.join(temp_dir, "public", "knowledge.pkl.gz")
    private_path = os.path.join(temp_dir, "private", "agent1", "knowledge.pkl.gz")

    # Create public knowledge
    os.makedirs(os.path.dirname(public_path), exist_ok=True)
    public_store = OrbitStore(public_path)
    public_store.put((0, 0), {"phenotype": "public", "confidence": 0.9})
    public_store.close()

    public_store = OrbitStore(public_path, read_only=True)
    private_store = OrbitStore(private_path)
    store = OrbitStore(private_path, public_store=public_store, private_store=private_store)
    yield store
    store.close()


@pytest.fixture
def agent_config(manifold_data, temp_dir) -> AgentConfig:
    """Create a test agent configuration."""
    manifold_path, _ = manifold_data
    return {
        "manifold_path": manifold_path,
        "knowledge_path": os.path.join(temp_dir, "knowledge.pkl.gz"),
        "enable_canonical_storage": False,
    }


@pytest.fixture
def gyrosi_agent(agent_config):
    """Create a GyroSI agent instance."""
    agent = GyroSI(agent_config)
    yield agent
    agent.close()


@pytest.fixture
def agent_pool(manifold_data, temp_dir):
    """Create an agent pool."""
    manifold_path, _ = manifold_data
    public_knowledge = os.path.join(temp_dir, "public_knowledge.pkl.gz")

    # Create empty public knowledge
    os.makedirs(os.path.dirname(public_knowledge), exist_ok=True)
    store = OrbitStore(public_knowledge)
    store.close()

    pool = AgentPool(manifold_path, public_knowledge)
    yield pool
    pool.close_all()


@pytest.fixture
def preferences_config() -> PreferencesConfig:
    """Create test preferences configuration."""
    return {
        "storage_backend": "pickle",
        "compression_level": 6,
        "max_file_size_mb": 100,
        "enable_auto_decay": False,
        "decay_interval_hours": 24,
        "decay_factor": 0.999,
        "confidence_threshold": 0.05,
        "max_agents_in_memory": 10,
        "agent_eviction_policy": "lru",
        "agent_ttl_minutes": 60,
        "encryption_enabled": False,
        "enable_profiling": False,
        "batch_size": 100,
        "cache_size_mb": 10,
    }


@pytest.fixture
def sample_phenotype_entry():
    """Create a sample phenotype entry for testing."""
    return {
        "phenotype": "A",
        "memory_mask": 0b10101010,
        "confidence": 0.75,
        "context_signature": (100, 42),
        "semantic_address": 12345,
        "usage_count": 10,
        "age_counter": 5,
        "created_at": 1234567890.0,
        "last_updated": 1234567890.0,
    }


@pytest.fixture
def mock_time(monkeypatch):
    """Mock time.time() for deterministic tests."""
    current_time = [1234567890.0]

    def mock_time_func():
        return current_time[0]

    def advance_time(seconds):
        current_time[0] += seconds

    monkeypatch.setattr("time.time", mock_time_func)

    # Return controller
    class TimeController:
        def advance(self, seconds):
            advance_time(seconds)

        @property
        def current(self):
            return current_time[0]

    return TimeController()


# Test data generators


def generate_test_introns(count: int, seed: int = 42) -> list:
    """Generate deterministic test introns."""
    import random

    random.seed(seed)
    return [random.randint(0, 255) for _ in range(count)]


def generate_test_bytes(text: str) -> bytes:
    """Convert text to bytes with padding if needed."""
    return text.encode("utf-8")


# Assertion helpers


def assert_phenotype_entry_valid(entry: Dict[str, Any]):
    """Assert that a phenotype entry has all required fields."""
    required_fields = ["phenotype", "memory_mask", "confidence", "context_signature", "usage_count"]
    for field in required_fields:
        assert field in entry, f"Missing required field: {field}"

    assert isinstance(entry["phenotype"], str)
    assert isinstance(entry["memory_mask"], int)
    assert 0 <= entry["memory_mask"] <= 255
    assert 0 <= entry["confidence"] <= 1.0
    assert isinstance(entry["context_signature"], tuple)
    assert len(entry["context_signature"]) == 2


def assert_manifold_valid(manifold_data: Dict[str, Any]):
    """Assert that manifold data is valid."""
    assert manifold_data["endogenous_modulus"] == 788_986
    assert manifold_data["manifold_diameter"] == 6
    assert "ontology_map" in manifold_data
    assert "schema_version" in manifold_data


# Performance measurement helpers


class Timer:
    """Simple timer context manager for performance tests."""

    def __init__(self):
        self.elapsed = 0

    def __enter__(self):
        import time

        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        import time

        self.elapsed = time.perf_counter() - self.start
