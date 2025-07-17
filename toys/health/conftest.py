"""
Shared pytest fixtures and configuration for GyroSI test suite.
"""

import os
import shutil
# Try to use ujson for speed, fall back to standard json if unavailable
try:
    import ujson as json  # type: ignore[import]
except ImportError:
    import json  # type: ignore
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any

# Add the baby module to the Python path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from baby import (
    discover_and_save_ontology,
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
def ontology_data(temp_dir):
    """Create and return test ontology data."""
    ontology_path = os.path.join(temp_dir, "ontology", "ontology_map.json")
    os.makedirs(os.path.dirname(ontology_path), exist_ok=True)

    # For testing, create a smaller mock ontology
    # In real tests, you'd use discover_and_save_ontology
    mock_ontology: ManifoldData = {
        "schema_version": "1.0.0",
        "ontology_map": {i: i for i in range(1000)},  # Mock 1000 states
        "endogenous_modulus": 788_986,  # Keep the real constant
        "ontology_diameter": 6,
        "total_states": 788_986,
        "build_timestamp": 1234567890.0,
    }

    with open(ontology_path, "w") as f:
        json.dump(mock_ontology, f)

    return ontology_path, mock_ontology


@pytest.fixture
def real_ontology(temp_dir):
    """Create the real ontology (expensive - use sparingly)."""
    ontology_path = os.path.join(temp_dir, "ontology", "ontology_map.json")
    os.makedirs(os.path.dirname(ontology_path), exist_ok=True)

    # This is expensive but necessary for integration tests
    discover_and_save_ontology(ontology_path)

    # Also build phenomenology map
    phenomenology_path = os.path.join(temp_dir, "ontology", "phenomenology_map.json")
    build_phenomenology_map(ontology_path, phenomenology_path)

    with open(ontology_path, "r") as f:
        ontology_data = json.load(f)

    return ontology_path, phenomenology_path, ontology_data


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
def agent_config(ontology_data, temp_dir) -> AgentConfig:
    """Create a test agent configuration."""
    ontology_path, _ = ontology_data
    return {
        "ontology_path": ontology_path,
        "knowledge_path": os.path.join(temp_dir, "knowledge.pkl.gz"),
        "enable_phenomenology_storage": False,
    }


@pytest.fixture
def gyrosi_agent(agent_config):
    """Create a GyroSI agent instance."""
    agent = GyroSI(agent_config)
    yield agent
    agent.close()


@pytest.fixture
def agent_pool(ontology_data, temp_dir):
    """Create an agent pool."""
    ontology_path, _ = ontology_data
    public_knowledge = os.path.join(temp_dir, "public_knowledge.pkl.gz")

    # Create empty public knowledge
    os.makedirs(os.path.dirname(public_knowledge), exist_ok=True)
    store = OrbitStore(public_knowledge)
    store.close()

    pool = AgentPool(ontology_path, public_knowledge)
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


def assert_ontology_valid(ontology_data: Dict[str, Any]):
    """Assert that ontology data is valid."""
    assert ontology_data["endogenous_modulus"] == 788_986
    assert ontology_data["ontology_diameter"] == 6
    assert "ontology_map" in ontology_data
    assert "schema_version" in ontology_data


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
