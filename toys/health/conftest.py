"""
Optimized pytest configuration for GyroSI test suite.
Uses main meta files but isolates all test data to temporary directories.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from baby.contracts import AgentConfig  # noqa: E402
from baby.intelligence import AgentPool, GyroSI  # noqa: E402
from baby.policies import OrbitStore  # noqa: E402

# Main files (read-only, shared across tests)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MAIN_META = PROJECT_ROOT / "memories" / "public" / "meta"
MAIN_TOKENIZERS = PROJECT_ROOT / "memories" / "public" / "tokenizers"


@pytest.fixture(scope="session", autouse=True)
def test_environment() -> Generator[None, None, None]:
    """No-op: all cleanup is handled by TemporaryDirectory context managers."""
    yield


@pytest.fixture(scope="session")
def session_temp_dir() -> Generator[Path, None, None]:
    """Session-wide temporary directory that is auto-deleted after tests."""
    with tempfile.TemporaryDirectory(prefix="gyrosi_test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_dir(session_temp_dir: Path) -> Generator[Path, None, None]:
    """Isolated temporary directory per test, under the session temp dir."""
    with tempfile.TemporaryDirectory(dir=session_temp_dir, prefix="test_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def meta_paths() -> Dict[str, str]:
    """Main meta file paths (read-only)."""
    return {
        "ontology": str(MAIN_META / "ontology_map.json"),
        "epistemology": str(MAIN_META / "epistemology.npy"),
        "phenomenology": str(MAIN_META / "phenomenology_map.json"),
        "theta": str(MAIN_META / "theta.npy"),
        "tokenizer": str(MAIN_TOKENIZERS / "bert-base-uncased"),
    }


@pytest.fixture
def temp_store(temp_dir: Path) -> Generator[OrbitStore, None, None]:
    """Temporary OrbitStore for testing."""
    store = OrbitStore(str(temp_dir / "test_knowledge.pkl.gz"))
    yield store
    store.close()


@pytest.fixture
def gyrosi_agent(meta_paths: Dict[str, str], temp_dir: Path) -> Generator[GyroSI, None, None]:
    """GyroSI agent with isolated storage."""
    config: AgentConfig = {
        "ontology_path": meta_paths["ontology"],
        "knowledge_path": str(temp_dir / "agent_knowledge.pkl.gz"),
        "enable_phenomenology_storage": True,
        "phenomenology_map_path": meta_paths["phenomenology"],
        "public_knowledge_path": str(temp_dir / "public_knowledge.pkl.gz"),
        "private_knowledge_path": str(temp_dir / "private_knowledge.pkl.gz"),
        "private_agents_base_path": str(temp_dir / "agents"),
        "base_path": str(temp_dir),
    }
    agent = GyroSI(config)
    yield agent
    agent.close()


@pytest.fixture
def agent_pool(meta_paths: Dict[str, str], temp_dir: Path) -> Generator[AgentPool, None, None]:
    """AgentPool with isolated storage."""
    public_path = str(temp_dir / "public_knowledge.pkl.gz")
    # Initialize empty public store
    OrbitStore(public_path).close()
    pool = AgentPool(
        meta_paths["ontology"],
        public_path,
        allowed_ids={"user", "system", "assistant"},
        allow_auto_create=True,
        private_agents_base_path=str(temp_dir / "agents"),
    )
    yield pool
    pool.close_all()


@pytest.fixture
def multi_agent_setup(meta_paths: Dict[str, str], temp_dir: Path) -> Generator[Dict[str, Any], None, None]:
    """Multi-agent setup with public/private knowledge separation."""
    public_path = str(temp_dir / "public.pkl.gz")
    OrbitStore(public_path).close()
    user_config: AgentConfig = {
        "ontology_path": meta_paths["ontology"],
        "public_knowledge_path": public_path,
        "private_knowledge_path": str(temp_dir / "user_private.pkl.gz"),
        "enable_phenomenology_storage": True,
        "phenomenology_map_path": meta_paths["phenomenology"],
        "private_agents_base_path": str(temp_dir / "agents"),
        "base_path": str(temp_dir),
    }
    assistant_config: AgentConfig = {
        "ontology_path": meta_paths["ontology"],
        "public_knowledge_path": public_path,
        "private_knowledge_path": str(temp_dir / "assistant_private.pkl.gz"),
        "enable_phenomenology_storage": True,
        "phenomenology_map_path": meta_paths["phenomenology"],
        "private_agents_base_path": str(temp_dir / "agents"),
        "base_path": str(temp_dir),
    }
    user_agent = GyroSI(user_config, agent_id="test_user")
    assistant_agent = GyroSI(assistant_config, agent_id="test_assistant")
    yield {
        "user": user_agent,
        "assistant": assistant_agent,
        "public_path": public_path,
        "temp_dir": temp_dir,
    }
    user_agent.close()
    assistant_agent.close()


@pytest.fixture
def sample_phenotype() -> Dict[str, Any]:
    """Sample phenotype entry for testing."""
    return {
        "phenotype": "P[100:42]",
        "exon_mask": 0b10101010,
        "confidence": 0.75,
        "context_signature": (100, 42),
        "usage_count": 5,
        "created_at": 1234567890.0,
        "last_updated": 1234567890.0,
        "governance_signature": {"neutral": 4, "li": 1, "fg": 1, "bg": 0, "dyn": 2},
        "_original_context": None,
    }


@pytest.fixture
def test_bytes() -> bytes:
    """Sample test bytes."""
    return b"Hello, GyroSI!"


@pytest.fixture
def client(patched_adapter_pool):
    from toys.communication.external_adapter import app
    return TestClient(app)


@pytest.fixture
def patched_adapter_pool(tmp_path, meta_paths):
    from toys.communication import external_adapter as ea

    pub = tmp_path / "public.pkl.gz"
    OrbitStore(str(pub)).close()

    if hasattr(ea, "agent_pool"):
        ea.agent_pool.close_all()

    ea.agent_pool = AgentPool(
        ontology_path=meta_paths["ontology"],
        base_knowledge_path=str(pub),
        allowed_ids={"user", "system", "assistant"},
        allow_auto_create=True,
        private_agents_base_path="agents",
        base_path=tmp_path,
    )
    ea.agent_pool.ensure_triad()

    yield ea.agent_pool

    ea.agent_pool.close_all()


# Test utilities
def assert_phenotype_valid(entry: Dict[str, Any]) -> None:
    """Validate phenotype entry structure."""
    required_fields = [
        "phenotype",
        "exon_mask",
        "confidence",
        "context_signature",
        "usage_count",
        "governance_signature",
    ]
    for field in required_fields:
        assert field in entry, f"Missing field: {field}"

    assert 0 <= entry["exon_mask"] <= 255
    assert 0 <= entry["confidence"] <= 1.0
    assert len(entry["context_signature"]) == 2


def assert_ontology_valid(data: Dict[str, Any]) -> None:
    """Validate ontology data structure."""
    assert data["endogenous_modulus"] == 788_986
    assert data["ontology_diameter"] == 6
    assert "ontology_map" in data
    assert "schema_version" in data


# Export utilities for use in tests
__all__ = [
    "assert_phenotype_valid",
    "assert_ontology_valid",
]


# Add a session finish hook to check for pollution in main memories/private/agents


def pytest_sessionfinish(session: Any, exitstatus: Any) -> None:
    # No manual cleanup needed; all temp dirs are auto-removed by TemporaryDirectory
    pass
