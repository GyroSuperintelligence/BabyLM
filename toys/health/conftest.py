"""
Optimized pytest configuration for GyroSI test suite.
Uses main meta files but isolates all test data to temporary directories.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from baby.contracts import AgentConfig
from baby.intelligence import AgentPool, GyroSI
from baby.policies import OrbitStore
from fastapi.testclient import TestClient

# Main files (read-only, shared across tests)
PROJECT_ROOT = Path(__file__).parent.parent.parent
MAIN_META = PROJECT_ROOT / "memories" / "public" / "meta"
MAIN_TOKENIZERS = PROJECT_ROOT / "memories" / "public" / "tokenizers"

# Test isolation directory
TEST_BASE = Path(__file__).parent / "temp_memories"


@pytest.fixture(scope="session", autouse=True)
def test_environment() -> Generator[None, None, None]:
    """Setup isolated test environment."""
    TEST_BASE.mkdir(exist_ok=True)
    yield
    shutil.rmtree(TEST_BASE, ignore_errors=True)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Isolated temporary directory per test."""
    with tempfile.TemporaryDirectory(dir=TEST_BASE, prefix="test_") as tmpdir:
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
    
    pool = AgentPool(meta_paths["ontology"], public_path)
    yield pool
    pool.close_all()


@pytest.fixture
def multi_agent_setup(meta_paths: Dict[str, str], temp_dir: Path) -> Generator[Dict[str, Any], None, None]:
    """Multi-agent setup with public/private knowledge separation."""
    public_path = str(temp_dir / "public.pkl.gz")
    
    # Initialize public store
    OrbitStore(public_path).close()
    
    # Create agent configs
    user_config: AgentConfig = {
        "ontology_path": meta_paths["ontology"],
        "public_knowledge_path": public_path,
        "private_knowledge_path": str(temp_dir / "user_private.pkl.gz"),
        "enable_phenomenology_storage": True,
        "phenomenology_map_path": meta_paths["phenomenology"],
    }
    
    assistant_config: AgentConfig = {
        "ontology_path": meta_paths["ontology"], 
        "public_knowledge_path": public_path,
        "private_knowledge_path": str(temp_dir / "assistant_private.pkl.gz"),
        "enable_phenomenology_storage": True,
        "phenomenology_map_path": meta_paths["phenomenology"],
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
def test_client() -> Generator[TestClient, None, None]:
    """FastAPI test client."""
    from toys.communication.external_adapter import app
    with TestClient(app) as client:
        yield client


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


# Test utilities
def assert_phenotype_valid(entry: Dict[str, Any]) -> None:
    """Validate phenotype entry structure."""
    required_fields = [
        "phenotype", "exon_mask", "confidence", 
        "context_signature", "usage_count", "governance_signature"
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


def assert_no_main_pollution() -> None:
    """Ensure no test data polluted main memories folder."""
    main_memories = PROJECT_ROOT / "memories"
    
    # Check no test files in main directories
    if main_memories.exists():
        for item in main_memories.rglob("*test*"):
            if item.is_file():
                assert False, f"Test pollution detected: {item}"
        
        # Check no agent directories in main private area
        private_dir = main_memories / "private" / "agents"
        if private_dir.exists():
            test_agents = [d for d in private_dir.iterdir() if d.name.startswith("test")]
            assert not test_agents, f"Test agent pollution: {test_agents}"


# Export utilities for use in tests
__all__ = [
    "assert_phenotype_valid",
    "assert_ontology_valid", 
    "assert_no_main_pollution",
]