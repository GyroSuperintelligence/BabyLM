"""
Consolidated pytest configuration for the GyroSI test suite.
Enforces canonical triad (user, system, assistant) and fixes binary_struct tuple serialization.
"""

import sys
import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, Callable, Tuple, TypedDict, cast, Literal
import pytest
from fastapi.testclient import TestClient
from baby.contracts import PreferencesConfig, AgentConfig
from baby.intelligence import AgentPool, GyroSI
from baby.policies import OrbitStore
from contextlib import ExitStack
import uuid

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --- Constants for Main Project Files (Read-Only) ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
MAIN_MEMORIES = PROJECT_ROOT / "memories"
MAIN_META = MAIN_MEMORIES / "public" / "meta"
MAIN_TOKENIZERS = MAIN_MEMORIES / "public" / "tokenizers"


class TestEnvDict(TypedDict):
    memories_dir: Path
    preferences_path: str
    preferences: Dict[str, Any]
    main_meta_files: Dict[str, str]


# =============================================================================
# Core test environment setup
# =============================================================================


@pytest.fixture(scope="session")
def test_env(tmp_path_factory: pytest.TempPathFactory) -> TestEnvDict:
    """
    Session-scoped fixture that creates a pristine, temporary 'memories'
    directory structure.
    """
    # Create a base temporary directory for the entire test session
    base_dir = tmp_path_factory.mktemp("gyrosi_test_session")

    # Define the required test directory structure
    memories_dir = base_dir / "toys" / "health" / "temp" / "memories"
    public_dir = memories_dir / "public"
    private_dir = memories_dir / "private"

    (public_dir / "meta").mkdir(parents=True)
    (private_dir / "agents").mkdir(parents=True)

    # Symlink tokenizers, which are needed at runtime (faster than copytree)
    tokenizer_src = MAIN_TOKENIZERS / "bert-base-uncased"
    tokenizer_dst = public_dir / "tokenizers" / "bert-base-uncased"
    tokenizer_dst.parent.mkdir(parents=True, exist_ok=True)
    if not tokenizer_dst.exists():
        os.symlink(tokenizer_src, tokenizer_dst)

    # The main, read-only meta files
    main_meta_files: Dict[str, str] = {
        "ontology": str(MAIN_META / "ontology_keys.npy"),
        "phenomenology": str(MAIN_META / "phenomenology_map.npy"),
        "theta": str(MAIN_META / "theta.npy"),
        "epistemology": str(MAIN_META / "epistemology.npy"),
    }

    # Create the master test preferences file within the temp directory
    prefs_path = memories_dir / "memory_preferences.json"
    test_preferences: Dict[str, Any] = {
        "base_path": str(memories_dir),
        "public_knowledge": {"path": str(public_dir / "knowledge.bin")},
        "private_knowledge": {"base_path": str(private_dir / "agents")},
        "ontology": {
            "ontology_map_path": main_meta_files["ontology"],
            "phenomenology_map_path": main_meta_files["phenomenology"],
        },
        "tokenizer": {"name": "bert-base-uncased"},
        "pruning": {"enable_auto_decay": False, "confidence_threshold": 0.05},
        "write_threshold": 100,
    }
    prefs_path.write_text(json.dumps(test_preferences, indent=2))

    # Initialize the public knowledge file so it exists
    store = OrbitStore(
        test_preferences["public_knowledge"]["path"], append_only=True, base_path=Path(test_preferences["base_path"])
    )
    store.close()

    return {
        "memories_dir": memories_dir,
        "preferences_path": str(prefs_path),
        "preferences": test_preferences,
        "main_meta_files": main_meta_files,
    }


# =============================================================================
# Session-scoped agent pool with canonical triad
# =============================================================================


@pytest.fixture(scope="session")
def agent_pool(test_env: TestEnvDict) -> Generator[AgentPool, None, None]:
    """Session-scoped agent pool with the canonical triad (user, system, assistant)."""
    prefs = test_env["preferences"]
    main_meta_files = test_env["main_meta_files"]

    # Initialize empty public store
    public_path = prefs["public_knowledge"]["path"]
    store = OrbitStore(public_path, append_only=True, base_path=Path(prefs["base_path"]))
    store.close()

    pool = AgentPool(
        ontology_path=main_meta_files["ontology"],
        base_knowledge_path=public_path,
        preferences=cast(PreferencesConfig, prefs),
        allowed_ids={"user", "system", "assistant"},
        allow_auto_create=True,
        private_agents_base_path=prefs["private_knowledge"]["base_path"],
        base_path=Path(prefs["base_path"]),
    )

    # Ensure triad exists immediately
    pool.ensure_triad()

    yield pool

    # Final cleanup
    pool.close_all()


# =============================================================================
# Canonical triad agent fixtures
# =============================================================================


@pytest.fixture
def user_agent(agent_pool: AgentPool) -> GyroSI:
    """User agent from the canonical triad."""
    return agent_pool.get("user")


@pytest.fixture
def assistant_agent(agent_pool: AgentPool) -> GyroSI:
    """Assistant agent from the canonical triad."""
    return agent_pool.get("assistant")


@pytest.fixture
def system_agent(agent_pool: AgentPool) -> GyroSI:
    """System agent from the canonical triad."""
    return agent_pool.get("system")


# =============================================================================
# Agent factory for triad roles
# =============================================================================


@pytest.fixture
def agent_factory(agent_pool: AgentPool) -> Callable[[Literal["user", "system", "assistant"]], GyroSI]:
    """Factory that returns agents from the canonical triad by role."""

    def _get_agent(role: Literal["user", "system", "assistant"]) -> GyroSI:
        return agent_pool.get(role)

    return _get_agent


# =============================================================================
# Multi-agent scenario fixture
# =============================================================================


@pytest.fixture
def multi_agent_scenario(user_agent: GyroSI, assistant_agent: GyroSI, test_env: TestEnvDict) -> Dict[str, Any]:
    """
    Multi-agent scenario using the canonical triad.
    """
    return {
        "user": user_agent,
        "assistant": assistant_agent,
        "public_path": test_env["preferences"]["public_knowledge"]["path"],
        "memories_dir": test_env["memories_dir"],
    }


# =============================================================================
# Isolated agent factory (for tests that truly need isolation)
# =============================================================================


@pytest.fixture
def isolated_agent_factory(tmp_path: Path, test_env: TestEnvDict) -> Generator[Callable[[Path], GyroSI], None, None]:
    stack = ExitStack()

    def _make_isolated_agent(agent_dir: Path) -> GyroSI:
        agent_dir.mkdir(parents=True, exist_ok=True)
        config: AgentConfig = {
            "ontology_path": test_env["main_meta_files"]["ontology"],
            "phenomenology_map_path": test_env["main_meta_files"]["phenomenology"],
            "knowledge_path": str(agent_dir / "knowledge.bin"),
            "base_path": str(tmp_path),
        }
        agent = GyroSI(config, agent_id=f"isolated_{uuid.uuid4().hex[:8]}", base_path=tmp_path)
        stack.callback(agent.close)
        return agent

    yield _make_isolated_agent
    stack.close()


@pytest.fixture
def gyrosi_agent(assistant_agent: GyroSI) -> GyroSI:
    """Legacy alias for assistant_agent for backward compatibility."""
    return assistant_agent


# =============================================================================
# API testing fixtures
# =============================================================================


@pytest.fixture
def api_test_setup(test_env: TestEnvDict, request: Any) -> Generator[Tuple[TestClient, AgentPool], None, None]:
    os.environ["GYROSI_PREFERENCES_PATH"] = test_env["preferences_path"]
    from toys.communication import external_adapter as ea

    ea.agent_pool.ensure_triad()
    client = TestClient(ea.app)

    def cleanup() -> None:
        # Do NOT close the global agent_pool here; only remove env var
        if "GYROSI_PREFERENCES_PATH" in os.environ:
            del os.environ["GYROSI_PREFERENCES_PATH"]

    request.addfinalizer(cleanup)
    yield client, ea.agent_pool


@pytest.fixture
def test_client(api_test_setup: Tuple[TestClient, AgentPool]) -> TestClient:
    """A convenient fixture that provides just the FastAPI TestClient."""
    return api_test_setup[0]


# =============================================================================
# Utility fixtures
# =============================================================================


@pytest.fixture
def sample_phenotype() -> Dict[str, Any]:
    """Sample phenotype entry for testing."""
    return {
        "key": (100, 42),  # (state_index, token_id)
        "mask": 0b10101010,
        "conf": 0.75,
    }


# =============================================================================
# Utility functions (exported for test use)
# =============================================================================


def assert_phenotype_valid(entry: Dict[str, Any]) -> None:
    """Validate phenotype entry structure."""
    for fld in ("key", "mask", "conf"):
        assert fld in entry, f"Missing field: {fld}"
    s_idx, tok_id = entry["key"]
    assert 0 <= s_idx < 788_986
    assert tok_id >= 0
    assert 0 <= entry["mask"] <= 255
    assert 0.0 <= entry["conf"] <= 1.0


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
