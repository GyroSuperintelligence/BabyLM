"""
Consolidated pytest configuration for the GyroSI test suite.
Enforces canonical triad (user, system, assistant) and fixes msgpack tuple serialization.
"""

import os
import sys
import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, Callable, Tuple, Set, TypedDict, cast, Literal
import pytest
from fastapi.testclient import TestClient
from baby.contracts import AgentConfig, PreferencesConfig
from baby.intelligence import AgentPool, GyroSI
from baby.policies import OrbitStore

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
# Fix msgpack tuple/list serialization issues
# =============================================================================


def _fix_context_signature_tuple(entry: Any) -> Any:
    """Convert context_signature from list back to tuple if needed."""
    if isinstance(entry, dict):
        entry = entry.copy()  # Don't modify original
        cs = entry.get("context_signature")
        if isinstance(cs, list) and len(cs) == 2:
            entry["context_signature"] = tuple(cs)
        # Also fix _original_context if present
        oc = entry.get("_original_context")
        if isinstance(oc, list) and len(oc) == 2:
            entry["_original_context"] = tuple(oc)
    return entry


@pytest.fixture(scope="session", autouse=True)
def fix_orbit_store_tuple_serialization() -> Generator[None, None, None]:
    """Fix OrbitStore tuple/list serialization for the entire test session."""
    from baby.policies import OrbitStore

    # Save original methods
    original_get = OrbitStore.get
    original_iter_entries = OrbitStore.iter_entries

    def get_with_tuple_fix(self: OrbitStore, context_key: Any) -> Any:
        result = original_get(self, context_key)
        return _fix_context_signature_tuple(result)

    def iter_entries_with_tuple_fix(self: OrbitStore) -> Any:
        for key, entry in original_iter_entries(self):
            yield key, _fix_context_signature_tuple(entry)

    # Apply monkey patches using setattr to avoid mypy errors
    setattr(OrbitStore, "get", get_with_tuple_fix)
    setattr(OrbitStore, "iter_entries", iter_entries_with_tuple_fix)

    try:
        yield
    finally:
        # Restore original methods
        setattr(OrbitStore, "get", original_get)
        setattr(OrbitStore, "iter_entries", original_iter_entries)


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

    # Copy tokenizers, which are needed at runtime
    shutil.copytree(MAIN_TOKENIZERS / "bert-base-uncased", public_dir / "tokenizers" / "bert-base-uncased")

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
        "public_knowledge": {"path": str(public_dir / "knowledge.mpk")},
        "private_knowledge": {"base_path": str(private_dir / "agents")},
        "ontology": {
            "ontology_map_path": main_meta_files["ontology"],
            "phenomenology_map_path": main_meta_files["phenomenology"],
        },
        "tokenizer": {"name": "bert-base-uncased"},
    }
    prefs_path.write_text(json.dumps(test_preferences, indent=2))

    # Initialize the public knowledge file so it exists
    OrbitStore(test_preferences["public_knowledge"]["path"], append_only=True).close()

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
    OrbitStore(public_path, append_only=True).close()

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


@pytest.fixture
def gyrosi_agent(assistant_agent: GyroSI) -> GyroSI:
    """Legacy alias for assistant_agent for backward compatibility."""
    return assistant_agent


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
def isolated_agent_factory() -> Generator[Callable[[Path], GyroSI], None, None]:
    """Factory for creating isolated agents outside the main pool."""
    created_agents = []

    def _make_isolated_agent(tmp_path: Path) -> GyroSI:
        """Create a completely isolated agent in tmp_path."""
        agent_dir = tmp_path / "isolated_agent"
        agent_dir.mkdir(parents=True, exist_ok=True)

        config: AgentConfig = {
            "ontology_path": str(MAIN_META / "ontology_keys.npy"),
            "phenomenology_map_path": str(MAIN_META / "phenomenology_map.npy"),
            "knowledge_path": str(agent_dir / "knowledge.mpk"),
            "base_path": str(tmp_path),
        }

        agent = GyroSI(config, agent_id=f"isolated_{uuid.uuid4().hex[:8]}", base_path=tmp_path)
        created_agents.append(agent)
        return agent

    yield _make_isolated_agent

    # Cleanup
    for agent in created_agents:
        agent.close()


# =============================================================================
# API testing fixtures
# =============================================================================


@pytest.fixture
def api_test_setup(test_env: TestEnvDict) -> Generator[Tuple[TestClient, AgentPool], None, None]:
    """
    Sets up the full environment for API testing.
    """
    # Point the adapter to our temporary, test-specific preferences file
    os.environ["GYROSI_PREFERENCES_PATH"] = test_env["preferences_path"]

    # The API creates its own pool on import, so we must reload it
    import importlib
    from toys.communication import external_adapter as ea

    importlib.reload(ea)

    # Ensure the triad exists for the API's pool
    ea.agent_pool.ensure_triad()

    client = TestClient(ea.app)

    yield client, ea.agent_pool

    # Cleanup
    ea.agent_pool.close_all()
    if "GYROSI_PREFERENCES_PATH" in os.environ:
        del os.environ["GYROSI_PREFERENCES_PATH"]


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
        "phenotype": "P[100:42]",
        "exon_mask": 0b10101010,
        "confidence": 0.75,
        "context_signature": (100, 42),  # Always a tuple
        "usage_count": 5,
        "created_at": 1234567890.0,
        "last_updated": 1234567890.0,
        "governance_signature": {"neutral": 4, "li": 1, "fg": 1, "bg": 0, "dyn": 2},
        "_original_context": None,
    }


# =============================================================================
# Legacy fixtures for backward compatibility (deprecated)
# =============================================================================


@pytest.fixture
def gyrosi_agent_factory() -> Callable[[Literal["user", "system", "assistant"]], GyroSI]:
    """
    DEPRECATED: Use agent_factory or specific agent fixtures instead.
    This factory only works with canonical triad roles.
    """

    def _legacy_factory(role: Literal["user", "system", "assistant"], _unused_filename: str = "") -> GyroSI:
        # Import here to avoid circular dependency
        pytest.fail(
            f"gyrosi_agent_factory is deprecated. Use 'agent_factory' or specific fixtures like '{role}_agent' instead."
        )

    return _legacy_factory


@pytest.fixture
def agent_pool_factory() -> Callable[[], AgentPool]:
    """
    DEPRECATED: Use the session-scoped 'agent_pool' fixture instead.
    """

    def _legacy_pool_factory(_unused_ids: Set[str] = set()) -> AgentPool:
        pytest.fail("agent_pool_factory is deprecated. Use the session-scoped 'agent_pool' fixture instead.")

    return _legacy_pool_factory


# =============================================================================
# Utility functions (exported for test use)
# =============================================================================


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
