import os
import sys
import json
import shutil
import atexit
from pathlib import Path
from typing import Any, Dict, Generator, Callable, Tuple, Set, TypedDict, cast
import pytest
from fastapi.testclient import TestClient
from baby.contracts import AgentConfig, PreferencesConfig
from baby.intelligence import AgentPool, GyroSI
from baby.policies import OrbitStore

"""
Truly consolidated pytest configuration for the GyroSI test suite.

This configuration uses the "Factory as a Fixture" pattern to eliminate
repetition. A single 'test_env' fixture sets up all necessary paths and
configurations. Factory fixtures then use this environment to build
agents, pools, and stores on demand, ensuring all setup logic is defined
only once.
"""

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# --- Constants for Main Project Files (Read-Only) ---
PROJECT_ROOT = Path(__file__).parent.parent.parent
MAIN_MEMORIES = PROJECT_ROOT / "memories"
MAIN_META = MAIN_MEMORIES / "public" / "meta"
MAIN_TOKENIZERS = MAIN_MEMORIES / "public" / "tokenizers"


# =============================================================================
# 1. CORE FIXTURE: The Single Source of Truth for Test Environments
# =============================================================================


class TestEnvDict(TypedDict):
    memories_dir: Path
    preferences_path: str
    preferences: Dict[str, Any]
    main_meta_files: Dict[str, str]


@pytest.fixture(scope="session")
def test_env(tmp_path_factory: pytest.TempPathFactory) -> TestEnvDict:
    """
    Session-scoped fixture that creates a pristine, temporary 'memories'
    directory structure. It acts as the single source of truth for all test
    paths and configurations. It yields a dictionary of essential paths.
    """
    # Create a base temporary directory for the entire test session
    base_dir = tmp_path_factory.mktemp("gyrosi_test_session")

    # Define the required test directory structure
    memories_dir = base_dir / "toys" / "health" / "temp" / "memories"
    public_dir = memories_dir / "public"
    private_dir = memories_dir / "private"

    (public_dir / "meta").mkdir(parents=True)
    (private_dir / "agents").mkdir(parents=True)

    # Print the temp directory for debugging
    print(f"[GyroSI Test Debug] Test environment base dir: {memories_dir}")

    # Copy tokenizers, which are needed at runtime
    shutil.copytree(MAIN_TOKENIZERS / "bert-base-uncased", public_dir / "tokenizers" / "bert-base-uncased")

    # The main, read-only meta files
    main_meta_files: Dict[str, str] = {
        "ontology": str(MAIN_META / "ontology_map.json"),
        "phenomenology": str(MAIN_META / "phenomenology_map.json"),
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
# 2. FACTORY FIXTURES: Reusable Builders for Core Objects
# =============================================================================


@pytest.fixture
def gyrosi_agent_factory(test_env: TestEnvDict) -> Generator[Callable[[str, str], GyroSI], None, None]:
    """
    Returns a factory function to create GyroSI agents within the test env.
    This centralizes all agent configuration logic.
    """
    created_agents = []

    def _make_agent(
        agent_id: str,
        private_knowledge_filename: str,
    ) -> GyroSI:
        """
        Creates a GyroSI agent with its own private knowledge file.
        Args:
            agent_id: A unique ID for the agent.
            private_knowledge_filename: The name for the private .mpk file.
        """
        prefs = test_env["preferences"]
        main_meta_files = test_env["main_meta_files"]

        config: AgentConfig = {
            "ontology_path": main_meta_files["ontology"],
            "phenomenology_map_path": main_meta_files["phenomenology"],
            "base_path": prefs["base_path"],
            "public_knowledge_path": prefs["public_knowledge"]["path"],
            "private_agents_base_path": prefs["private_knowledge"]["base_path"],
            "private_knowledge_path": str(Path(prefs["private_knowledge"]["base_path"]) / private_knowledge_filename),
            "knowledge_path": str(Path(prefs["private_knowledge"]["base_path"]) / private_knowledge_filename),
            # Removed 'store_options' to match AgentConfig TypedDict
        }
        agent = GyroSI(config, agent_id=agent_id)
        created_agents.append(agent)
        return agent

    yield _make_agent

    # Teardown: close all agents created by this factory
    for agent in created_agents:
        agent.close()


@pytest.fixture
def agent_pool_factory(test_env: TestEnvDict) -> Generator[Callable[[Set[str]], AgentPool], None, None]:
    """
    Returns a factory function to create an AgentPool within the test env.
    """
    created_pools = []

    def _make_pool(allowed_ids: Set[str] = {"user", "system", "assistant"}) -> AgentPool:
        prefs = test_env["preferences"]
        main_meta_files = test_env["main_meta_files"]
        pool = AgentPool(
            ontology_path=main_meta_files["ontology"],
            base_knowledge_path=prefs["public_knowledge"]["path"],
            preferences=cast(PreferencesConfig, prefs),
            allowed_ids=allowed_ids,
            allow_auto_create=True,
            private_agents_base_path=prefs["private_knowledge"]["base_path"],
            base_path=Path(prefs["base_path"]),
        )
        created_pools.append(pool)
        return pool

    yield _make_pool

    # Teardown: close all pools created by this factory
    for pool in created_pools:
        pool.close_all()


# =============================================================================
# 3. STANDARD FIXTURES: Convenient Wrappers Around Factories
# =============================================================================


@pytest.fixture
def gyrosi_agent(gyrosi_agent_factory: Callable[[str, str], GyroSI]) -> GyroSI:
    """Provides a single, standard GyroSI agent for simple tests."""
    return gyrosi_agent_factory("test_agent", "private_agent.mpk")


@pytest.fixture
def agent_pool(agent_pool_factory: Callable[[Set[str]], AgentPool]) -> AgentPool:
    """Provides a standard AgentPool for simple tests."""
    return agent_pool_factory({"user", "system", "assistant"})


@pytest.fixture
def multi_agent_scenario(gyrosi_agent_factory: Callable[[str, str], GyroSI], test_env: TestEnvDict) -> Dict[str, Any]:
    """
    Provides a common multi-agent scenario using the agent factory,
    demonstrating how to create multiple, distinct agents without config duplication.
    """
    user_agent = gyrosi_agent_factory("user", "user_private.mpk")
    assistant_agent = gyrosi_agent_factory("assistant", "assistant_private.mpk")

    return {
        "user": user_agent,
        "assistant": assistant_agent,
        "public_path": test_env["preferences"]["public_knowledge"]["path"],
        "memories_dir": test_env["memories_dir"],
    }


# =============================================================================
# 4. API & UTILITY FIXTURES
# =============================================================================


@pytest.fixture
def api_test_setup(
    test_env: TestEnvDict, agent_pool_factory: Callable[[Set[str]], AgentPool]
) -> Generator[Tuple[TestClient, AgentPool], None, None]:
    """
    Sets up the full environment for API testing, patches the environment,
    reloads the adapter, and yields a test client and the live agent pool.
    """
    # Point the adapter to our temporary, test-specific preferences file
    os.environ["GYROSI_PREFERENCES_PATH"] = test_env["preferences_path"]

    # The API creates its own pool on import, so we must reload it
    import importlib
    from toys.communication import external_adapter as ea

    importlib.reload(ea)

    # Ensure the triad exists for the API's pool
    ea.agent_pool.ensure_triad()

    # Ensure the pool is closed at exit
    atexit.register(ea.agent_pool.close_all)

    client = TestClient(ea.app)

    yield client, ea.agent_pool

    # Cleanup: atexit handles closing, but we can remove the env var
    del os.environ["GYROSI_PREFERENCES_PATH"]


@pytest.fixture
def test_client(api_test_setup: Tuple[TestClient, AgentPool]) -> TestClient:
    """A convenient fixture that provides just the FastAPI TestClient."""
    return api_test_setup[0]


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
