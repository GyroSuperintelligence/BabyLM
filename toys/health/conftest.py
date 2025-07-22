import os
import sys
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, cast
import pytest
from baby.contracts import AgentConfig
from baby.intelligence import AgentPool, GyroSI
from baby.policies import OrbitStore, OverlayView, ReadOnlyView

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
"""
Shared pytest fixtures and configuration for GyroSI test suite.
"""

# Define these paths as needed for your test environment
MEM_META = Path("memories/public/meta")
MEM_TOKENIZERS = Path("memories/public/tokenizers")
ONTOLOGY_PATH = MEM_META / "ontology_map.json"
PHENOMENOLOGY_PATH = MEM_META / "phenomenology_map.json"
THETA_PATH = MEM_META / "theta.npy"
EPISTEMOLOGY_PATH = MEM_META / "epistemology.npy"
TOKENIZER_DIR = MEM_TOKENIZERS / "bert-base-uncased"
TOKENIZER_JSON = TOKENIZER_DIR / "tokenizer.json"
TOKENIZER_VOCAB = TOKENIZER_DIR / "vocab.txt"

# Use the main stateless data files for all tests
MAIN_MEMORIES_META = Path(__file__).parent.parent.parent / "memories" / "public" / "meta"

# Base temp directory for all test-generated files
BASE_TEMP_DIR = Path(__file__).parent / "memories"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    """Setup and teardown test environment."""
    BASE_TEMP_DIR.mkdir(exist_ok=True)
    yield
    # For CI or production, uncomment the following line to ensure a clean slate after tests:
    shutil.rmtree(BASE_TEMP_DIR, ignore_errors=True)


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a unique temporary directory for each test. Safe for parallel test runs."""
    test_dir = BASE_TEMP_DIR / f"test_{os.getpid()}_{id(object())}"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield str(test_dir)
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def real_ontology() -> tuple[str, str, str]:
    """
    Fixture returning paths to the real ontology and phenomenology files, for integration tests.
    """
    return str(ONTOLOGY_PATH), str(PHENOMENOLOGY_PATH), str(EPISTEMOLOGY_PATH)


@pytest.fixture
def orbit_store(temp_dir: str) -> Generator[OrbitStore, None, None]:
    """Create an OrbitStore instance (empty) in a test directory."""
    store_path = os.path.join(temp_dir, "knowledge.pkl.gz")
    store = OrbitStore(store_path)
    yield store
    store.close()


@pytest.fixture
def overlay_store(temp_dir: str) -> Generator[OverlayView, None, None]:
    """
    Provide an OverlayView composed of a read-only public store and a writable private store.
    Public store points to a main (or shared) store, private store is in temp_dir.
    """
    # Public store is a (possibly empty) shared file, treated as read-only
    public_store_path = os.path.join(temp_dir, "public_knowledge.pkl.gz")
    os.makedirs(os.path.dirname(public_store_path), exist_ok=True)
    # Ensure the file exists and has one entry
    public_store = OrbitStore(public_store_path)
    public_store.put((0, 0), {"phenotype": "public", "confidence": 0.9, "context_signature": (0, 0)})
    public_store.commit()
    public_store.close()
    # Wrap as read-only
    public_readonly = ReadOnlyView(OrbitStore(public_store_path))
    # Private store for this test
    private_store_path = os.path.join(temp_dir, "private_knowledge.pkl.gz")
    private_store = OrbitStore(private_store_path)
    overlay = OverlayView(public_readonly, private_store)
    yield overlay
    overlay.close()


@pytest.fixture
def real_orbit_store(temp_dir: str) -> Generator[OrbitStore, None, None]:
    """
    Provides a real OrbitStore using the main ontology and phenomenology, but stores all data in temp_dir.
    Use this fixture in tests that want to test the real OrbitStore logic without polluting the main memories folder.
    """
    store_path = os.path.join(temp_dir, "real_knowledge.pkl.gz")
    store = OrbitStore(store_path)
    yield store
    store.close()


@pytest.fixture
def agent_config(temp_dir: str) -> AgentConfig:
    """Create a test agent configuration, always using the real ontology."""
    return {
        "ontology_path": str(ONTOLOGY_PATH),
        "knowledge_path": os.path.join(temp_dir, "knowledge.pkl.gz"),
        "enable_phenomenology_storage": False,
    }


@pytest.fixture
def gyrosi_agent(agent_config: AgentConfig) -> Generator[GyroSI, None, None]:
    """Create a GyroSI agent instance."""
    agent = GyroSI(config=agent_config)
    yield agent
    agent.close()


@pytest.fixture
def agent_pool(tmp_path) -> Generator[AgentPool, None, None]:
    """Create an agent pool using the real ontology, with all agent data isolated to temp_dir."""
    public_knowledge = str(tmp_path / "public_knowledge.pkl.gz")
    OrbitStore(public_knowledge).close()
    pool = AgentPool(str(ONTOLOGY_PATH), public_knowledge)
    yield pool
    pool.close_all()


@pytest.fixture
def preferences_config() -> dict:
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
        "enable_profiling": False,
        "batch_size": 100,  # changed key
        "cache_size_mb": 10,
    }


@pytest.fixture
def sample_phenotype_entry() -> dict[str, Any]:
    """Create a sample phenotype entry for testing."""
    return {
        "phenotype": "A",
        "exon_mask": 0b10101010,
        "confidence": 0.75,
        "context_signature": (100, 42),
        "usage_count": 10,
        "created_at": 1234567890.0,
        "last_updated": 1234567890.0,
    }


class TimeController:
    """Context manager for timing code blocks in tests."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0

    def __enter__(self) -> "TimeController":
        import time

        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        import time

        self.elapsed = time.perf_counter() - self.start


@pytest.fixture
def mock_time(monkeypatch: pytest.MonkeyPatch) -> GyroSI:
    """Mock time.time() for deterministic tests."""
    current_time = [1234567890.0]

    def mock_time_func() -> float:
        return current_time[0]

    def advance_time(seconds: float) -> None:
        current_time[0] += seconds

    monkeypatch.setattr("time.time", mock_time_func)

    class _TimeController(TimeController):
        def advance(self, seconds: float) -> None:
            advance_time(seconds)

        @property
        def current(self) -> float:
            return current_time[0]

    return cast(GyroSI, _TimeController())


@pytest.fixture
def generate_test_introns() -> Callable[[int], List[int]]:
    import random

    def _gen(count: int, seed: int = 42) -> List[int]:
        random.seed(seed)
        return [random.randint(0, 255) for _ in range(count)]

    return _gen


def generate_test_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def assert_phenotype_entry_valid(entry: Dict[str, Any]) -> None:
    required_fields = ["phenotype", "exon_mask", "confidence", "context_signature", "usage_count"]
    for field in required_fields:
        assert field in entry, f"Missing required field: {field}"
    assert isinstance(entry["phenotype"], str)
    assert isinstance(entry["exon_mask"], int)
    assert 0 <= entry["exon_mask"] <= 255
    assert 0 <= entry["confidence"] <= 1.0
    # Accept both tuple and list of length 2 for context_signature
    assert isinstance(entry["context_signature"], (tuple, list)) and len(entry["context_signature"]) == 2


def assert_ontology_valid(ontology_data: Dict[str, Any]) -> None:
    assert ontology_data["endogenous_modulus"] == 788_986
    assert ontology_data["ontology_diameter"] == 6
    assert "ontology_map" in ontology_data
    assert "schema_version" in ontology_data


class Timer:
    """Context manager for timing code blocks in tests."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        import time

        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        import time

        self.elapsed = time.perf_counter() - self.start


@pytest.fixture
def theta_path() -> Path:
    """Fixture for the path to theta.npy."""
    return THETA_PATH


@pytest.fixture
def tokenizer_dir() -> Path:
    """Fixture for the path to the tokenizer directory."""
    return TOKENIZER_DIR


@pytest.fixture
def tokenizer_json() -> Path:
    """Fixture for the path to the tokenizer.json file."""
    return TOKENIZER_JSON


@pytest.fixture
def tokenizer_vocab() -> Path:
    """Fixture for the path to the vocab.txt file."""
    return TOKENIZER_VOCAB
