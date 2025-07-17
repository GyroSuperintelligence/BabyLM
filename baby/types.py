"""
Shared type definitions for the GyroSI system.
"""

from typing import Protocol, Optional, Dict, Any, Tuple, TypedDict


class PhenotypeEntry(TypedDict, total=False):
    """Structure of a phenotype entry in the knowledge store."""

    phenotype: str
    memory_mask: int
    confidence: float
    context_signature: Tuple[int, int]
    semantic_address: int
    usage_count: int
    age_counter: int
    created_at: float
    last_updated: float


class ManifoldData(TypedDict):
    """Structure of the discovered manifold data."""

    schema_version: str
    ontology_map: Dict[int, int]
    endogenous_modulus: int
    manifold_diameter: int
    total_states: int
    build_timestamp: float


class AgentConfig(TypedDict, total=False):
    """Configuration for GyroSI agents."""

    manifold_path: str
    knowledge_path: Optional[str]
    public_knowledge_path: Optional[str]
    private_knowledge_path: Optional[str]
    agent_metadata: Optional[Dict[str, Any]]
    max_memory_mb: Optional[int]
    enable_canonical_storage: Optional[bool]


class PreferencesConfig(TypedDict, total=False):
    """Preferences and settings configuration."""

    # Storage preferences
    storage_backend: str  # "pickle", "sqlite", "rocksdb"
    compression_level: int  # 1-9 for gzip
    max_file_size_mb: int

    # Maintenance preferences
    enable_auto_decay: bool
    decay_interval_hours: float
    decay_factor: float
    confidence_threshold: float

    # Agent pool preferences
    max_agents_in_memory: int
    agent_eviction_policy: str  # "lru", "lfu", "ttl"
    agent_ttl_minutes: int

    # Security preferences
    encryption_enabled: bool
    encryption_key: Optional[str]

    # Performance preferences
    enable_profiling: bool
    batch_size: int
    cache_size_mb: int


class ValidationReport(TypedDict):
    """Report structure for system validation."""

    total_entries: int
    average_confidence: float
    store_type: str
    modified_entries: Optional[int]


class CycleHookFunction(Protocol):
    """Protocol for post-cycle hook functions."""

    def __call__(
        self,
        engine: Any,  # Would be IntelligenceEngine but avoiding circular import
        phenotype_entry: PhenotypeEntry,
        last_intron: int,
    ) -> None:
        """Post-cycle hook callback."""
        ...


class MaintenanceReport(TypedDict):
    """Report from maintenance operations."""

    operation: str
    success: bool
    entries_processed: int
    entries_modified: int
    elapsed_seconds: float
