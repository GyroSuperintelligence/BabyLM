"""
Shared contracts (protocols and type definitions) for the GyroSI S4 system.
"""

from typing import Any, Dict, Optional, Protocol, Tuple, TypedDict


class GovernanceSignature(TypedDict):
    neutral: int  # 0‑6
    li: int  # 0‑2
    fg: int  # 0‑2
    bg: int  # 0‑2
    dyn: int  # 0‑6


class PhenotypeEntry(TypedDict):
    """
    Structure of a phenotype entry in the knowledge store.

    - context_signature MAY be canonical; if canonicalisation is applied (e.g.,
      via CanonicalView), the original physical context is stored in
      _original_context.
    - exon_mask is immutable under decay (decay only affects confidence, not
      exon_mask).
    """

    phenotype: str
    confidence: float
    exon_mask: int
    usage_count: int
    last_updated: float
    created_at: float
    governance_signature: GovernanceSignature
    context_signature: Tuple[int, int]
    _original_context: Optional[Tuple[int, int]]


class PreferencesConfig(TypedDict, total=False):
    """Preferences and settings configuration.

    write_batch_size: Number of store writes to buffer before flushing to disk.
    """

    # Storage preferences
    storage_backend: str  # "msgpack"
    compression_level: int
    max_file_size_mb: int

    # Maintenance preferences
    enable_auto_decay: bool
    decay_interval_hours: float
    decay_factor: float
    confidence_threshold: float

    # Pruning preferences
    pruning: Dict[str, Any]  # Contains confidence_threshold, decay_factor, decay_interval_hours, enable_auto_decay

    # Agent pool preferences
    max_agents_in_memory: int
    agent_eviction_policy: str  # "lru", "lfu", "ttl"
    agent_ttl_minutes: int

    # Security preferences

    # Performance preferences
    enable_profiling: bool
    write_batch_size: int
    cache_size_mb: int


class AgentConfig(TypedDict, total=False):
    ontology_path: str
    knowledge_path: Optional[str]
    public_knowledge_path: Optional[str]
    private_knowledge_path: Optional[str]
    enable_phenomenology_storage: Optional[bool]
    phenomenology_map_path: Optional[str]
    learn_batch_size: Optional[int]
    agent_metadata: Optional[Dict[str, Any]]
    private_agents_base_path: Optional[str]  # for test path overrides
    base_path: Optional[str]  # for test path overrides
    preferences: Optional[PreferencesConfig]  # preferences for the agent


class ValidationReport(TypedDict):
    """Report structure for system validation."""

    total_entries: int
    average_confidence: float
    store_type: str
    modified_entries: int


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


__all__ = [
    "PhenotypeEntry",
    "AgentConfig",
    "PreferencesConfig",
    "CycleHookFunction",
    "ValidationReport",
    "MaintenanceReport",
]
