"""
Shared contracts (protocols and type definitions) for the GyroSI S4 system.
"""

from typing import TypedDict, Protocol, Dict, Any, Optional


class PhenotypeEntry(TypedDict):
    """
    Minimal phenotype record.
      mask  : 8-bit Monodromic-Fold residue  (0-255)
      conf  : epistemic confidence           (0.0-1.0)  – monotone ↑ / decay ↓
      key   : composite key (state_idx, token_id)
    Everything else is derivable on-the-fly.
    """

    mask: int  # uint8   (exon_mask)
    conf: float  # float32
    key: tuple[int, int]  # composite key


class PreferencesConfig(TypedDict, total=False):
    """Preferences and settings configuration."""

    # Storage preferences
    storage_backend: str  # "binary_struct"
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
    epistemology_path: Optional[str]
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
        last_token_byte: int,
        token_id: Optional[int] = None,
        state_index: Optional[int] = None,
    ) -> None:
        """Post-cycle hook callback with token-level information."""
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
