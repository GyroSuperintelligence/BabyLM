"""
GyroSI Baby Language Model - Core Implementation

A physics-grounded architecture for superintelligence through recursive structural alignment.
Based on the Common Governance Model (CGM) and gyrogroup algebra.
"""

from baby.contracts import (
    AgentConfig,
    CycleHookFunction,
    PhenotypeEntry,
    PreferencesConfig,
    ValidationReport,
)
from baby.information import discover_and_save_ontology
from baby.policies import (
    CanonicalView,
    OverlayView,
    ReadOnlyView,
    apply_global_confidence_decay,
    export_knowledge_statistics,
    merge_phenotype_maps,
    validate_ontology_integrity,
    prune_and_compact_store,
)

__version__ = "0.9.6"
__all__ = [
    "PhenotypeEntry",
    "AgentConfig",
    "PreferencesConfig",
    "CycleHookFunction",
    "ValidationReport",
    "CanonicalView",
    "OverlayView",
    "ReadOnlyView",
    "merge_phenotype_maps",
    "apply_global_confidence_decay",
    "export_knowledge_statistics",
    "validate_ontology_integrity",
    "discover_and_save_ontology",
    "prune_and_compact_store",
]
