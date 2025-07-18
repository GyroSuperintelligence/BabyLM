"""
GyroSI Baby Language Model - Core Implementation

A physics-grounded architecture for superintelligence through recursive structural alignment.
Based on the Common Governance Model (CGM) and gyrogroup algebra.
"""

from baby.contracts import (
    PhenotypeEntry,
    ManifoldData,
    AgentConfig,
    PreferencesConfig,
    CycleHookFunction,
    ValidationReport,
)
from baby.policies import (
    CanonicalView,
    OverlayView,
    ReadOnlyView,
    merge_phenotype_maps,
    apply_global_confidence_decay,
    export_knowledge_statistics,
    validate_ontology_integrity,
)
from baby.information import discover_and_save_ontology

__version__ = "0.9.6"
__all__ = [
    "PhenotypeEntry",
    "ManifoldData",
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
]
