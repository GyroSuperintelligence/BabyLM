"""
GyroSI Baby Language Model - Core Implementation

A physics-grounded architecture for superintelligence through recursive structural alignment.
Based on the Common Governance Model (CGM) and gyrogroup algebra.
"""

from .governance import (
    GENE_Mic_S,
    GENE_Mac_S,
    FG_MASK,
    BG_MASK,
    FULL_MASK,
    INTRON_BROADCAST_MASKS,
    apply_gyration_and_transform,
    transcribe_byte,
    coadd,
    batch_introns_coadd_ordered,
)

from .information import (
    InformationEngine,
    PickleStore,
    MultiAgentPhenotypeStore,
    CanonicalizingStore,
    discover_and_save_manifold,
    build_canonical_map,
)

from .inference import (
    EndogenousInferenceOperator,
)

from .intelligence import (
    IntelligenceEngine,
    GyroSI,
    AgentPool,
    orchestrate_turn,
)

from .types import (
    PhenotypeStore,
    PhenotypeEntry,
    ManifoldData,
    AgentConfig,
    PreferencesConfig,
)

from .maintenance import (
    merge_phenotype_maps,
    apply_global_confidence_decay,
    export_knowledge_statistics,
    validate_manifold_integrity,
)

__version__ = "1.0.0"
__all__ = [
    # Core constants and functions
    "GENE_Mic_S",
    "GENE_Mac_S", 
    "apply_gyration_and_transform",
    "transcribe_byte",
    "coadd",
    "batch_introns_coadd_ordered",
    
    # Engines
    "InformationEngine",
    "EndogenousInferenceOperator", 
    "IntelligenceEngine",
    "GyroSI",
    
    # Storage implementations
    "PickleStore",
    "MultiAgentPhenotypeStore",
    "CanonicalizingStore",
    
    # Utilities
    "discover_and_save_manifold",
    "build_canonical_map",
    "AgentPool",
    "orchestrate_turn",
    
    # Maintenance
    "merge_phenotype_maps",
    "apply_global_confidence_decay",
    "export_knowledge_statistics",
    "validate_manifold_integrity",
    
    # Types
    "PhenotypeStore",
    "PhenotypeEntry",
    "ManifoldData",
    "AgentConfig",
    "PreferencesConfig",
]