"""
GyroSI Baby LM - A Gyroscopic Superintelligence Language Model

GyroSI Baby LM is an open-source language model that learns without reinforcement,
rewards, traditional neural network parameters, or gradient descent. Instead, it
leverages quantum physics-inspired tensor operations to achieve intrinsic
Alignment-Based recursive intelligence.

The system follows the Common Governance Model (CGM), organizing components into
four layers of responsibility:

- S1 Governance: Pure tensor operations (governance.py)
- S2 Information: Stream processing (information.py)
- S3 Inference: Pattern recognition (inference.py)
- S4 Intelligence: Orchestration and decision-making (intelligence.py)
"""

from baby.governance import gene_stateless, gene_com, gene_nest, gene_add
from baby.inference import InferenceEngine
from baby.information import InformationEngine
from baby.intelligence import IntelligenceEngine, initialize_intelligence_engine

__version__ = "0.9.5"
__all__ = [
    "InferenceEngine",
    "InformationEngine",
    "IntelligenceEngine",
    "initialize_intelligence_engine",
    "gene_stateless",
    "gene_com",
    "gene_nest",
    "gene_add",
]
