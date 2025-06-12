"""Core constants for GyroSI system."""

# Core constants from CGM
ALPHA = 1.0  # Primary scaling factor
BETA = 2.0   # Secondary scaling factor
GAMMA = 3.0  # Tertiary scaling factor

# Mass constant for quantization
M_P = 1 / (2 * (2 * 3.141592653589793) ** 0.5)  # 1/(2√(2π))

# Phase space constants
HALF_HORIZON = 3.141592653589793  # π
HORIZON_CYCLES = 2 * 3.141592653589793  # 2π
BIN_COUNT = 8  # Number of quantization bins
PHASE_HORIZON_2PI = 2 * 3.141592653589793  # 2π

# Tensor value constraints
VALID_TENSOR_VALUES = {-1, 0, 1}

# Memory type constants
MEMORY_TYPES = {
    "GENETIC": "g1_genetic",
    "EPIGENETIC": "g2_epigenetic",
    "STRUCTURAL": "g3_structural",
    "SOMATIC": "g4_somatic",
    "IMMUNITY": "g5_immunity",
}

# System level constants
SYSTEM_LEVELS = {
    "G1": "GyroAlignment",
    "G2": "GyroInformation",
    "G3": "GyroInference",
    "G4": "GyroCooperation",
    "G5": "GyroPolicy",
}

# Stage constants
STAGES = {
    "CS": "Control System",
    "UNA": "Unitary Normalization",
    "ONA": "Orthogonal Normalization",
    "BU_IN": "Bottom-Up Integration",
    "BU_EN": "Bottom-Up Generation",
} 