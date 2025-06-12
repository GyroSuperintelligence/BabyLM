"""
Core constants for the GyroSI system.
These constants define the fundamental parameters and configurations used across all G-systems.
"""

# System-wide constants
SYSTEM_VERSION = "0.1.0"
SYSTEM_NAME = "GyroSI Baby LM"

# G-system identifiers
G1_NAME = "GyroAlignment"
G2_NAME = "GyroInformation"
G3_NAME = "GyroInference"
G4_NAME = "GyroCooperation"
G5_NAME = "GyroPolicy"
G6_NAME = "GyroCirculation"

# Operational constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_SEQUENCE_LENGTH = 512

# Memory management
MAX_MEMORY_BUFFER = 1024 * 1024 * 1024  # 1GB
CACHE_SIZE = 1000

# Tensor operations
DEFAULT_DTYPE = "float32"
DEFAULT_DEVICE = "cpu"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# File paths
PATTERNS_DIR = "patterns"
TRANSITIONS_DIR = "transitions"
DATA_DIR = "data"
AUDIT_DIR = "audit"

# Export all constants
__all__ = [
    "SYSTEM_VERSION",
    "SYSTEM_NAME",
    "G1_NAME",
    "G2_NAME",
    "G3_NAME",
    "G4_NAME",
    "G5_NAME",
    "G6_NAME",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_SEQUENCE_LENGTH",
    "MAX_MEMORY_BUFFER",
    "CACHE_SIZE",
    "DEFAULT_DTYPE",
    "DEFAULT_DEVICE",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "PATTERNS_DIR",
    "TRANSITIONS_DIR",
    "DATA_DIR",
    "AUDIT_DIR",
]
