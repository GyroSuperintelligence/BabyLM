"""
Error taxonomy for GyroSI system.
All errors inherit from GyroError to maintain consistent error handling.
"""

class GyroError(Exception):
    """Base class for all GyroSI errors."""
    pass

class GyroAlignmentError(GyroError):
    """Errors in G1 alignment operations."""
    pass

class GyroInformationError(GyroError):
    """Errors in G2 information processing."""
    pass

class GyroInferenceError(GyroError):
    """Errors in G3 inference operations."""
    pass

class GyroIntelligenceError(GyroError):
    """Errors in G4/G5 intelligence operations."""
    pass

class GyroCirculationError(GyroError):
    """Errors in G6 circulation operations."""
    pass

class GyroMemoryError(GyroError):
    """Errors in memory operations."""
    pass

class GyroConfigError(GyroError):
    """Errors in configuration."""
    pass

class GyroCommunicationError(GyroError):
    """Errors in inter-system communication."""
    pass 