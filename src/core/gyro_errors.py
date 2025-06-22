"""
gyro_errors.py - GyroSI Error Hierarchy

Defines all custom exceptions used throughout the system, as specified in
CORE-SPEC-06. All errors inherit from a base GyroError.
"""


class GyroError(Exception):
    """Base exception for all GyroSI errors."""
    pass


# --- Core System Errors ---

class GyroTagError(GyroError):
    """TAG expression violations per CORE-SPEC-04."""
    pass


class GyroPhaseError(GyroError):
    """Navigation cycle constraint violations."""
    pass


class GyroNoResonanceError(GyroError):
    """No operator resonance occurred."""
    pass


class GyroIntegrityError(GyroError):
    """Structural integrity failures (e.g., checksum mismatch)."""
    pass


# --- Knowledge & Session Errors ---

class GyroImmutabilityError(GyroError):
    """Knowledge modification attempts on immutable packages."""
    pass


class GyroNavigationError(GyroError):
    """Navigation log operation failures."""
    pass


class GyroForkError(GyroError):
    """Knowledge forking failures."""
    pass


class GyroSessionError(GyroError):
    """Session management failures (e.g., session not found)."""
    pass


# --- Extension & I/O Errors ---

class GyroExtensionError(GyroError):
    """Generic extension operation failures."""
    pass


class GyroStorageError(GyroError):
    """File I/O and persistence failures."""
    pass


class GyroConfigError(GyroError):
    """Configuration file errors."""
    pass
