"""
gyro_errors.py - GyroSI Error Hierarchy

Defines all custom exceptions used throughout the system.
"""


class GyroError(Exception):
    """Base exception for all GyroSI errors."""
    pass


class GyroTagError(GyroError):
    """TAG expression violations per CORE-SPEC-04."""
    pass


class GyroPhaseError(GyroError):
    """Navigation cycle constraint violations."""
    pass


class GyroNoResonanceError(GyroError):
    """No operator resonance occurred."""
    pass


class GyroImmutabilityError(GyroError):
    """Knowledge modification attempts on immutable packages."""
    pass


class GyroIntegrityError(GyroError):
    """Structural integrity failures."""
    pass


class GyroExtensionError(GyroError):
    """Extension operation failures."""
    pass


class GyroNavigationError(GyroError):
    """Navigation log operation failures."""
    pass


class GyroForkError(GyroError):
    """Knowledge forking failures."""
    pass


class GyroSessionError(GyroError):
    """Session management failures."""
    pass