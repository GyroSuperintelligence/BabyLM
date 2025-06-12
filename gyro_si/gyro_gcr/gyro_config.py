"""
GyroCardioRespiratory Configuration Layer.

This module provides the global configuration object that controls
system-wide features and operational modes. All G-systems must respect
these settings and adapt their behavior accordingly.

The configuration is a shared singleton instance that can be imported
and referenced by any module in the system.
"""

class GyroCardioRespiratoryConfig:
    """
    Global configuration for the GyroCardioRespiratory system.

    This class controls core coordination mechanisms, optional
    cryptographic extensions, and performance switches. All settings
    have carefully chosen defaults that align with the CGM principles.
    """

    def __init__(self):
        """Initialize with default settings."""
        # Core coordination
        self.enable_bootstrap = True          # Controls bootstrap broadcast gating
        self.enable_transactions = True       # Enables atomic tensor transactions
        self.enable_recovery_beacons = True   # Enables circuit recovery signals

        # Optional cryptographic extensions
        self.enable_entropy_tracking = False  # Enables entropy_id chaining
        self.enable_crypto_evolution = False  # Enables differential crypto evolution

        # Performance switches
        self.lightweight_transactions = True  # Uses optimized transaction snapshots
        self.fast_entropy_hash = True         # Uses CRC32 instead of SHA-256 for entropy

    def __str__(self) -> str:
        """String representation showing all settings."""
        settings = []
        for key, value in vars(self).items():
            settings.append(f"{key}={value}")
        return f"GyroCardioRespiratoryConfig({', '.join(settings)})"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return self.__str__()

# Shared singleton instance to be imported by all modules
config = GyroCardioRespiratoryConfig() 