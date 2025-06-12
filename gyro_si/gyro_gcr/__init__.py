"""
GyroCardioRespiratory Integration Layer.

This package provides the coordination mechanisms that ensure
system-wide coherence, fault tolerance, and recovery capabilities.
It implements the bootstrap protocol, tensor transactions, circuit
breakers, and optional cryptographic extensions.
"""

from gyro_gcr.gyro_config import config, GyroCardioRespiratoryConfig

__all__ = ['config', 'GyroCardioRespiratoryConfig'] 