"""
Extensions package for GyroSI.

This package contains all system and application extensions that operate
through the canonical information interfaces defined in the s1_governance.
"""

# Import all extensions for easy access
from .base import GyroExtension

# System extensions
# from .ext_storage_manager import ext_StorageManager
from .ext_fork_manager import ext_ForkManager
from .ext_state_helper import ext_StateHelper
from .ext_error_handler import ext_ErrorHandler
from .ext_cryptographer import ext_Cryptographer

# Application extensions
# from .ext_compression import ext_Compression
from .ext_phase_controller import ext_PhaseController
from .ext_resonance_processor import ext_ResonanceProcessor

# TODO: Add API Gateway back in
# from .ext_api_gateway import ext_APIGateway

__all__ = [
    # Base class
    "GyroExtension",
    # System extensions
    # "ext_StorageManager",
    "ext_ForkManager",
    "ext_StateHelper",
    "ext_ErrorHandler",
    "ext_Cryptographer",
    # Application extensions
    # "ext_Compression",
    "ext_PhaseController",
    "ext_ResonanceProcessor",
    # "ext_APIGateway",
]
