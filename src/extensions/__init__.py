"""
Extensions package for GyroSI.

This package contains all system and application extensions that operate
through the canonical memory interfaces defined in the core.
"""

# Import all extensions for easy access
from .base import GyroExtension

# System extensions
from .ext_storage_manager import ext_StorageManager
from .ext_fork_manager import ext_ForkManager
from .ext_state_helper import ext_StateHelper
from .ext_event_classifier import ext_EventClassifier
from .ext_error_handler import ext_ErrorHandler
from .ext_navigation_helper import ext_NavigationHelper
from .ext_api_gateway import ext_APIGateway
from .ext_system_monitor import ext_SystemMonitor
from .ext_performance_tracker import ext_PerformanceTracker
from .ext_cryptographer import ext_Cryptographer
from .ext_language_egress import ext_LanguageEgress

# Application extensions
from .ext_multi_resolution import ext_MultiResolution
from .ext_bloom_filter import ext_BloomFilter
from .ext_spin_piv import ext_SpinPIV
from .ext_coset_knowledge import ext_CosetKnowledge

__all__ = [
    # Base class
    "GyroExtension",
    # System extensions
    "ext_StorageManager",
    "ext_ForkManager",
    "ext_StateHelper",
    "ext_EventClassifier",
    "ext_ErrorHandler",
    "ext_NavigationHelper",
    "ext_APIGateway",
    "ext_SystemMonitor",
    "ext_PerformanceTracker",
    "ext_Cryptographer",
    "ext_LanguageEgress",
    # Application extensions
    "ext_MultiResolution",
    "ext_BloomFilter",
    "ext_SpinPIV",
    "ext_CosetKnowledge",
]
