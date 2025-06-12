"""
Configuration and feature toggles for GyroSI system.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class GyroConfig:
    """Configuration settings for GyroSI."""
    # Feature toggles
    enable_entropy: bool = True
    enable_crypto: bool = True
    enable_gpu: bool = False
    enable_flet_ui: bool = False

    # System settings
    max_retries: int = 3
    timeout: float = 30.0
    debug_mode: bool = False

    # Custom settings
    custom_settings: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_settings is None:
            self.custom_settings = {}

# Global configuration instance
config = GyroConfig()

def update_config(**kwargs) -> None:
    """Update configuration settings."""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            config.custom_settings[key] = value 