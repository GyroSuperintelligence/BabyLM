"""
config.py - Configuration settings for GyroSI Baby LM

Defines constants, paths, and default settings.
"""

import os
from pathlib import Path

# Application info
APP_NAME = "GyroSI Baby LM"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "A fully operational language model that learns and grows on its own."

# Directory paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = os.environ.get("GYRO_DATA_DIR", str(BASE_DIR / "s2_information"))
TEMP_DIR = os.environ.get("GYRO_TEMP_DIR", str(BASE_DIR / "temp"))

# Default settings
DEFAULT_SETTINGS = {
    "encryption_enabled": True,
    "auto_save": True,
    "max_recent_messages": 250,
    "show_dev_info": False,
    "theme": "dark",
}

# Memory limits
MAX_MESSAGES_IN_MEMORY = 1000
MAX_PATTERN_SIZE = 16
MAX_GENERATE_LENGTH = 1000

# File extensions
ALLOWED_EXTENSIONS = [".txt", ".md", ".json", ".csv", ".py", ".js", ".html", ".css"]

# UI constants
PAGE_TITLE = "GyroSI Baby LM"
PAGE_HEIGHT = 800
PAGE_WIDTH = 1200
PAGE_MIN_WIDTH = 800
PAGE_MIN_HEIGHT = 600
THREAD_LIST_WIDTH = 250
MOBILE_BREAKPOINT = 600

# Feature flags
ENABLE_ANIMATIONS = True
ENABLE_FILE_PROCESSING = True
DEBUG_MODE = os.environ.get("GYRO_DEBUG", "0") == "1"


def get_data_path() -> Path:
    """
    Get the data directory path, creating it if it doesn't exist.

    Returns:
        Path object for the data directory
    """
    path = Path(DATA_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_temp_path() -> Path:
    """
    Get the temporary directory path, creating it if it doesn't exist.

    Returns:
        Path object for the temporary directory
    """
    path = Path(TEMP_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path
