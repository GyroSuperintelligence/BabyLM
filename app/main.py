"""
main.py - Entry point for GyroSI Baby LM

Initializes the Flet app and handles command line arguments.
"""

import flet as ft
import argparse
import sys
import os
from pathlib import Path
from typing import cast

# Ensure parent directory is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.app import main as app_main_function  # Import the actual function
from config import PAGE_TITLE, PAGE_WIDTH, PAGE_HEIGHT, DATA_DIR, DEBUG_MODE


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GyroSI Baby LM - Intelligent Language Model")

    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        help="Path to data directory (default: ./s2_information)",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    parser.add_argument(
        "--port", type=int, default=0, help="Port to run web server on (0 for desktop mode)"
    )

    return parser.parse_args()


def desktop_wrapper(page: ft.Page):
    """Wrapper for desktop mode to configure window properties."""
    # Set page properties directly instead of through window attribute
    page.title = PAGE_TITLE
    page.window_width = PAGE_WIDTH
    page.window_height = PAGE_HEIGHT
    page.window_frameless = False
    page.window_minimizable = True
    page.window_maximizable = True
    page.window_resizable = True

    # Call the actual main function
    app_main_function(page)


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Set environment variables from arguments
    if args.data_dir:
        os.environ["GYRO_DATA_DIR"] = args.data_dir

    if args.debug:
        os.environ["GYRO_DEBUG"] = "1"

    # Determine app mode
    if args.port > 0:
        # Web mode
        ft.app(target=app_main_function, port=args.port, view=ft.AppView.WEB_BROWSER)
    else:
        # Desktop mode - use wrapper to set window properties
        ft.app(target=desktop_wrapper, assets_dir="assets")
