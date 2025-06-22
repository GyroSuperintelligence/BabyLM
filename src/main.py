# src/main.py
import flet as ft
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from frontend.gyro_app import GyroApp

# Unused variables for future debug: (none listed)

async def main(page: ft.Page):
    """Main entry point for GyroSI Baby ML"""
    app = GyroApp()
    await app.main(page)


if __name__ == "__main__":
    ft.app(target=main, assets_dir="assets", view=ft.AppView.FLET_APP_WEB, port=8550)
