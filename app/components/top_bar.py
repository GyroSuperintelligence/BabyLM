"""
top_bar.py - Top navigation bar for GyroSI Baby LM

App branding, navigation, and quick access to settings/dev tools.
"""

import flet as ft
from typing import Callable, Optional

from state import AppState


class TopBar(ft.UserControl):
    """
    Top navigation bar with app branding and navigation controls.

    Layout:
    - Left: Baby emoji + Title
    - Right: Settings icon, Dev icon
    """

    def __init__(
        self,
        state: AppState,
        on_settings_click: Callable[[], None],
        on_dev_click: Callable[[], None],
        page: ft.Page,
    ):
        super().__init__()
        self.state = state
        self.on_settings_click = on_settings_click
        self.on_dev_click = on_dev_click
        self.page = page

    def build(self):
        # App branding
        app_brand = ft.Row(
            controls=[
                ft.Text("👶", size=24),  # Baby emoji
                ft.Text("GyroSI Baby LM", size=18, weight=ft.FontWeight.W_600, color="#FFFFFF"),
            ],
            spacing=10,
        )

        # Settings button
        settings_button = ft.IconButton(
            icon=ft.icons.SETTINGS,
            icon_color="#8E8E93",
            icon_size=20,
            on_click=lambda _: self.on_settings_click(),
            tooltip="Settings",
        )

        # Dev button (only shown if enabled in settings)
        dev_button = ft.IconButton(
            icon=ft.icons.CODE,
            icon_color="#8E8E93",
            icon_size=20,
            on_click=lambda _: self.on_dev_click(),
            tooltip="Developer Tools",
            visible=self.state.settings.get("show_dev_info", False),
        )

        # Navigation buttons container
        nav_buttons = ft.Row(controls=[settings_button, dev_button], spacing=5)

        return ft.Container(
            content=ft.Row(
                controls=[app_brand, nav_buttons], alignment=ft.MainAxisAlignment.SPACE_BETWEEN
            ),
            height=56,
            padding=ft.padding.symmetric(horizontal=20, vertical=10),
            bgcolor="#1C1C1E",
            border=ft.border.only(bottom=ft.BorderSide(1, "#38383A")),
        )

    def update(self):
        """Update visibility of dev button based on settings."""
        if hasattr(self, "controls") and self.controls:
            row = getattr(self.controls[0], "content", None)
            if row and hasattr(row, "controls") and len(row.controls) > 1:
                nav_row = row.controls[1]
                if hasattr(nav_row, "controls") and len(nav_row.controls) > 1:
                    nav_row.controls[1].visible = self.state.settings.get("show_dev_info", False)
        super().update()
