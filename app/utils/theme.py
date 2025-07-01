"""
theme.py - Theme configuration for GyroSI Baby LM

Defines color palette, dimensions, and styling constants.
"""

import flet as ft
from typing import Dict, Any

# Dark Apple WatchOS palette
COLORS = {
    "background": "#000000",
    "surface": "#1C1C1E",
    "surface_variant": "#2C2C2E",
    "primary": "#0A84FF",
    "secondary": "#5E5CE6",
    "success": "#30D158",
    "warning": "#FF9F0A",
    "error": "#FF453A",
    "text_primary": "#FFFFFF",
    "text_secondary": "#8E8E93",
    "border": "#38383A",
    "divider": "#38383A",
    "overlay": "#000000CC",
}

# Dimensions
DIMENSIONS = {
    "border_radius": 12,
    "button_radius": 8,
    "input_radius": 20,
    "padding_small": 5,
    "padding_medium": 10,
    "padding_large": 20,
    "icon_size_small": 16,
    "icon_size_medium": 20,
    "icon_size_large": 24,
    "font_size_small": 12,
    "font_size_medium": 14,
    "font_size_large": 16,
    "font_size_xlarge": 20,
    "spacing_small": 5,
    "spacing_medium": 10,
    "spacing_large": 20,
}


def apply_theme(page: ft.Page) -> None:
    """Apply theme settings to the Flet page."""
    page.bgcolor = COLORS["background"]
    page.padding = 0
    page.window_bgcolor = COLORS["background"]
    page.theme = ft.Theme(
        color_scheme_seed=COLORS["primary"],
        color_scheme=ft.ColorScheme(
            primary=COLORS["primary"],
            secondary=COLORS["secondary"],
            error=COLORS["error"],
            surface=COLORS["surface"],
            background=COLORS["background"],
        ),
    )

    # Set platform-specific properties
    page.fonts = {
        "SF Pro": "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
    }

    # Override default text styles
    page.theme.text_theme = ft.TextTheme(
        body_large=ft.TextStyle(
            size=DIMENSIONS["font_size_medium"], color=COLORS["text_primary"], font_family="SF Pro"
        ),
        body_medium=ft.TextStyle(
            size=DIMENSIONS["font_size_small"], color=COLORS["text_secondary"], font_family="SF Pro"
        ),
        label_large=ft.TextStyle(
            size=DIMENSIONS["font_size_medium"],
            weight=ft.FontWeight.W_500,
            color=COLORS["text_primary"],
            font_family="SF Pro",
        ),
    )


def get_card_style() -> Dict[str, Any]:
    """Return standard card container style."""
    return {
        "bgcolor": COLORS["surface"],
        "border_radius": DIMENSIONS["border_radius"],
        "padding": DIMENSIONS["padding_medium"],
    }


def get_button_style(button_type: str = "primary") -> ft.ButtonStyle:
    """
    Return button style based on type.

    Args:
        button_type: "primary", "secondary", "danger", or "text"
    """
    styles = {
        "primary": ft.ButtonStyle(
            color=COLORS["text_primary"],
            bgcolor=COLORS["primary"],
            shape=ft.RoundedRectangleBorder(radius=DIMENSIONS["button_radius"]),
            padding=ft.padding.symmetric(horizontal=16, vertical=10),
        ),
        "secondary": ft.ButtonStyle(
            color=COLORS["text_primary"],
            bgcolor=COLORS["surface_variant"],
            shape=ft.RoundedRectangleBorder(radius=DIMENSIONS["button_radius"]),
            padding=ft.padding.symmetric(horizontal=16, vertical=10),
        ),
        "danger": ft.ButtonStyle(
            color=COLORS["text_primary"],
            bgcolor=COLORS["error"],
            shape=ft.RoundedRectangleBorder(radius=DIMENSIONS["button_radius"]),
            padding=ft.padding.symmetric(horizontal=16, vertical=10),
        ),
        "text": ft.ButtonStyle(
            color=COLORS["primary"],
            bgcolor=ft.colors.TRANSPARENT,
            shape=ft.RoundedRectangleBorder(radius=DIMENSIONS["button_radius"]),
            padding=ft.padding.symmetric(horizontal=10, vertical=8),
        ),
    }

    return styles.get(button_type, styles["primary"])
