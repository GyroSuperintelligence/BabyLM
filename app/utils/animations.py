"""
animations.py - Animation utilities for GyroSI Baby LM

Defines animation presets and helpers for smooth UI transitions.
"""

import flet as ft
from typing import Optional, Dict, Any

# Animation durations in milliseconds
DURATIONS = {"fast": 150, "normal": 250, "slow": 350}

# Animation curves
CURVES = {
    "standard": ft.AnimationCurve.EASE_IN_OUT,
    "decelerate": ft.AnimationCurve.EASE_OUT,
    "accelerate": ft.AnimationCurve.EASE_IN,
    "sharp": ft.AnimationCurve.EASE,
}


def get_animation(
    duration_preset: str = "normal",
    curve_preset: str = "standard",
    custom_duration: Optional[int] = None,
) -> ft.Animation:
    """
    Create an animation with preset or custom duration and curve.

    Args:
        duration_preset: "fast", "normal", or "slow"
        curve_preset: "standard", "decelerate", "accelerate", or "sharp"
        custom_duration: Optional custom duration in milliseconds

    Returns:
        Flet Animation object
    """
    duration = custom_duration or DURATIONS.get(duration_preset, DURATIONS["normal"])
    curve = CURVES.get(curve_preset, CURVES["standard"])

    return ft.Animation(duration, curve)


def fade_in_animation() -> Dict[str, Any]:
    """
    Create a fade-in animation for controls.

    Returns:
        Format of animation properties
    """
    return {"opacity": 0, "animate_opacity": get_animation("normal", "decelerate")}


def slide_in_animation(direction: str = "up") -> Dict[str, Any]:
    """
    Create a slide-in animation for controls.

    Args:
        direction: "up", "down", "left", or "right"

    Returns:
        Format of animation properties
    """
    offset_map = {
        "up": ft.Offset(0, 0.1),
        "down": ft.Offset(0, -0.1),
        "left": ft.Offset(0.1, 0),
        "right": ft.Offset(-0.1, 0),
    }

    offset = offset_map.get(direction, offset_map["up"])

    return {"offset": offset, "animate_offset": get_animation("normal", "decelerate")}


def animate_size_change(
    control: ft.Control, width: Optional[float] = None, height: Optional[float] = None
):
    """
    Animate size change for a control.

    Args:
        control: Control to animate
        width: New width or None to keep current
        height: New height or None to keep current
    """
    # Only assign width/height/animate to controls that support them
    if isinstance(control, (ft.Container, ft.TextField, ft.ElevatedButton)):
        if width is not None:
            control.width = width
        if height is not None:
            control.height = height
        if isinstance(control, ft.Container) and not getattr(control, "animate", None):
            control.animate = get_animation()
        control.update()
