# src/frontend/styles/animations.py

import flet as ft

class GyroAnimations:
    """Animation configurations"""

    FADE_IN = ft.animation.Animation(duration=300, curve=ft.AnimationCurve.EASE_IN)

    SLIDE_IN = ft.animation.Animation(duration=400, curve=ft.AnimationCurve.EASE_OUT)

    BOUNCE = ft.animation.Animation(duration=600, curve=ft.AnimationCurve.BOUNCE_OUT)
