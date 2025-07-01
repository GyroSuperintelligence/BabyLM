"""
common.py - Common UI components for GyroSI Baby LM

Reusable components used across multiple views.
"""

import flet as ft
from typing import Optional, List, Any, Callable


class Section(ft.UserControl):
    """
    Reusable section container with title and content.
    """

    def __init__(self, title: str, controls: List[ft.Control]):
        super().__init__()
        self.title = title
        # Flatten controls_list if any Columns are present, so only Controls are in the list
        self.controls_list: List[ft.Control] = []
        for ctrl in controls:
            if isinstance(ctrl, ft.Column):
                self.controls_list.extend(ctrl.controls)
            else:
                self.controls_list.append(ctrl)

    def build(self):
        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Container(
                        content=ft.Text(
                            self.title, size=16, weight=ft.FontWeight.W_600, color="#FFFFFF"
                        ),
                        padding=ft.padding.only(left=20, right=20, top=10, bottom=5),
                    ),
                    ft.Container(
                        content=ft.Column(controls=self.controls_list, spacing=0),
                        bgcolor="#1C1C1E",
                        border_radius=12,
                    ),
                ],
                spacing=10,
            )
        )


class SettingRow(ft.UserControl):
    """
    Reusable setting row with title, subtitle, and control.
    """

    def __init__(
        self, title: str, subtitle: Optional[str] = None, control: Optional[ft.Control] = None
    ):
        super().__init__()
        self.title = title
        self.subtitle = subtitle
        self.control = control

    def build(self):
        # Text content
        text_controls: List[ft.Control] = [
            ft.Text(self.title, size=14, color="#FFFFFF", weight=ft.FontWeight.W_500)
        ]
        if self.subtitle:
            text_controls.append(ft.Text(self.subtitle, size=12, color="#8E8E93"))
        text_content = ft.Column(controls=text_controls, spacing=2, expand=True)

        # Build row
        row_controls: List[ft.Control] = [text_content]
        if self.control:
            row_controls.append(self.control)

        return ft.Container(
            content=ft.Row(
                controls=row_controls,
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.padding.all(16),
            border=ft.border.only(bottom=ft.BorderSide(1, "#000000")),
        )


class ActionButton(ft.UserControl):
    """
    Reusable action button with icon and text.
    """

    def __init__(
        self,
        text: str,
        icon: Optional[str] = None,
        on_click: Optional[Callable] = None,
        style: str = "primary",  # primary, secondary, danger
        full_width: bool = False,
    ):
        super().__init__()
        self.text = text
        self.icon = icon
        self.on_click = on_click
        self.style = style
        self.full_width = full_width

    def build(self):
        # Style configurations
        styles = {
            "primary": {"bgcolor": "#0A84FF", "color": "#FFFFFF", "hover_bgcolor": "#0970D9"},
            "secondary": {"bgcolor": "#38383A", "color": "#FFFFFF", "hover_bgcolor": "#48484A"},
            "danger": {"bgcolor": "#FF453A", "color": "#FFFFFF", "hover_bgcolor": "#E5352B"},
        }

        style_config = styles.get(self.style, styles["primary"])

        button = ft.ElevatedButton(
            text=self.text,
            icon=self.icon,
            on_click=self.on_click,
            style=ft.ButtonStyle(
                color=style_config["color"],
                bgcolor=style_config["bgcolor"],
                overlay_color=style_config["hover_bgcolor"],
                shape=ft.RoundedRectangleBorder(radius=8),
                padding=ft.padding.symmetric(horizontal=16, vertical=12),
            ),
            expand=self.full_width,
        )

        if self.full_width:
            return ft.Container(content=button, width=float("inf"))
        return button


class MetricCard(ft.UserControl):
    """
    Metric display card with icon, value, and subtitle.
    """

    def __init__(
        self,
        title: str,
        value: str,
        subtitle: Optional[str] = None,
        icon: Optional[str] = None,
        color: str = "#0A84FF",
    ):
        super().__init__()
        self.title = title
        self.value = value
        self.subtitle = subtitle
        self.icon = icon
        self.color = color

    def build(self):
        content_items: List[ft.Control] = []

        # Icon if provided
        if self.icon:
            content_items.append(ft.Icon(self.icon, size=24, color=self.color))

        # Text content
        text_controls: List[ft.Control] = [
            ft.Text(self.title, size=12, color="#8E8E93", weight=ft.FontWeight.W_500),
            ft.Text(self.value, size=20, color="#FFFFFF", weight=ft.FontWeight.W_600),
        ]

        if self.subtitle:
            text_controls.append(ft.Text(self.subtitle, size=11, color="#8E8E93"))

        # Instead of appending a Column to a list of Controls, add all text_controls directly if needed
        content_items.extend(text_controls)

        return ft.Container(
            content=ft.Row(
                controls=content_items, spacing=10, alignment=ft.MainAxisAlignment.START
            ),
            padding=ft.padding.all(16),
            bgcolor="#1C1C1E",
            border_radius=12,
            border=ft.border.all(1, "#38383A"),
            width=150,
            height=100,
        )
