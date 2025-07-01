"""
message_area.py - Message display component for GyroSI Baby LM

Displays the conversation history with proper formatting and animations.
"""

import flet as ft
from typing import Optional, List
from datetime import datetime

from state import AppState, Message


class MessageBubble(ft.UserControl):
    """
    Individual message bubble with role-based styling.
    """

    def __init__(self, message: Message):
        super().__init__()
        self.message = message

    def build(self):
        # Determine alignment and colors based on role
        is_user = self.message.role == "user"
        alignment = ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START
        bg_color = "#0A84FF" if is_user else "#1C1C1E"
        text_color = "#FFFFFF"

        # Format timestamp
        time_str = self.message.timestamp.strftime("%H:%M")

        # Message content
        content = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(self.message.content, color=text_color, size=14, selectable=True),
                    ft.Text(
                        time_str,
                        color="#8E8E93",
                        size=11,
                        text_align=ft.TextAlign.RIGHT if is_user else ft.TextAlign.LEFT,
                    ),
                ],
                spacing=5,
                horizontal_alignment=(
                    ft.CrossAxisAlignment.END if is_user else ft.CrossAxisAlignment.START
                ),
            ),
            padding=ft.padding.all(12),
            bgcolor=bg_color,
            border_radius=ft.border_radius.all(16),
            animate=ft.animation.Animation(200, ft.AnimationCurve.EASE_OUT),
        )

        # Add metadata indicator if present
        if self.message.metadata:
            if "file_path" in self.message.metadata:
                file_indicator = ft.Row(
                    controls=[
                        ft.Icon(ft.icons.ATTACH_FILE, size=14, color="#8E8E93"),
                        ft.Text(
                            f"{self.message.metadata.get('file_size', 0)} bytes",
                            size=11,
                            color="#8E8E93",
                        ),
                    ],
                    spacing=5,
                )
                if hasattr(content, "content") and isinstance(content.content, ft.Column):
                    content.content.controls.insert(1, file_indicator)

            if "processing_stats" in self.message.metadata:
                stats = self.message.metadata["processing_stats"]
                stats_text = ft.Text(
                    f"Ops: {stats.get('accepted_ops', 0)} | Patterns: {stats.get('pattern_promotions', 0)}",
                    size=10,
                    color="#8E8E93",
                    italic=True,
                )
                if hasattr(content, "content") and isinstance(content.content, ft.Column):
                    content.content.controls.append(stats_text)

        # Wrap in row for alignment
        return ft.Row(controls=[content], alignment=alignment)


class MessageArea(ft.UserControl):
    """
    Scrollable area displaying all messages in the current thread.

    Features:
    - Auto-scroll to bottom on new messages
    - Loading indicator during processing
    - Empty state for new threads
    - Smooth animations
    """

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.scroll_view: Optional[ft.ListView] = None
        self.message_column: Optional[ft.Column] = None

    def build(self):
        # Message container
        self.message_column = ft.Column(
            controls=self._build_messages(), spacing=10, scroll=ft.ScrollMode.AUTO
        )

        # Scroll view
        self.scroll_view = ft.ListView(
            controls=[self.message_column] if self.message_column is not None else [],
            expand=True,
            padding=ft.padding.all(20),
            auto_scroll=True,
            spacing=10,
        )

        # Container with loading overlay
        stack_controls = [
            c
            for c in [self.scroll_view, self._build_loading_overlay(), self._build_empty_state()]
            if isinstance(c, ft.Control)
        ]
        return ft.Stack(controls=stack_controls, expand=True)

    def _build_messages(self) -> List[ft.Control]:
        """Build message list from current state."""
        if not self.state.current_messages:
            return []

        controls: List[ft.Control] = []

        # Add date separators and messages
        last_date = None
        for message in self.state.current_messages:
            # Add date separator if needed
            message_date = message.timestamp.date()
            if last_date != message_date:
                date_text = (
                    "Today"
                    if message_date == datetime.now().date()
                    else message_date.strftime("%B %d, %Y")
                )
                controls.append(
                    ft.Container(
                        content=ft.Text(
                            date_text, size=12, color="#8E8E93", text_align=ft.TextAlign.CENTER
                        ),
                        alignment=ft.alignment.center,
                        margin=ft.margin.symmetric(vertical=10),
                    )
                )
                last_date = message_date

            # Add message bubble
            controls.append(MessageBubble(message))

        # Filter out any None or unknown types
        return [c for c in controls if isinstance(c, ft.Control)]

    def _build_loading_overlay(self) -> ft.Control:
        """Build loading overlay shown during processing."""
        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.ProgressRing(width=30, height=30, stroke_width=3, color="#0A84FF"),
                    ft.Text("Processing...", size=14, color="#8E8E93"),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
            ),
            alignment=ft.alignment.bottom_center,
            padding=ft.padding.only(bottom=100),
            visible=self.state.processing,
            animate_opacity=ft.animation.Animation(200),
        )

    def _build_empty_state(self) -> ft.Control:
        """Build empty state for new threads."""
        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Icon(ft.icons.CHAT_BUBBLE_OUTLINE, size=64, color="#38383A"),
                    ft.Text(
                        "Start a conversation", size=18, color="#8E8E93", weight=ft.FontWeight.W_500
                    ),
                    ft.Text(
                        "Type a message or upload a file to begin",
                        size=14,
                        color="#8E8E93",
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
            ),
            alignment=ft.alignment.center,
            visible=len(self.state.current_messages) == 0 and not self.state.processing,
            animate_opacity=ft.animation.Animation(200),
        )

    def update(self):
        """Update the message area when state changes."""
        if self.message_column:
            # Rebuild messages
            self.message_column.controls = self._build_messages()

        # Update loading overlay visibility
        if hasattr(self, "controls") and len(self.controls) > 1:
            self.controls[1].visible = self.state.processing
            self.controls[2].visible = (
                len(self.state.current_messages) == 0 and not self.state.processing
            )

        # Auto-scroll to bottom if we have a scroll view
        if self.scroll_view and self.state.current_messages:
            self.scroll_view.scroll_to(offset=-1, duration=300)

        super().update()
