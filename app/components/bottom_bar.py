"""
bottom_bar.py - Bottom status bar for GyroSI Baby LM

Displays errors, alerts, and system status with a restart button.
"""

import flet as ft
from typing import Optional
import threading
from datetime import datetime

from state import AppState


class BottomBar(ft.UserControl):
    """
    Bottom bar for displaying system status and alerts.

    Features:
    - Error/alert messages with emoji indicators
    - Processing status
    - Restart button for frozen states
    - Auto-dismiss for non-critical messages
    """

    def __init__(self, state: AppState, page: ft.Page):
        super().__init__()
        self.state = state
        self.page = page
        self.message_container: Optional[ft.Container] = None
        self.auto_dismiss_timer: Optional[threading.Timer] = None
        self.status_dot: Optional[ft.Container] = None
        self.status_text: Optional[ft.Text] = None
        self.message_emoji: Optional[ft.Text] = None
        self.message_text: Optional[ft.Text] = None
        self.clear_button: Optional[ft.IconButton] = None

    def build(self):
        # Status indicator components
        self.status_dot = ft.Container(
            width=8,
            height=8,
            bgcolor=self._get_status_color(),
            border_radius=4,
            animate=ft.animation.Animation(300, ft.AnimationCurve.EASE_IN_OUT),
        )

        self.status_text = ft.Text(self._get_status_text(), size=12, color="#8E8E93")

        # Status indicator row
        status_indicator = ft.Row(
            controls=[self.status_dot, self.status_text], spacing=8, visible=True
        )

        # Message components
        self.message_emoji = ft.Text(self._get_message_emoji(), size=16)

        self.message_text = ft.Text(
            self._get_message_text(),
            size=13,
            color="#FFFFFF",
            expand=True,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )

        # Message container
        self.message_container = ft.Container(
            content=ft.Row(controls=[self.message_emoji, self.message_text], spacing=8),
            padding=ft.padding.symmetric(horizontal=10, vertical=5),
            bgcolor=self._get_message_bgcolor(),
            border_radius=8,
            visible=self._has_message(),
            animate=ft.animation.Animation(200, ft.AnimationCurve.EASE_OUT),
        )

        # Clear button for messages
        self.clear_button = ft.IconButton(
            icon=ft.icons.CLOSE,
            icon_size=16,
            icon_color="#8E8E93",
            on_click=self._clear_message,
            visible=self._has_message(),
        )

        # Restart button
        restart_button = ft.TextButton(
            text="Restart",
            icon=ft.icons.REFRESH,
            icon_color="#FF9F0A",
            style=ft.ButtonStyle(
                color="#FF9F0A", padding=ft.padding.symmetric(horizontal=10, vertical=5)
            ),
            on_click=self._on_restart_click,
        )

        # Create row with only valid controls
        row_controls = [
            status_indicator,
            ft.Container(expand=True),  # Spacer
            self.message_container,
            self.clear_button,
            ft.Container(width=20),  # Spacer
            restart_button,
        ]

        return ft.Container(
            content=ft.Row(controls=row_controls, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            height=40,
            padding=ft.padding.symmetric(horizontal=20, vertical=5),
            bgcolor="#000000",
            border=ft.border.only(top=ft.BorderSide(1, "#38383A")),
        )

    def _get_status_color(self) -> str:
        """Get status indicator color based on state."""
        if self.state.error_message:
            return "#FF453A"  # Red for error
        elif self.state.processing:
            return "#FF9F0A"  # Orange for processing
        else:
            return "#30D158"  # Green for ready

    def _get_status_text(self) -> str:
        """Get status text based on state."""
        if self.state.error_message:
            return "Error"
        elif self.state.processing:
            return "Processing"
        else:
            return "Ready"

    def _get_message_emoji(self) -> str:
        """Get emoji for current message type."""
        if self.state.error_message:
            return "❌"
        elif self.state.status_message:
            if "pattern" in self.state.status_message.lower():
                return "🧠"
            elif "cycle" in self.state.status_message.lower():
                return "🔄"
            else:
                return "ℹ️"
        return ""

    def _get_message_text(self) -> str:
        """Get current message text."""
        return self.state.error_message or self.state.status_message or ""

    def _get_message_bgcolor(self) -> str:
        """Get message background color based on type."""
        if self.state.error_message:
            return "#FF453A20"  # Red with transparency
        else:
            return "#0A84FF20"  # Blue with transparency

    def _has_message(self) -> bool:
        """Check if there's a message to display."""
        return bool(self.state.error_message or self.state.status_message)

    def _clear_message(self, e):
        """Clear the current message."""
        # Cancel any pending auto-dismiss
        self._cancel_auto_dismiss()

        self.state.clear_error()
        self.state.status_message = None
        self.update()

    def _cancel_auto_dismiss(self):
        """Cancel the auto-dismiss timer if it exists."""
        if self.auto_dismiss_timer and self.auto_dismiss_timer.is_alive():
            self.auto_dismiss_timer.cancel()
            self.auto_dismiss_timer = None

    def _handle_auto_dismiss(self):
        """Handle the auto-dismiss timer timeout."""
        if self.page:
            self.state.status_message = None
            if self.message_container:
                self.message_container.visible = False
            if self.clear_button:
                self.clear_button.visible = False
            self.update()

    def _on_restart_click(self, e):
        """Handle restart button click."""
        # Reset the agent state
        if self.state.current_agent:
            try:
                agent_uuid = self.state.agent_uuid
                self.state.current_agent.close()
                # Check if agent_uuid is not None before passing
                if agent_uuid is not None:
                    self.state.set_agent(agent_uuid)
                self.state.status_message = "System restarted successfully"
            except Exception as ex:
                self.state.error_message = f"Restart failed: {str(ex)}"
        self.update()

    def set_status_message(self, message: str, auto_dismiss: bool = True):
        """Set a status message with optional auto-dismiss."""
        # Cancel any existing timer
        self._cancel_auto_dismiss()

        self.state.status_message = message

        # Update the message display
        if self.message_emoji:
            self.message_emoji.value = self._get_message_emoji()
        if self.message_text:
            self.message_text.value = self._get_message_text()
        if self.message_container:
            self.message_container.visible = True
            self.message_container.bgcolor = self._get_message_bgcolor()
        if self.clear_button:
            self.clear_button.visible = True

        # Handle auto-dismiss using Python's threading.Timer
        if auto_dismiss:
            self.auto_dismiss_timer = threading.Timer(5.0, self._schedule_auto_dismiss)
            self.auto_dismiss_timer.daemon = True  # So timer doesn't prevent app exit
            self.auto_dismiss_timer.start()

        self.update()

    def _schedule_auto_dismiss(self):
        """Schedule auto-dismiss to run on the UI thread."""
        if self.page:
            # Directly call the handler, since add_to_update_queue does not exist in Flet 0.19.0
            self._handle_auto_dismiss()

    def update(self):
        """Update the bottom bar based on state changes."""
        # Update status indicator
        if self.status_dot:
            self.status_dot.bgcolor = self._get_status_color()

        if self.status_text:
            self.status_text.value = self._get_status_text()

        # Update message content
        if self.message_emoji:
            self.message_emoji.value = self._get_message_emoji()

        if self.message_text:
            self.message_text.value = self._get_message_text()

        # Update container properties
        if self.message_container:
            self.message_container.visible = self._has_message()
            self.message_container.bgcolor = self._get_message_bgcolor()

        # Update clear button visibility
        if self.clear_button:
            self.clear_button.visible = self._has_message()

        super().update()

    def did_unmount(self):
        """Clean up when component is removed from the page."""
        self._cancel_auto_dismiss()
