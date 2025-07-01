"""
input_area.py - Message input component for GyroSI Baby LM

Handles text input, file attachments, and message sending.
"""

import flet as ft
from typing import Optional, Callable
import os

from state import AppState


class InputArea(ft.UserControl):
    """
    Input area with text field, file attachment, and send button.

    Features:
    - Multi-line text input with auto-resize
    - File attachment button
    - Send button with loading state
    - Keyboard shortcuts (Enter to send, Shift+Enter for new line)
    """

    def __init__(
        self, state: AppState, on_send: Callable[[str], None], on_file_attach: Callable[[str], None]
    ):
        super().__init__()
        self.state = state
        self.on_send = on_send
        self.on_file_attach = on_file_attach

        # Controls
        self.text_field: Optional[ft.TextField] = None
        self.send_button: Optional[ft.IconButton] = None
        self.attach_button: Optional[ft.IconButton] = None
        self.file_picker: Optional[ft.FilePicker] = None

    def build(self):
        # File picker
        self.file_picker = ft.FilePicker(on_result=self._on_file_picked)

        # Text input field
        self.text_field = ft.TextField(
            hint_text="Type a message...",
            multiline=True,
            min_lines=1,
            max_lines=5,
            expand=True,
            border_radius=20,
            bgcolor="#1C1C1E",
            border_color="#38383A",
            focused_border_color="#0A84FF",
            hint_style=ft.TextStyle(color="#8E8E93"),
            text_style=ft.TextStyle(color="#FFFFFF"),
            on_submit=self._on_submit,
            shift_enter=True,  # Shift+Enter for new line
            autofocus=True,
            filled=True,
        )

        # Attach button
        self.attach_button = ft.IconButton(
            icon=ft.icons.ATTACH_FILE,
            icon_color="#8E8E93",
            icon_size=24,
            on_click=self._on_attach_click,
            tooltip="Attach file",
            disabled=self.state.processing,
        )

        # Send button
        self.send_button = ft.IconButton(
            icon=ft.icons.SEND if not self.state.processing else ft.icons.HOURGLASS_EMPTY,
            icon_color="#0A84FF" if not self.state.processing else "#8E8E93",
            icon_size=24,
            on_click=self._on_send_click,
            tooltip="Send message",
            disabled=self.state.processing,
        )

        # Layout
        return ft.Column(
            controls=[
                self.file_picker,
                ft.Container(
                    content=ft.Row(
                        controls=[
                            self.attach_button,
                            ft.Container(
                                content=self.text_field,
                                expand=True,
                                padding=ft.padding.only(left=5, right=5),
                            ),
                            self.send_button,
                        ],
                        spacing=5,
                        vertical_alignment=ft.CrossAxisAlignment.END,
                    ),
                    padding=ft.padding.all(5),
                    bgcolor="#000000",
                    border_radius=25,
                ),
            ],
            spacing=0,
        )

    def _on_submit(self, e):
        """Handle Enter key press."""
        if not e.control.shift_enter:  # Regular Enter sends the message
            self._send_message()

    def _on_send_click(self, e):
        """Handle send button click."""
        self._send_message()

    def _send_message(self):
        """Send the current message."""
        if self.state.processing or not self.text_field:
            return

        text = self.text_field.value.strip() if self.text_field.value is not None else ""
        if not text:
            return

        # Clear the input
        self.text_field.value = ""
        self.text_field.focus()

        # Send through callback
        self.on_send(text)

        # Update UI
        self.update()

    def _on_attach_click(self, e):
        """Handle attach button click."""
        if self.file_picker:
            self.file_picker.pick_files(
                allow_multiple=False, dialog_title="Select a file to process"
            )

    def _on_file_picked(self, e: ft.FilePickerResultEvent):
        """Handle file picker result."""
        if e.files and len(e.files) > 0:
            file_path = e.files[0].path
            if file_path and os.path.exists(file_path):
                self.on_file_attach(file_path)

    def update(self):
        """Update the component based on state."""
        if self.send_button:
            self.send_button.disabled = self.state.processing
            self.send_button.icon = (
                ft.icons.HOURGLASS_EMPTY if self.state.processing else ft.icons.SEND
            )
            self.send_button.icon_color = "#8E8E93" if self.state.processing else "#0A84FF"

        if self.attach_button:
            self.attach_button.disabled = self.state.processing

        if self.text_field:
            self.text_field.disabled = self.state.processing

        super().update()
