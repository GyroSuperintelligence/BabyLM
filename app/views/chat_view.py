"""
chat_view.py - Main chat interface for GyroSI Baby LM

Implements the primary chat view with thread list and message area.
"""

import flet as ft
from typing import Optional, Callable
from datetime import datetime

from state import AppState
from components.thread_list import ThreadList
from components.message_area import MessageArea
from components.input_area import InputArea


class ChatView(ft.UserControl):
    """
    Main chat view containing thread list, message area, and input.

    Layout:
    - Left: Thread list (collapsible)
    - Center: Message area
    - Bottom: Input area
    """

    def __init__(self, state: AppState, page: ft.Page):
        super().__init__()
        self.state = state
        self.page = page
        self.thread_list_visible = True

        # Components
        self.thread_list: Optional[ThreadList] = None
        self.message_area: Optional[MessageArea] = None
        self.input_area: Optional[InputArea] = None

        # Register for state updates
        self.state.add_update_callback(self._on_state_update)

    def build(self):
        # Create components
        self.thread_list = ThreadList(
            state=self.state,
            on_thread_select=self._on_thread_select,
            on_new_thread=self._on_new_thread,
        )

        self.message_area = MessageArea(state=self.state)

        self.input_area = InputArea(
            state=self.state, on_send=self._on_send_message, on_file_attach=self._on_file_attach
        )

        # Thread list toggle button
        toggle_button = ft.IconButton(
            icon=ft.icons.MENU,
            icon_color="#8E8E93",
            icon_size=20,
            on_click=self._toggle_thread_list,
            tooltip="Toggle thread list",
        )

        # Main content area
        main_content = ft.Column(
            controls=[
                # Header with toggle
                ft.Container(
                    content=ft.Row(
                        controls=[
                            toggle_button,
                            ft.Text(
                                self._get_current_thread_title(),
                                size=16,
                                weight=ft.FontWeight.W_500,
                                color="#FFFFFF",
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.START,
                    ),
                    padding=ft.padding.all(10),
                    border=ft.border.only(bottom=ft.BorderSide(1, "#38383A")),
                ),
                # Message area (expanded)
                ft.Container(content=self.message_area, expand=True, padding=ft.padding.all(0)),
                # Input area
                ft.Container(
                    content=self.input_area,
                    padding=ft.padding.all(10),
                    border=ft.border.only(top=ft.BorderSide(1, "#38383A")),
                ),
            ],
            spacing=0,
            expand=True,
        )

        # Layout with animated thread list
        self.thread_list_container = ft.Container(
            content=self.thread_list,
            width=250,
            bgcolor="#1C1C1E",
            border=ft.border.only(right=ft.BorderSide(1, "#38383A")),
            animate=ft.animation.Animation(200, ft.AnimationCurve.EASE_IN_OUT),
        )

        return ft.Row(
            controls=[
                self.thread_list_container,
                ft.Container(content=main_content, expand=True, bgcolor="#000000"),
            ],
            spacing=0,
            expand=True,
        )

    def _get_current_thread_title(self) -> str:
        """Get the title of the current thread."""
        if not self.state.current_thread_id:
            return "No thread selected"
        thread = self.state.threads.get(self.state.current_thread_id)
        return thread.title if thread else "Unknown thread"

    def _toggle_thread_list(self, e):
        """Toggle thread list visibility with animation."""
        self.thread_list_visible = not self.thread_list_visible
        self.thread_list_container.width = 250 if self.thread_list_visible else 0
        self.update()

    def _on_thread_select(self, thread_id: str):
        """Handle thread selection."""
        try:
            self.state.set_current_thread(thread_id)
        except Exception as e:
            self.state.error_message = f"Failed to load thread: {str(e)}"

    def _on_new_thread(self):
        """Handle new thread creation."""
        try:
            thread_id = self.state.create_thread()
            # Optionally show a dialog to set thread title
            self._show_thread_title_dialog(thread_id)
        except Exception as e:
            self.state.error_message = f"Failed to create thread: {str(e)}"

    def _show_thread_title_dialog(self, thread_id: str):
        """Show dialog to set thread title."""
        title_field = ft.TextField(
            label="Thread Title",
            value=f"Thread {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            autofocus=True,
            border_radius=12,
            bgcolor="#1C1C1E",
            border_color="#38383A",
            focused_border_color="#0A84FF",
        )

        def save_title(e):
            if thread_id in self.state.threads:
                title_value = title_field.value
                if title_value is not None:
                    self.state.threads[thread_id].title = title_value
                    self.state._notify_updates()
            if self.page is not None:
                dlg = self.page.dialog
                if dlg is not None:
                    dlg.visible = False
                self.page.update()

        def cancel(e):
            if self.page is not None:
                dlg = self.page.dialog
                if dlg is not None:
                    dlg.visible = False
                self.page.update()

        dialog = ft.AlertDialog(
            title=ft.Text("New Thread", color="#FFFFFF"),
            content=ft.Container(content=title_field, width=300, padding=ft.padding.all(10)),
            actions=[
                ft.TextButton("Cancel", on_click=cancel),
                ft.TextButton("Save", on_click=save_title),
            ],
            content_padding=ft.padding.all(10),
        )

        # Style the dialog with a container
        dialog_container = ft.Container(content=dialog, bgcolor="#1C1C1E")

        if self.page is not None:
            self.page.dialog = dialog
            self.page.overlay.append(dialog_container)
            self.page.update()

    def _on_send_message(self, text: str):
        """Handle sending a message."""
        if not text.strip():
            return

        # Process the message through the state
        self.state.process_user_input(text)

    def _on_file_attach(self, file_path: str):
        """Handle file attachment."""
        self.state.process_file(file_path)

    def _on_state_update(self):
        """Handle state updates."""
        # Update components if they exist
        if self.thread_list:
            self.thread_list.update()
        if self.message_area:
            self.message_area.update()
        if self.input_area:
            self.input_area.update()

        # Update thread title
        if hasattr(self, "controls") and self.controls:
            self.update()

    def did_mount(self):
        """Called when the view is mounted."""
        # Initialize with a thread if none exists
        if not self.state.current_thread_id and not self.state.threads:
            self.state.create_thread("Welcome Thread")

    def will_unmount(self):
        """Called when the view is being unmounted."""
        # Unregister from state updates
        self.state.remove_update_callback(self._on_state_update)
