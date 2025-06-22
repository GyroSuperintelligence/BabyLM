# src/frontend/gyro_app.py
import flet as ft
from typing import Optional
import asyncio
from pathlib import Path

from .components.gyro_threads_panel import GyroThreadsPanel
from .components.gyro_chat_interface import GyroChatInterface
from .components.gyro_document_upload import GyroDocumentUpload
from .styles.theme import GyroTheme

# Unused variables for future debug: (none listed)

class GyroApp:
    """Main GyroSI Baby ML application with Apple-like design"""

    def __init__(self):
        self.page: Optional[ft.Page] = None
        self.threads_panel: Optional[GyroThreadsPanel] = None
        self.chat_interface: Optional[GyroChatInterface] = None
        self.document_upload: Optional[GyroDocumentUpload] = None
        self.current_session_id: Optional[str] = None

    async def main(self, page: ft.Page):
        """Initialize and configure the main application"""
        self.page = page

        # Configure page with Apple-like dark theme
        page.title = "GyroSI Baby ML"
        page.theme_mode = ft.ThemeMode.DARK
        page.bgcolor = GyroTheme.BACKGROUND
        page.padding = 0

        # Load Nunito font
        page.fonts = {
            "Nunito": "https://fonts.googleapis.com/css2?family=Nunito:wght@200;300;400;500;600;700;800;900&display=swap"
        }
        page.theme = ft.Theme(font_family="Nunito")

        # Initialize components
        self.threads_panel = GyroThreadsPanel(on_thread_select=self._on_thread_select)
        self.chat_interface = GyroChatInterface()
        self.document_upload = GyroDocumentUpload(on_upload=self._on_document_upload)

        # Build layout
        await self._build_layout()
        self._setup_keyboard_shortcuts()

    async def _build_layout(self):
        """Build the main application layout"""

        # Settings button (top-right)
        settings_button = ft.IconButton(
            icon=ft.icons.SETTINGS_OUTLINED,
            icon_color=GyroTheme.TEXT_SECONDARY,
            icon_size=20,
            on_click=self._show_settings,
            tooltip="Settings",
        )

        # Header bar
        header = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Text(
                        "GyroSI Baby ML",
                        size=18,
                        weight=ft.FontWeight.W_600,
                        color=GyroTheme.TEXT_PRIMARY,
                    ),
                    settings_button,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=ft.padding.symmetric(horizontal=20, vertical=15),
            bgcolor=GyroTheme.SURFACE,
            border=ft.border.only(bottom=ft.BorderSide(1, GyroTheme.BORDER)),
        )

        # Main content area with chat and document upload
        main_content = ft.Container(
            content=ft.Column(
                controls=[
                    # Chat interface (expandable)
                    ft.Container(
                        content=self.chat_interface, expand=True, bgcolor=GyroTheme.BACKGROUND
                    ),
                    # Document upload area
                    ft.Container(
                        content=self.document_upload, padding=20, bgcolor=GyroTheme.BACKGROUND
                    ),
                ],
                spacing=0,
                expand=True,
            ),
            expand=True,
        )

        # Main layout with sidebar and content
        main_layout = ft.Row(
            controls=[
                # Left sidebar with threads
                ft.Container(
                    content=self.threads_panel,
                    width=300,
                    bgcolor=GyroTheme.SIDEBAR,
                    border=ft.border.only(right=ft.BorderSide(1, GyroTheme.BORDER)),
                ),
                # Main content area
                main_content,
            ],
            spacing=0,
            expand=True,
        )

        # Add everything to page
        self.page.add(ft.Column(controls=[header, main_layout], spacing=0, expand=True))

        await self.page.update()

    async def _on_thread_select(self, session_id: str):
        """Handle thread selection"""
        self.current_session_id = session_id
        await self.chat_interface.load_session(session_id)

    async def _on_document_upload(self, file_path: str):
        """Handle document upload"""
        if self.current_session_id:
            # Process document through G2_BU_In
            await self.chat_interface.process_document(file_path)

    async def _show_settings(self, e):
        """Show settings dialog"""
        settings_dialog = SettingsDialog()
        self.page.dialog = settings_dialog
        settings_dialog.open = True
        await self.page.update()

    # ---------- keyboard shortcuts ----------
    def _setup_keyboard_shortcuts(self):
        self.page.on_keyboard_event = self._handle_keyboard

    def _handle_keyboard(self, e: ft.KeyboardEvent):
        # Cmd/Ctrl + N  → new thread
        if (e.meta or e.ctrl) and e.key == "N":
            asyncio.create_task(self.threads_panel._create_new_thread(None))
        # Cmd/Ctrl + O  → open document picker
        if (e.meta or e.ctrl) and e.key == "O":
            self.document_upload._open_file_picker(None)
        # Cmd/Ctrl + ,  → settings
        if (e.meta or e.ctrl) and e.key == ",":
            asyncio.create_task(self._show_settings(None))


class SettingsDialog(ft.AlertDialog):
    """Minimal settings dialog with Apple-like design"""

    def __init__(self):
        self.system_prompt_field = ft.TextField(
            label="System Prompt",
            multiline=True,
            min_lines=3,
            max_lines=5,
            filled=True,
            bgcolor=GyroTheme.INPUT_BG,
            border_radius=8,
            border_color=GyroTheme.BORDER,
            focused_border_color=GyroTheme.ACCENT,
        )

        super().__init__(
            modal=True,
            title=ft.Text("Settings", size=20, weight=ft.FontWeight.W_600),
            content=ft.Container(
                content=ft.Column(
                    controls=[
                        self.system_prompt_field,
                        ft.Divider(height=20, color=GyroTheme.BORDER),
                        ft.Row(
                            controls=[
                                ft.TextButton("Export Memory", on_click=self._export_memory),
                                ft.TextButton("Import Memory", on_click=self._import_memory),
                            ]
                        ),
                    ],
                    spacing=15,
                ),
                width=400,
                padding=20,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=self._close),
                ft.FilledButton(
                    "Save",
                    on_click=self._save_settings,
                    style=ft.ButtonStyle(bgcolor=GyroTheme.ACCENT, color=GyroTheme.TEXT_ON_ACCENT),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            shape=ft.RoundedRectangleBorder(radius=12),
        )

    async def _export_memory(self, e):
        """Handle memory export"""
        # Implement G2_BU_Eg export
        pass

    async def _import_memory(self, e):
        """Handle memory import"""
        # Implement G2_BU_In import
        pass

    async def _save_settings(self, e):
        """Save settings to G2 epigenetic memory"""
        # Save system prompt
        self._close(e)

    def _close(self, e):
        """Close dialog"""
        self.open = False
        e.page.update()


def main():
    """Entry point for the application"""
    app = GyroApp()
    ft.app(target=app.main, assets_dir="assets")
