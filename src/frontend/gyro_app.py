# src/frontend/gyro_app.py
import flet as ft
from typing import Optional

from .components.gyro_threads_panel import GyroThreadsPanel
from .components.gyro_chat_interface import GyroChatInterface
from .components.gyro_document_upload import GyroDocumentUpload
from .assets.styles.theme import GyroTheme

# Import the real system
from core.extension_manager import ExtensionManager


class GyroApp:
    """Main GyroSI Baby ML application integrated with ExtensionManager"""

    def __init__(self):
        self.page: Optional[ft.Page] = None
        self.threads_panel: Optional[GyroThreadsPanel] = None
        self.chat_interface: Optional[GyroChatInterface] = None
        self.document_upload: Optional[GyroDocumentUpload] = None
        self.current_session_id: Optional[str] = None

        # Initialize the real GyroSI system
        self.extension_manager: Optional[ExtensionManager] = None

    def main(self, page: ft.Page):
        """Initialize and configure the main application"""
        self.page = page

        # Configure page
        page.title = "GyroSI Baby ML"
        page.theme_mode = ft.ThemeMode.DARK
        page.bgcolor = GyroTheme.BACKGROUND
        page.padding = 0

        # Load fonts
        page.fonts = {
            "Nunito": "https://fonts.googleapis.com/css2?family=Nunito:wght@200;300;400;500;600;700;800;900&display=swap"
        }
        page.theme = ft.Theme(font_family="Nunito")

        # Initialize GyroSI system
        try:
            self.extension_manager = ExtensionManager()
            self.current_session_id = self.extension_manager.get_session_id()

            # Initialize components with real system
            self.threads_panel = GyroThreadsPanel(
                on_thread_select=self._on_thread_select, extension_manager=self.extension_manager
            )
            self.chat_interface = GyroChatInterface(extension_manager=self.extension_manager)
            self.document_upload = GyroDocumentUpload(on_upload=self._on_document_upload)

            # Build layout
            self._build_layout()
            self._setup_keyboard_shortcuts()

        except Exception as e:
            # Show error in UI
            page.add(
                ft.Text(f"Failed to initialize GyroSI system: {str(e)}", color=GyroTheme.ERROR)
            )
            page.update()

    def _build_layout(self):
        """Build the main application layout"""

        # Get system health for header
        if self.extension_manager:
            phase = self.extension_manager.engine.phase
            knowledge_id = self.extension_manager.get_knowledge_id()
        else:
            phase = 0
            knowledge_id = "demo"

        # Settings button
        settings_button = ft.IconButton(
            icon=ft.icons.SETTINGS_OUTLINED,
            icon_color=GyroTheme.TEXT_SECONDARY,
            icon_size=20,
            on_click=self._show_settings,
            tooltip="Settings",
        )

        # Header with real system info
        header = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Text(
                        "GyroSI Baby ML",
                        size=18,
                        weight=ft.FontWeight.W_600,
                        color=GyroTheme.TEXT_PRIMARY,
                    ),
                    ft.Column(
                        controls=[
                            ft.Text(f"Phase: {phase}/47", size=12, color=GyroTheme.TEXT_SECONDARY),
                            ft.Text(
                                f"Knowledge: {knowledge_id[:8]}...",
                                size=10,
                                color=GyroTheme.TEXT_TERTIARY,
                            ),
                        ],
                        spacing=2,
                    ),
                    settings_button,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=ft.padding.symmetric(horizontal=20, vertical=15),
            bgcolor=GyroTheme.SURFACE,
            border=ft.border.only(bottom=ft.BorderSide(1, GyroTheme.BORDER)),
        )

        # Main content
        main_content = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Container(
                        content=self.chat_interface, expand=True, bgcolor=GyroTheme.BACKGROUND
                    ),
                    ft.Container(
                        content=self.document_upload, padding=20, bgcolor=GyroTheme.BACKGROUND
                    ),
                ],
                spacing=0,
                expand=True,
            ),
            expand=True,
        )

        # Main layout
        main_layout = ft.Row(
            controls=[
                ft.Container(
                    content=self.threads_panel,
                    width=300,
                    bgcolor=GyroTheme.SIDEBAR,
                    border=ft.border.only(right=ft.BorderSide(1, GyroTheme.BORDER)),
                ),
                main_content,
            ],
            spacing=0,
            expand=True,
        )

        if self.page:
            self.page.add(ft.Column(controls=[header, main_layout], spacing=0, expand=True))
            self.page.update()

    def _on_thread_select(self, session_id: str):
        """Handle thread selection - create new session or switch"""
        try:
            # For now, just load the session in chat interface
            # TODO: Implement proper session switching via ExtensionManager
            self.current_session_id = session_id
            if self.chat_interface:
                self.chat_interface.load_session(session_id)
        except Exception as e:
            self._show_error(f"Failed to load session: {str(e)}")

    def _on_document_upload(self, file_path: str):
        """Handle document upload through G2_BU_In"""
        try:
            if self.current_session_id and self.chat_interface:
                self.chat_interface.process_document(file_path)
        except Exception as e:
            self._show_error(f"Failed to process document: {str(e)}")

    def _show_settings(self, e):
        """Show settings dialog with real system integration"""
        if self.extension_manager and self.page:
            settings_dialog = SettingsDialog(self.extension_manager)
            self.page.dialog = settings_dialog
            self.page.update()

    def _show_error(self, message: str):
        """Show error message to user"""
        if not self.page:
            return

        error_dialog = ft.AlertDialog(
            title=ft.Text("Error"),
            content=ft.Text(message),
            actions=[ft.TextButton("OK", on_click=lambda _: self._close_dialog())],
        )
        self.page.dialog = error_dialog
        self.page.update()

    def _close_dialog(self):
        """Close current dialog"""
        if self.page and self.page.dialog:
            self.page.dialog = None
            self.page.update()

    def _setup_keyboard_shortcuts(self):
        if self.page:
            self.page.on_keyboard_event = self._handle_keyboard

    def _handle_keyboard(self, e: ft.KeyboardEvent):
        if (e.meta or e.ctrl) and e.key == "N":
            if self.threads_panel:
                self.threads_panel._create_new_thread(None)
        elif (e.meta or e.ctrl) and e.key == "O":
            if self.document_upload:
                self.document_upload._open_file_picker(None)
        elif (e.meta or e.ctrl) and e.key == ",":
            self._show_settings(None)


class SettingsDialog(ft.AlertDialog):
    """Settings dialog integrated with ExtensionManager"""

    def __init__(self, extension_manager: ExtensionManager):
        self.extension_manager = extension_manager

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

        # System health display
        health_text = f"""Session: {extension_manager.get_session_id()[:8]}...
Knowledge: {extension_manager.get_knowledge_id()[:8]}...
Phase: {extension_manager.engine.phase}/47
Extensions: {len(extension_manager.extensions)} loaded"""

        super().__init__(
            modal=True,
            title=ft.Text("GyroSI Settings", size=20, weight=ft.FontWeight.W_600),
            content=ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text("System Status:", weight=ft.FontWeight.W_500),
                        ft.Text(health_text, size=12, color=GyroTheme.TEXT_SECONDARY),
                        ft.Divider(height=20, color=GyroTheme.BORDER),
                        self.system_prompt_field,
                        ft.Divider(height=20, color=GyroTheme.BORDER),
                        ft.Row(
                            controls=[
                                ft.TextButton("Export Knowledge", on_click=self._export_knowledge),
                                ft.TextButton("Import Knowledge", on_click=self._import_knowledge),
                                ft.TextButton("Fork Knowledge", on_click=self._fork_knowledge),
                            ]
                        ),
                    ],
                    spacing=15,
                ),
                width=500,
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

    def _export_knowledge(self, e):
        """Export current knowledge package"""
        try:
            # Use file picker to get save location
            # For now, just export to default location
            output_path = f"knowledge_export_{self.extension_manager.get_knowledge_id()[:8]}.gyro"
            self.extension_manager.export_knowledge(output_path)
            # TODO: Show success message
            print(f"Knowledge exported to: {output_path}")
        except Exception as ex:
            print(f"Error exporting knowledge: {ex}")

    def _import_knowledge(self, e):
        """Import knowledge package"""
        try:
            # TODO: Use file picker to select .gyro file
            # For now, just show a placeholder
            print("Import knowledge - file picker not implemented yet")
            # self.extension_manager.import_knowledge(bundle_path)
        except Exception as ex:
            print(f"Error importing knowledge: {ex}")

    def _fork_knowledge(self, e):
        """Fork current knowledge"""
        try:
            new_knowledge_id = self.extension_manager.fork_knowledge()
            print(f"Knowledge forked to: {new_knowledge_id}")
            # TODO: Show success message with new knowledge ID
        except Exception as ex:
            print(f"Error forking knowledge: {ex}")

    def _save_settings(self, e):
        """Save settings to G2 epigenetic memory"""
        try:
            # Store system prompt in epigenetic memory
            prompt = self.system_prompt_field.value
            if prompt:
                self.extension_manager.gyro_epigenetic_memory("current.gyrotensor_com", prompt)
            self._close(e)
        except Exception as ex:
            print(f"Error saving settings: {ex}")

    def _close(self, e):
        """Close dialog"""
        if e.page:
            e.page.dialog = None
            e.page.update()


def main():
    """Entry point for the application"""
    app = GyroApp()
    ft.app(target=app.main)
