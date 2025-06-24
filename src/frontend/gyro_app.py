# src/frontend/gyro_app.py
import flet as ft
from typing import Optional, Dict, List, Any, Callable, cast, Sized
import os
import time
import threading
import subprocess
from dataclasses import dataclass
from enum import Enum
import queue

# Optional pyperclip import
try:
    import pyperclip  # type: ignore[import]

    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False

from src.frontend.components.gyro_threads_panel import GyroThreadsPanel
from src.frontend.components.gyro_chat_interface import GyroChatInterface
from src.frontend.components.gyro_document_upload import GyroDocumentUpload
from src.frontend.assets.styles.theme import GyroTheme

# Import the real system
from core.extension_manager import ExtensionManager
from core.gyro_api import ingest_curriculum_resource, CURRICULUM_RESOURCES


class AppState(Enum):
    """Application state enumeration"""

    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    LOADING = "loading"


@dataclass
class CurriculumResource:
    """Data class for curriculum resources"""

    name: str
    type: str
    size: str
    desc: str
    key: Optional[str] = None


class GyroApp:
    """Main GyroSI Baby ML application integrated with ExtensionManager"""

    def __init__(self):
        self.page: Optional[ft.Page] = None
        self.threads_panel: Optional[GyroThreadsPanel] = None
        self.chat_interface: Optional[GyroChatInterface] = None
        self.document_upload: Optional[GyroDocumentUpload] = None
        self.current_session_id: Optional[str] = None
        self.extension_manager: Optional[ExtensionManager] = None

        # UI components
        self.header_container: Optional[ft.Container] = None
        self.phase_text: Optional[ft.Text] = None
        self.knowledge_text: Optional[ft.Text] = None

        # File pickers
        self.export_picker = ft.FilePicker(on_result=self._on_export_result)
        self.import_picker = ft.FilePicker(on_result=self._on_import_result)
        self.file_pickers_added = False
        self.active_file_handler = None

        # State management
        self.app_state = AppState.INITIALIZING
        self.error_message: Optional[str] = None

        self._ui_update_queue = queue.Queue()  # For thread-safe UI updates
        # UI updates will be processed when needed via the queue

    def main(self, page: ft.Page):
        """Initialize and configure the main application"""
        self.page = page
        self._configure_page()
        if not self.file_pickers_added:
            self.page.overlay.extend([self.export_picker, self.import_picker])
            self.file_pickers_added = True

        # Check for harmonics matrix
        harmonics_path = os.path.join(os.path.dirname(__file__), "../../core/gyro_harmonics.dat")
        if not os.path.exists(harmonics_path):
            # Auto-generate harmonics matrix
            self._auto_generate_harmonics_matrix(harmonics_path)
            return

        self._initialize_system()

    def _configure_page(self):
        """Configure page settings"""
        if not self.page:
            return

        self.page.title = "GyroSI Baby"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.bgcolor = GyroTheme.BACKGROUND
        self.page.padding = 0

        # Load fonts
        self.page.fonts = {
            "Nunito": "https://fonts.googleapis.com/css2?family=Nunito:wght@200;300;400;500;600;700;800;900&display=swap"
        }
        self.page.theme = ft.Theme(font_family="Nunito")

    def _initialize_system(self):
        """Initialize the GyroSI system"""
        try:
            self.extension_manager = ExtensionManager()
            self.current_session_id = self.extension_manager.get_session_id()

            # Initialize components with lazy imports and error handling
            try:
                from src.frontend.components.gyro_threads_panel import GyroThreadsPanel

                self.threads_panel = GyroThreadsPanel(
                    on_thread_select=self._on_thread_select,
                    extension_manager=self.extension_manager,
                )
            except Exception as e:
                self.threads_panel = None
                self._show_error(f"Failed to load threads panel: {e}")

            try:
                from src.frontend.components.gyro_chat_interface import GyroChatInterface

                self.chat_interface = GyroChatInterface(extension_manager=self.extension_manager)
            except Exception as e:
                self.chat_interface = None
                self._show_error(f"Failed to load chat interface: {e}")

            try:
                from src.frontend.components.gyro_document_upload import GyroDocumentUpload

                self.document_upload = GyroDocumentUpload(on_upload=self._on_document_upload)
            except Exception as e:
                self.document_upload = None
                self._show_error(f"Failed to load document upload: {e}")

            self.app_state = AppState.READY
            self._build_layout()
            self._setup_keyboard_shortcuts()

        except Exception as e:
            self.app_state = AppState.ERROR
            self.error_message = str(e)
            self._show_error_screen(f"Failed to initialize GyroSI system: {str(e)}")

    def _build_header(self) -> ft.Container:
        """Build the header with current session and knowledge info"""
        phase = 0
        knowledge_id = "demo"

        if self.extension_manager:
            phase = self.extension_manager.engine.phase
            knowledge_id = self.extension_manager.get_knowledge_id()

        # Create text components for dynamic updates
        self.phase_text = ft.Text(f"Phase: {phase}/47", size=12, color=GyroTheme.TEXT_SECONDARY)
        self.knowledge_text = ft.Text(
            f"Knowledge: {knowledge_id[:8]}...", size=10, color=GyroTheme.TEXT_TERTIARY
        )

        # Custom SVG icon
        icon_path = os.path.join(os.path.dirname(__file__), "assets/icons/mingcute--baby-fill.svg")
        header_icon = ft.Image(
            src=icon_path,
            width=32,
            height=32,
            fit=ft.ImageFit.CONTAIN,
        )

        header_content = ft.Row(
            controls=[
                ft.Row(
                    controls=[
                        header_icon,
                        ft.Text(
                            "GyroSI Baby ML",
                            size=18,
                            weight=ft.FontWeight.W_600,
                            color=GyroTheme.TEXT_PRIMARY,
                        ),
                    ],
                    spacing=8,
                ),
                ft.Column(
                    controls=[self.phase_text, self.knowledge_text],
                    spacing=2,
                ),
                ft.Row(
                    controls=[
                        self._create_icon_button(
                            ft.icons.STORAGE,
                            "Knowledge Management",
                            self._show_knowledge_management,
                        ),
                        self._create_icon_button(
                            ft.icons.PEOPLE_OUTLINE,
                            "Session Management",
                            self._show_session_management,
                        ),
                        self._create_icon_button(
                            ft.icons.SETTINGS_OUTLINED, "Settings", self._show_settings
                        ),
                    ],
                    spacing=5,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        self.header_container = ft.Container(
            content=header_content,
            padding=ft.padding.symmetric(horizontal=20, vertical=15),
            bgcolor=GyroTheme.SURFACE,
            border=ft.border.only(bottom=ft.BorderSide(1, GyroTheme.BORDER)),
        )

        return self.header_container

    def _create_icon_button(self, icon: str, tooltip: str, on_click: Callable) -> ft.IconButton:
        """Create a styled icon button"""
        return ft.IconButton(
            icon=icon,
            icon_color=GyroTheme.TEXT_SECONDARY,
            icon_size=20,
            on_click=on_click,
            tooltip=tooltip,
        )

    def _update_header(self):
        """Update the header to reflect current session and knowledge info"""
        if not self.extension_manager or not self.page:
            return

        phase = self.extension_manager.engine.phase
        knowledge_id = self.extension_manager.get_knowledge_id()

        if self.phase_text:
            self.phase_text.value = f"Phase: {phase}/47"
        if self.knowledge_text:
            self.knowledge_text.value = f"Knowledge: {knowledge_id[:8]}..."

        if self.header_container:
            self.header_container.update()

    def _build_layout(self):
        """Build the main application layout"""
        if not self.page:
            return

        header = self._build_header()

        # Main content area
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

        # Main layout with sidebar
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

        # Clear and add to page
        if self.page and hasattr(self.page, "controls") and self.page.controls is not None:
            self.page.controls.clear()
            self.page.update()
            self.page.add(ft.Column(controls=[header, main_layout], spacing=0, expand=True))

    def _show_error_screen(self, error_message: str):
        """Display error screen"""
        if not self.page or not hasattr(self.page, "controls") or self.page.controls is None:
            return
        self.page.controls.clear()
        self.page.update()
        self.page.add(
            ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Icon(ft.icons.ERROR_OUTLINE, size=64, color=GyroTheme.ERROR),
                        ft.Text(
                            "Initialization Error",
                            size=24,
                            weight=ft.FontWeight.W_600,
                            color=GyroTheme.ERROR,
                        ),
                        ft.Text(
                            error_message,
                            size=14,
                            color=GyroTheme.TEXT_SECONDARY,
                            text_align=ft.TextAlign.CENTER,
                        ),
                        ft.ElevatedButton("Retry", on_click=lambda _: self._initialize_system()),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=20,
                ),
                alignment=ft.alignment.center,
                expand=True,
            )
        )
        self.page.update()

    def _on_thread_select(self, session_id: str):
        """Handle thread selection"""
        try:
            self.current_session_id = session_id
            if self.chat_interface:
                self.chat_interface.load_session(session_id)
            self._update_header()
        except Exception as e:
            self._show_error(f"Failed to load session: {str(e)}")

    def _on_document_upload(self, file_path: str):
        """Handle document upload"""
        try:
            if self.current_session_id and self.chat_interface:
                self.chat_interface.process_document(file_path)
        except Exception as e:
            self._show_error(f"Failed to process document: {str(e)}")

    def reload_current_context(self):
        """Reload the current context"""
        if self.current_session_id and self.chat_interface:
            self.chat_interface.load_session(self.current_session_id)
        self._update_header()

    def _show_settings(self, e):
        """Show settings dialog"""
        if not self.extension_manager or not self.page:
            return

        dialog = SettingsDialog(
            self.extension_manager,
            page=self.page,
            app=self,  # Pass self reference
            on_close=self._update_header,
            export_picker=self.export_picker,
            import_picker=self.import_picker,
        )
        self.active_file_handler = dialog
        self.page.dialog = dialog
        dialog.open = True
        self.page.update()

    def _show_knowledge_management(self, e):
        """Show knowledge management panel"""
        if not self.page:
            return

        try:
            from src.frontend.components.knowledge_management_panel import KnowledgeManagementPanel

            panel = KnowledgeManagementPanel(
                page=self.page,
                import_picker=self.import_picker,
                export_picker=self.export_picker,
            )
            self.page.dialog = panel
            panel.open = True
            self.page.update()
        except Exception as ex:
            self._show_error(f"Knowledge management panel not available: {ex}")

    def _show_session_management(self, e):
        """Show session management panel"""
        if not self.page:
            return

        try:
            from src.frontend.components.session_management_panel import SessionManagementPanel

            panel = SessionManagementPanel(page=self.page)
            self.page.dialog = panel
            panel.open = True
            self.page.update()
        except Exception as ex:
            self._show_error(f"Session management panel not available: {ex}")

    def _show_error(self, message: str):
        """Show error message to user"""
        if not self.page:
            return

        self.page.snack_bar = ft.SnackBar(
            content=ft.Text(message),
            bgcolor=GyroTheme.ERROR,
            open=True,
        )
        self.page.update()

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        if self.page:
            self.page.on_keyboard_event = self._handle_keyboard

    def _handle_keyboard(self, e: ft.KeyboardEvent):
        """Handle keyboard events"""
        key = e.key.lower() if hasattr(e, "key") and isinstance(e.key, str) else e.key
        if e.ctrl or e.meta:
            if key == "n" and self.threads_panel:
                self.threads_panel._create_new_thread(None)
            elif key == "o" and self.document_upload:
                self.document_upload._open_file_picker(None)
            elif key == ",":
                self._show_settings(None)

    def _auto_generate_harmonics_matrix(self, harmonics_path: str):
        """Auto-generate harmonics matrix"""
        if not self.page:
            return

        # Show progress dialog
        progress_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Initializing GyroSI"),
            content=ft.Column(
                controls=[
                    ft.Text("Setting up your system for the first time..."),
                    ft.ProgressRing(),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=20,
                width=300,
            ),
        )

        if self.page:
            self.page.dialog = progress_dialog
            progress_dialog.open = True
            self.page.update()

        def generate():
            try:
                # Call builder script (no key needed - universal harmonics matrix)
                result = subprocess.run(
                    ["python", "gyro_tools/build_operator_matrix.py", harmonics_path],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    # Success - reinitialize app
                    if self.page:

                        def close_dialog_and_update():
                            if self.page is not None:
                                self.page.dialog = None
                                self.page.update()

                        self._ui_update_queue.put(close_dialog_and_update)
                    self._initialize_system()
                else:
                    # Error
                    self._show_error(f"Failed to initialize system: {result.stderr}")
            except Exception as e:
                self._show_error(f"Error initializing system: {str(e)}")

        # Run in background thread
        threading.Thread(target=generate, daemon=True).start()

    def _on_export_result(self, e: ft.FilePickerResultEvent):
        """Handle export file picker result"""
        if self.active_file_handler and hasattr(self.active_file_handler, "_on_export_result"):
            self.active_file_handler._on_export_result(e)
        self.active_file_handler = None

    def _on_import_result(self, e: ft.FilePickerResultEvent):
        """Handle import file picker result"""
        if self.active_file_handler and hasattr(self.active_file_handler, "_on_import_result"):
            self.active_file_handler._on_import_result(e)
        self.active_file_handler = None

    def _process_ui_update_queue(self, _):
        while not self._ui_update_queue.empty():
            try:
                fn = self._ui_update_queue.get_nowait()
                fn()
            except Exception as e:
                print(f"UI update error: {e}")


class SettingsDialog(ft.AlertDialog):
    """Enhanced settings dialog with better organization and error handling"""

    def __init__(
        self,
        extension_manager: ExtensionManager,
        page: ft.Page,
        app: Optional["GyroApp"] = None,
        on_close: Optional[Callable] = None,
        export_picker: Optional[ft.FilePicker] = None,
        import_picker: Optional[ft.FilePicker] = None,
    ):
        self.extension_manager = extension_manager
        self.page = page
        self.app = app  # Store app reference
        self.on_close = on_close
        self.export_picker = export_picker
        self.import_picker = import_picker

        # Initialize components
        self._init_components()
        self._init_curriculum_resources()

        # Build dialog content
        content = self._build_content()

        super().__init__(
            modal=True,
            title=ft.Text("GyroSI Settings", size=20, weight=ft.FontWeight.W_600),
            content=ft.Container(
                content=ft.Column(
                    controls=[content],
                    scroll=ft.ScrollMode.AUTO,
                ),
                width=600,
                height=700,
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

    def _init_components(self):
        """Initialize dialog components"""
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
        if self.page:
            self.export_picker = ft.FilePicker(on_result=self._on_export_result)
            self.import_picker = ft.FilePicker(on_result=self._on_import_result)
            self.page.overlay.extend([self.export_picker, self.import_picker])

    def _init_curriculum_resources(self):
        """Initialize curriculum resources"""
        self.curriculum_resources = [
            CurriculumResource(
                "WordNet", "Lexical DB", "30MB", "Synonyms, definitions, word relations", "wordnet"
            ),
            CurriculumResource(
                "Wiktionary", "Dictionary", "1-2GB", "Definitions, etymology, usage", "wiktionary"
            ),
            CurriculumResource(
                "Wikipedia (Simple)",
                "Encyclopedia",
                "800MB",
                "Simple English, easier to parse",
                "wikipedia_simple",
            ),
            CurriculumResource(
                "Gutenberg Top 100",
                "Literature",
                "50MB",
                "Classic books, public domain",
                "gutenberg",
            ),
            CurriculumResource(
                "UDHR", "Legal/Philosophy", "<1MB", "Universal Declaration of Human Rights", "udhr"
            ),
            CurriculumResource(
                "Tatoeba", "Sentence DB", "500MB", "Example sentences, translations", "tatoeba"
            ),
            CurriculumResource(
                "News Crawl", "News", "100MB-1GB", "Recent, open news articles", "news"
            ),
            CurriculumResource(
                "OpenSubtitles (sample)",
                "Dialogues",
                "100MB",
                "Conversational, real-world",
                "opensubtitles",
            ),
            CurriculumResource(
                "English Wikibooks",
                "Textbooks",
                "200MB",
                "How-tos, educational content",
                "wikibooks",
            ),
            CurriculumResource(
                "English Wikisource", "Docs", "300MB", "Public domain docs, speeches", "wikisource"
            ),
            CurriculumResource(
                "British Literary Classics",
                "Literature",
                "50MB",
                "Austen, Dickens, Wilde",
                "british_lit",
            ),
            CurriculumResource(
                "Etiquette & Manners", "Nonfiction", "5MB", "Politeness, communication", "etiquette"
            ),
        ]

        self.curriculum_checkboxes: List[ft.Checkbox] = []
        self.curriculum_progress = ft.ProgressBar(width=400, visible=False)
        self.curriculum_status = ft.Text("", size=12, color=GyroTheme.TEXT_SECONDARY)

    def _build_content(self) -> ft.Column:
        """Build dialog content"""
        tabs = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(
                    text="System Health",
                    content=self._build_health_tab(),
                ),
                ft.Tab(
                    text="Configuration",
                    content=self._build_config_tab(),
                ),
                ft.Tab(
                    text="Knowledge",
                    content=self._build_knowledge_tab(),
                ),
                ft.Tab(
                    text="Curriculum",
                    content=self._build_curriculum_tab(),
                ),
            ],
        )
        return ft.Column(
            controls=[tabs],
            spacing=0,
            expand=True,
        )

    def _build_health_tab(self) -> ft.Container:
        """Build system health tab"""
        health = self.extension_manager.get_system_health()

        # Parse health data
        status = health.get("status", "unknown")
        anomalies = health.get("anomalies", [])
        metrics = health.get("system_metrics", {})
        perf = health.get("performance", {})
        gyro_state = health.get("gyro_state", {})
        uptime = health.get("uptime", 0)

        status_color = GyroTheme.ACCENT if status == "healthy" else GyroTheme.ERROR

        # Build health cards
        status_card = self._create_info_card(
            "System Status",
            [
                (f"Status: {status.title()}", status_color),
                (f"Uptime: {self._format_uptime(uptime)}", GyroTheme.TEXT_SECONDARY),
                (f"Phase: {gyro_state.get('phase', '?')}/47", GyroTheme.TEXT_SECONDARY),
            ],
        )

        metrics_card = self._create_info_card(
            "System Metrics",
            [
                (f"CPU: {metrics.get('cpu_percent', '?')}%", GyroTheme.TEXT_SECONDARY),
                (
                    f"Memory: {metrics.get('memory_percent', '?')}% ({int(metrics.get('memory_available_mb', 0))} MB free)",
                    GyroTheme.TEXT_SECONDARY,
                ),
                (f"Disk: {metrics.get('disk_usage_percent', '?')}%", GyroTheme.TEXT_SECONDARY),
            ],
        )

        # Extensions table
        extensions_card = self._build_extensions_table()

        # Performance metrics
        perf_card = self._create_info_card(
            "Performance",
            [
                (f"Response Time: {perf.get('avg_response_ms', '?')}ms", GyroTheme.TEXT_SECONDARY),
                (f"Throughput: {perf.get('requests_per_min', '?')}/min", GyroTheme.TEXT_SECONDARY),
                (f"Error Rate: {perf.get('error_rate', '?')}%", GyroTheme.TEXT_SECONDARY),
            ],
        )

        # Anomalies
        if anomalies:
            anomaly_texts = [f"• {anomaly}" for anomaly in anomalies[:5]]  # Show first 5
            if len(anomalies) > 5:
                anomaly_texts.append(f"... and {len(anomalies) - 5} more")
            anomaly_content = [
                ft.Text(text, size=12, color=GyroTheme.ERROR) for text in anomaly_texts
            ]
            anomaly_card = self._create_info_card("Recent Anomalies", content=anomaly_content)
        else:
            anomaly_card = self._create_info_card(
                "Recent Anomalies", [(f"No anomalies detected", GyroTheme.ACCENT)]
            )

        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("System Health Overview", weight=ft.FontWeight.W_600, size=16),
                    ft.Text(
                        "Monitor system performance and identify potential issues.",
                        size=12,
                        color=GyroTheme.TEXT_SECONDARY,
                    ),
                    status_card,
                    metrics_card,
                    perf_card,
                    anomaly_card,
                    extensions_card,
                ],
                spacing=15,
            ),
            padding=10,
        )

    def _build_config_tab(self) -> ft.Container:
        """Build configuration tab"""
        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Configuration", weight=ft.FontWeight.W_600, size=16),
                    ft.Text(
                        "Configure system behavior and preferences.",
                        size=12,
                        color=GyroTheme.TEXT_SECONDARY,
                    ),
                    self.system_prompt_field,
                ],
                spacing=15,
            ),
            padding=10,
        )

    def _build_knowledge_tab(self) -> ft.Container:
        """Build knowledge management tab"""
        # Get current knowledge info
        knowledge_id = self.extension_manager.get_knowledge_id()
        session_count = self._get_session_count()

        # Knowledge info card
        info_card = self._create_info_card(
            "Current Knowledge",
            [
                (f"ID: {knowledge_id[:8]}...", GyroTheme.TEXT_SECONDARY),
                (f"Active Sessions: {session_count}", GyroTheme.TEXT_SECONDARY),
            ],
        )

        # Actions
        actions = ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.FilledButton(
                            "Export Knowledge",
                            icon=ft.icons.UPLOAD,
                            on_click=self._export_knowledge,
                        ),
                        ft.FilledButton(
                            "Import Knowledge",
                            icon=ft.icons.DOWNLOAD,
                            on_click=self._import_knowledge,
                        ),
                        ft.FilledButton(
                            "Fork Knowledge",
                            icon=ft.icons.CONTENT_COPY,
                            on_click=self._fork_knowledge,
                        ),
                    ],
                    spacing=10,
                ),
            ],
            spacing=10,
        )

        # Encryption section
        encryption_section = self._build_encryption_section()

        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Knowledge Management", weight=ft.FontWeight.W_600, size=16),
                    ft.Text(
                        "Manage knowledge packages and encryption settings.",
                        size=12,
                        color=GyroTheme.TEXT_SECONDARY,
                    ),
                    info_card,
                    actions,
                    encryption_section,
                ],
                spacing=15,
            ),
            padding=10,
        )

    def _build_curriculum_tab(self) -> ft.Container:
        """Build curriculum tab"""
        # Clear existing checkboxes to prevent accumulation
        self.curriculum_checkboxes.clear()

        # Create table header
        header = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Text("Resource", weight=ft.FontWeight.W_600, width=150),
                    ft.Text("Type", weight=ft.FontWeight.W_600, width=100),
                    ft.Text("Size", weight=ft.FontWeight.W_600, width=80),
                    ft.Text("Description", weight=ft.FontWeight.W_600, expand=True),
                    ft.Text("Select", weight=ft.FontWeight.W_600, width=60),
                ],
                spacing=10,
            ),
            padding=ft.padding.only(bottom=10),
        )

        # Create table rows
        rows = []
        for resource in self.curriculum_resources:
            checkbox = ft.Checkbox(value=False)
            self.curriculum_checkboxes.append(checkbox)

            row = ft.Container(
                content=ft.Row(
                    controls=[
                        ft.Text(resource.name, width=150, size=12),
                        ft.Text(resource.type, width=100, size=12),
                        ft.Text(resource.size, width=80, size=12),
                        ft.Text(resource.desc, size=12, expand=True),
                        ft.Container(content=checkbox, width=60),
                    ],
                    spacing=10,
                ),
                padding=ft.padding.symmetric(vertical=5),
                on_hover=self._on_row_hover,
            )
            rows.append(row)

        # Actions
        actions = ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        ft.FilledButton(
                            "Download & Ingest Selected",
                            icon=ft.icons.DOWNLOAD,
                            on_click=self._download_and_ingest_curriculum,
                        ),
                        ft.TextButton(
                            "Select All",
                            on_click=lambda _: self._select_all_curriculum(True),
                        ),
                        ft.TextButton(
                            "Clear All",
                            on_click=lambda _: self._select_all_curriculum(False),
                        ),
                    ],
                    spacing=10,
                ),
                self.curriculum_progress,
                self.curriculum_status,
            ],
            spacing=10,
        )

        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Standard Curriculum Resources", weight=ft.FontWeight.W_600, size=16),
                    ft.Text(
                        "Select resources to download and ingest into the knowledge base.",
                        size=12,
                        color=GyroTheme.TEXT_SECONDARY,
                    ),
                    header,
                    ft.Column(controls=rows, scroll=ft.ScrollMode.AUTO, height=300),
                    actions,
                ],
                spacing=15,
            ),
            padding=10,
        )

    def _save_settings(self, e):
        """Save settings and close dialog"""
        try:
            # Save system prompt if changed
            if hasattr(self, "system_prompt_field") and self.system_prompt_field.value:
                # TODO: Implement system prompt saving
                pass

            self._show_message("Settings saved successfully")
            self._close(e)
        except Exception as ex:
            self._show_message(f"Failed to save settings: {str(ex)}", error=True)

    def _build_encryption_section(self) -> ft.Container:
        """Build encryption configuration section"""
        crypto = self.extension_manager.extensions.get("crypto")

        if crypto and hasattr(crypto, "user_key"):
            status = ft.Text(
                "Encryption: ENABLED", color=GyroTheme.ACCENT, weight=ft.FontWeight.W_600
            )
            key_value = crypto.user_key.hex()

            key_field = ft.TextField(
                label="Current Key",
                value=key_value[:16] + "..." + key_value[-8:],  # Show partial key
                read_only=True,
                password=True,
                can_reveal_password=True,
                width=320,
            )

            actions = ft.Row(
                controls=[
                    ft.IconButton(
                        icon=ft.icons.CONTENT_COPY,
                        tooltip="Copy Key",
                        on_click=lambda _: self._copy_to_clipboard(key_value),
                    ),
                    ft.TextButton("Export Key", on_click=lambda _: self._export_key(key_value)),
                    ft.TextButton("Change Key", on_click=self._change_key),
                ],
                spacing=5,
            )
        else:
            status = ft.Text(
                "Encryption: DISABLED", color=GyroTheme.ERROR, weight=ft.FontWeight.W_600
            )
            key_field = ft.TextField(
                label="No encryption key set",
                read_only=True,
                width=320,
                disabled=True,
            )
            actions = ft.Row(
                controls=[
                    ft.FilledButton(
                        "Enable Encryption",
                        on_click=self._enable_encryption,
                    ),
                ],
            )

        return self._create_info_card(
            "Encryption Settings",
            content=[
                status,
                key_field,
                actions,
                ft.Text(
                    "All user, session, and extension data is encrypted at rest when enabled.",
                    size=12,
                    color=GyroTheme.TEXT_SECONDARY,
                ),
            ],
        )

    def _build_extensions_table(self) -> ft.Container:
        """Build extensions table"""
        rows = []

        for ext in self.extension_manager.extensions.values():
            try:
                name = ext.get_extension_name()
                version = ext.get_extension_version()
                footprint = ext.get_footprint_bytes()

                row = ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(name, size=12)),
                        ft.DataCell(ft.Text(version, size=12)),
                        ft.DataCell(ft.Text(f"{footprint:,} B", size=12)),
                    ]
                )
                rows.append(row)
            except Exception:
                continue

        table = ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("Extension", weight=ft.FontWeight.W_600)),
                ft.DataColumn(ft.Text("Version", weight=ft.FontWeight.W_600)),
                ft.DataColumn(ft.Text("Footprint", weight=ft.FontWeight.W_600)),
            ],
            rows=rows,
            border=ft.border.all(1, GyroTheme.BORDER),
            border_radius=8,
        )

        return self._create_info_card("Loaded Extensions", content=[table])

    def _create_info_card(
        self, title: str, info: Optional[List[tuple]] = None, content: Optional[List] = None
    ) -> ft.Container:
        """Create an information card"""
        card_content = []

        # Title
        card_content.append(ft.Text(title, weight=ft.FontWeight.W_600, size=14))

        # Info pairs
        if info:
            for text, color in info:
                card_content.append(ft.Text(text, size=12, color=color))

        # Custom content
        if content:
            card_content.extend(content)

        return ft.Container(
            content=ft.Column(controls=card_content, spacing=8),
            padding=15,
            bgcolor=GyroTheme.SURFACE,
            border_radius=8,
            border=ft.border.all(1, GyroTheme.BORDER),
        )

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
        else:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            return f"{days}d {hours}h"

    def _on_row_hover(self, e):
        """Handle row hover effect"""
        if e.data == "true":
            e.control.bgcolor = GyroTheme.HOVER_BG
        else:
            e.control.bgcolor = None
        e.control.update()

    def _select_all_curriculum(self, select: bool):
        """Select or deselect all curriculum checkboxes"""
        for checkbox in self.curriculum_checkboxes:
            checkbox.value = select
        self.update()

    def _download_and_ingest_curriculum(self, e):
        """Download and ingest selected curriculum resources"""
        selected = [
            resource
            for resource, checkbox in zip(self.curriculum_resources, self.curriculum_checkboxes)
            if checkbox.value
        ]

        if not selected:
            self._show_message("Please select at least one resource", error=True)
            return

        self.curriculum_progress.visible = True
        self.curriculum_progress.value = 0
        self.curriculum_status.value = "Starting download..."
        self.curriculum_status.update()
        self.curriculum_progress.update()

        def download_worker():
            try:
                for i, resource in enumerate(selected):

                    def update_status():
                        self.curriculum_status.value = f"Downloading {resource.name}..."
                        self.curriculum_status.update()

                    if self.app and hasattr(self.app, "_ui_update_queue"):
                        self.app._ui_update_queue.put(update_status)
                    else:
                        update_status()

                    # Find matching key in CURRICULUM_RESOURCES
                    key = None
                    for k, v in CURRICULUM_RESOURCES.items():
                        if resource.key == k or v["name"].lower() in resource.name.lower():
                            key = k
                            break

                    if not key:

                        def update_skip():
                            self.curriculum_status.value = f"Skipping {resource.name} - not found"
                            self.curriculum_status.update()

                        if self.app and hasattr(self.app, "_ui_update_queue"):
                            self.app._ui_update_queue.put(update_skip)
                        else:
                            update_skip()
                        continue

                    # Progress callback
                    def progress_cb(downloaded, total):
                        progress = (i + (downloaded / (total or 1))) / len(selected)

                        def update_progress():
                            self.curriculum_progress.value = progress
                            self.curriculum_progress.update()

                        if self.app and hasattr(self.app, "_ui_update_queue"):
                            self.app._ui_update_queue.put(update_progress)
                        else:
                            update_progress()

                    # Ingest resource
                    ingest_curriculum_resource(key, "./data/curriculum", progress_cb=progress_cb)

                    # Update progress
                    def update_progress_final():
                        self.curriculum_progress.value = (i + 1) / len(selected)
                        self.curriculum_progress.update()

                    if self.app and hasattr(self.app, "_ui_update_queue"):
                        self.app._ui_update_queue.put(update_progress_final)
                    else:
                        update_progress_final()

                # Complete
                def update_complete():
                    self.curriculum_status.value = (
                        f"Successfully ingested {len(selected)} resources"
                    )
                    self.curriculum_status.color = GyroTheme.ACCENT
                    self.curriculum_status.update()

                if self.app and hasattr(self.app, "_ui_update_queue"):
                    self.app._ui_update_queue.put(update_complete)
                else:
                    update_complete()

            except Exception as ex:

                def update_error():
                    self.curriculum_status.value = f"Error: {str(ex)}"
                    self.curriculum_status.color = GyroTheme.ERROR
                    self.curriculum_status.update()

                if self.app and hasattr(self.app, "_ui_update_queue"):
                    self.app._ui_update_queue.put(update_error)
                else:
                    update_error()
            finally:
                # Hide progress after delay
                time.sleep(2)

                def hide_progress():
                    self.curriculum_progress.visible = False
                    self.curriculum_progress.update()

                if self.app and hasattr(self.app, "_ui_update_queue"):
                    self.app._ui_update_queue.put(hide_progress)
                else:
                    hide_progress()

        # Run in background thread
        threading.Thread(target=download_worker, daemon=True).start()

    def _export_knowledge(self, e):
        """Export knowledge package"""
        if self.export_picker is not None:
            filename = (
                f"knowledge_{self.extension_manager.get_knowledge_id()[:8]}_{int(time.time())}.gyro"
            )
            self.export_picker.save_file(
                dialog_title="Export Knowledge Package",
                file_name=filename,
                allowed_extensions=["gyro"],
            )

    def _on_export_result(self, e: ft.FilePickerResultEvent):
        """Handle export file picker result"""
        if not e.path:
            return

        try:
            self.extension_manager.export_knowledge(e.path)
            self._show_message(f"Knowledge exported to: {os.path.basename(e.path)}")
        except Exception as ex:
            self._show_message(f"Export failed: {str(ex)}", error=True)

    def _import_knowledge(self, e):
        """Import knowledge package"""
        if self.import_picker is not None:
            self.import_picker.pick_files(
                dialog_title="Import Knowledge Package",
                allowed_extensions=["gyro"],
                allow_multiple=False,
            )

    def _on_import_result(self, e: ft.FilePickerResultEvent):
        """Handle import file picker result"""
        if not e.files:
            return
        try:
            file_path = e.files[0].path
            new_id = self.extension_manager.import_knowledge(file_path)
            if self.app is not None:
                self.app.reload_current_context()
            self._show_message(f"Knowledge imported. New ID: {new_id[:8]}...")
        except Exception as ex:
            self._show_message(f"Import failed: {str(ex)}", error=True)

    def _fork_knowledge(self, e):
        """Fork current knowledge"""
        try:
            new_id = self.extension_manager.fork_knowledge()
            if self.app is not None:
                self.app.reload_current_context()
            self._show_message(f"Knowledge forked. New ID: {new_id[:8]}...")
        except Exception as ex:
            self._show_message(f"Fork failed: {str(ex)}", error=True)

    def _copy_to_clipboard(self, value: str):
        """Copy value to clipboard"""
        if PYPERCLIP_AVAILABLE:
            try:
                pyperclip.copy(value)
                self._show_message("Copied to clipboard")
                return
            except Exception:
                pass
        # Fallback: show the value in a dialog for manual copying
        copy_dialog = ft.AlertDialog(
            title=ft.Text("Copy Value"),
            content=ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text("Copy this value manually:", size=12),
                        ft.TextField(
                            value=value,
                            read_only=True,
                            multiline=True,
                            min_lines=2,
                            max_lines=4,
                        ),
                    ],
                    spacing=10,
                ),
                width=400,
            ),
            actions=[
                ft.TextButton("Close", on_click=lambda _: self._close_subdialog(copy_dialog)),
            ],
        )
        if self.page is not None:
            self.page.dialog = copy_dialog
            copy_dialog.open = True
            self.page.update()

    def _export_key(self, key: str):
        """Export encryption key"""
        # Create a simple dialog showing the key
        key_dialog = ft.AlertDialog(
            title=ft.Text("Encryption Key"),
            content=ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text(
                            "Store this key securely. You'll need it to decrypt your data.", size=12
                        ),
                        ft.TextField(
                            value=key,
                            read_only=True,
                            multiline=True,
                            min_lines=3,
                            max_lines=5,
                        ),
                        ft.Row(
                            controls=[
                                ft.TextButton(
                                    "Copy",
                                    on_click=lambda _: self._copy_to_clipboard(key),
                                ),
                            ],
                        ),
                    ],
                    spacing=10,
                ),
                width=400,
            ),
            actions=[
                ft.TextButton("Close", on_click=lambda _: self._close_subdialog(key_dialog)),
            ],
        )

        if self.page is not None:
            self.page.dialog = key_dialog
            key_dialog.open = True
            self.page.update()

    def _change_key(self, e):
        """Change encryption key"""
        new_key_field = ft.TextField(
            label="New Key",
            password=True,
            can_reveal_password=True,
            helper_text="Minimum 16 characters",
            on_change=lambda e: self._validate_min_length(e, 16),
        )
        confirm_key_field = ft.TextField(
            label="Confirm Key",
            password=True,
            can_reveal_password=True,
        )

        def on_change(e):
            new_key = new_key_field.value
            confirm_key = confirm_key_field.value
            if not new_key or not confirm_key:
                self._show_message("Please enter and confirm the new key", error=True)
                return
            if new_key != confirm_key:
                self._show_message("Keys do not match", error=True)
                return
            if len(new_key) < 16:
                self._show_message("Key must be at least 16 characters", error=True)
                return
            self._show_message("Key change not yet implemented", error=True)

        change_dialog = ft.AlertDialog(
            title=ft.Text("Change Encryption Key"),
            content=ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text(
                            "Warning: Changing the key will re-encrypt all data. This may take some time.",
                            size=12,
                            color=GyroTheme.ERROR,
                        ),
                        new_key_field,
                        confirm_key_field,
                    ],
                    spacing=10,
                ),
                width=350,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: self._close_subdialog(change_dialog)),
                ft.FilledButton("Change Key", on_click=on_change),
            ],
        )
        if self.page is not None:
            self.page.dialog = change_dialog
            change_dialog.open = True
            self.page.update()

    def _enable_encryption(self, e):
        """Enable encryption"""
        self._show_message("Encryption enablement not yet implemented", error=True)

    def _close_subdialog(self, dialog):
        """Close a sub-dialog and return to settings"""
        if dialog:
            dialog.open = False
        if self.page:
            if getattr(self.page, "dialog", None) is dialog:
                self.page.dialog = None
            self.page.update()

    def _close(self, e):
        """Close the settings dialog"""
        if self.on_close:
            self.on_close()
        if self.page:
            self.page.dialog = None
            self.page.update()

    def _show_message(self, message: str, error: bool = False):
        """Show a message to the user"""
        if self.page:
            self.page.snack_bar = ft.SnackBar(
                content=ft.Text(message),
                bgcolor=GyroTheme.ERROR if error else GyroTheme.ACCENT,
                open=True,
            )
            self.page.update()

    def _validate_min_length(self, e, min_len):
        """Validate minimum length for text fields"""
        if len(e.control.value or "") < min_len:
            e.control.helper_text = f"Minimum {min_len} characters required"
            e.control.helper_text_color = GyroTheme.ERROR
        else:
            e.control.helper_text = ""
        e.control.update()

    def _get_session_count(self) -> int:
        """Get the number of active sessions"""
        try:
            from core.gyro_api import list_active_sessions

            return len(list_active_sessions())
        except Exception:
            return 0


def main():
    """Entry point for the application"""
    app = GyroApp()
    ft.app(target=app.main)


if __name__ == "__main__":
    main()
