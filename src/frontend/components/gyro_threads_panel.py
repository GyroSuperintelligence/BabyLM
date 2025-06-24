# src/frontend/components/gyro_threads_panel.py
import flet as ft
from typing import Callable, Optional, Dict, List
import uuid
from datetime import datetime
from ..assets.styles.theme import GyroTheme
from core.gyro_api import link_session_to_knowledge


class GyroThreadsPanel(ft.UserControl):
    """Thread list with folder support and Apple-like design"""

    def __init__(self, on_thread_select: Callable, extension_manager=None):
        super().__init__()
        self.on_thread_select = on_thread_select
        self.extension_manager = extension_manager
        self.threads: Dict[str, ThreadItem] = {}
        self.folders: Dict[str, FolderItem] = {}
        self.selected_thread_id: Optional[str] = None
        self.knowledge_ids: List[str] = []
        self.knowledge_dropdown: Optional[ft.Dropdown] = None
        self._init_knowledge_list()
        self._snackbar = None
        self._pending_delete_thread_id = None

    def _init_knowledge_list(self):
        """Initialize the list of available knowledge packages from backend."""
        if self.extension_manager:
            try:
                # Implement a local listing using the filesystem if not present in API
                import os
                from pathlib import Path

                knowledge_root = Path("data/knowledge")
                self.knowledge_ids = [
                    d.name
                    for d in knowledge_root.iterdir()
                    if d.is_dir() and (d / "knowledge.meta.json").exists()
                ]
            except Exception:
                self.knowledge_ids = []
        else:
            self.knowledge_ids = []

    def build(self):
        # Search bar
        self.search_field = ft.TextField(
            hint_text="Search conversations...",
            prefix_icon=ft.icons.SEARCH,
            border=ft.InputBorder.NONE,
            filled=True,
            bgcolor=GyroTheme.INPUT_BG,
            text_size=14,
            height=36,
            border_radius=8,
            on_change=self._on_search,
        )

        # New thread button
        new_thread_btn = ft.IconButton(
            icon=ft.icons.EDIT_OUTLINED,
            icon_color=GyroTheme.ACCENT,
            icon_size=20,
            tooltip="New conversation",
            on_click=self._create_new_thread,
        )

        # Knowledge selector dropdown
        self.knowledge_dropdown = ft.Dropdown(
            label="Knowledge",
            options=[ft.dropdown.Option(kid, f"{kid[:8]}...") for kid in self.knowledge_ids],
            value=self.knowledge_ids[0] if self.knowledge_ids else None,
            on_change=self._on_knowledge_select,
            width=220,
            filled=True,
            bgcolor=GyroTheme.INPUT_BG,
            border_radius=8,
        )

        # Header
        header = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Text(
                                "Conversations",
                                size=16,
                                weight=ft.FontWeight.W_600,
                                color=GyroTheme.TEXT_PRIMARY,
                            ),
                            new_thread_btn,
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    self.knowledge_dropdown,
                    self.search_field,
                ],
                spacing=10,
            ),
            padding=ft.padding.all(15),
            border=ft.border.only(bottom=ft.BorderSide(1, GyroTheme.BORDER)),
        )

        # Thread list container
        self.thread_list = ft.ListView(spacing=2, padding=ft.padding.all(10), expand=True)

        # Load existing sessions if extension manager is available
        if self.extension_manager:
            self._load_existing_sessions()
        else:
            # Add sample threads for demonstration
            self._add_sample_threads()

        return ft.Column(controls=[header, self.thread_list], spacing=0, expand=True)

    def _load_existing_sessions(self):
        """Load all active GyroSI sessions from backend"""
        try:
            from core.gyro_api import list_active_sessions, get_session_info

            session_ids = list_active_sessions()
            for session_id in session_ids:
                info = get_session_info(session_id)
                knowledge_id = info.get("knowledge_id", "unknown")
                phase = info.get("phase", 0)
                thread_item = ThreadItem(
                    thread_id=session_id,
                    title=f"Session {session_id[:8]}...",
                    preview=f"Knowledge: {knowledge_id[:8]}... | Phase: {phase}/47",
                    timestamp=datetime.now(),
                    on_click=self._on_thread_click,
                    on_delete=self._on_thread_delete,
                )
                self.threads[session_id] = thread_item
                self.thread_list.controls.append(thread_item)
            if session_ids:
                self._on_thread_click(session_ids[0])
        except Exception as e:
            self._add_sample_threads()

    def _add_sample_threads(self):
        """Add sample threads for demonstration"""
        sample_threads = [
            ("Learning Session 1", "Tell me about quantum physics..."),
            ("Document Analysis", "Analyzing uploaded document..."),
            ("Creative Writing", "Write a story about..."),
        ]

        for title, preview in sample_threads:
            thread_id = str(uuid.uuid4())
            thread_item = ThreadItem(
                thread_id=thread_id,
                title=title,
                preview=preview,
                timestamp=datetime.now(),
                on_click=self._on_thread_click,
                on_delete=self._on_thread_delete,
            )
            self.threads[thread_id] = thread_item
            self.thread_list.controls.append(thread_item)

    def _create_new_thread(self, e):
        """Create a new conversation thread with GyroSI session"""
        try:
            if self.extension_manager:
                from core.gyro_api import initialize_session, get_session_info

                new_session_id = initialize_session()
                info = get_session_info(new_session_id)
                knowledge_id = info.get("knowledge_id", "unknown")
                phase = info.get("phase", 0)
                thread_item = ThreadItem(
                    thread_id=new_session_id,
                    title="New Conversation",
                    preview=f"Knowledge: {knowledge_id[:8]}... | Phase: {phase}/47",
                    timestamp=datetime.now(),
                    on_click=self._on_thread_click,
                    on_delete=self._on_thread_delete,
                )
                self.threads[new_session_id] = thread_item
                self.thread_list.controls.insert(0, thread_item)
                self.update()
                self._on_thread_click(new_session_id)
            else:
                thread_id = str(uuid.uuid4())
                thread_item = ThreadItem(
                    thread_id=thread_id,
                    title="New Conversation",
                    preview="Start typing...",
                    timestamp=datetime.now(),
                    on_click=self._on_thread_click,
                    on_delete=self._on_thread_delete,
                )
                self.threads[thread_id] = thread_item
                self.thread_list.controls.insert(0, thread_item)
                self.update()
                self._on_thread_click(thread_id)
        except Exception as e:
            print(f"Error creating new thread: {e}")

    def _on_thread_click(self, thread_id: str):
        """Handle thread selection"""
        # Update selection state
        if self.selected_thread_id and self.selected_thread_id in self.threads:
            self.threads[self.selected_thread_id].selected = False

        self.selected_thread_id = thread_id
        if thread_id in self.threads:
            self.threads[thread_id].selected = True
            self.update()
            try:
                self.on_thread_select(thread_id)
            except Exception as ex:
                self._show_message(f"Error loading session: {ex}", error=True)
        else:
            self._show_message("Session not found or has been deleted.", error=True)

    def _on_thread_delete(self, thread_id: str):
        """Show confirmation dialog before deleting a session/thread"""
        self._pending_delete_thread_id = thread_id
        confirm_dialog = ft.AlertDialog(
            title=ft.Text("Delete Session?", weight=ft.FontWeight.W_600),
            content=ft.Text(
                "Are you sure you want to delete this session? This action cannot be undone."
            ),
            actions=[
                ft.TextButton("Cancel", on_click=self._cancel_delete),
                ft.FilledButton(
                    "Delete",
                    on_click=self._confirm_delete,
                    style=ft.ButtonStyle(bgcolor=GyroTheme.ERROR, color="#fff"),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        if hasattr(self, "page") and self.page:
            self.page.dialog = confirm_dialog
            self.page.update()
        else:
            self._snackbar = confirm_dialog
            self.update()

    def _cancel_delete(self, e):
        if hasattr(self, "page") and self.page:
            self.page.dialog = None
            self.page.update()
        self._pending_delete_thread_id = None

    def _confirm_delete(self, e):
        if hasattr(self, "page") and self.page:
            self.page.dialog = None
            self.page.update()
        thread_id = self._pending_delete_thread_id
        self._pending_delete_thread_id = None
        try:
            if thread_id in self.threads:
                from core.gyro_api import shutdown_session

                shutdown_session(thread_id)
                thread_item = self.threads[thread_id]
                self.thread_list.controls.remove(thread_item)
                del self.threads[thread_id]
                if self.selected_thread_id == thread_id:
                    self.selected_thread_id = None
                self.update()
                self._show_message("Session deleted.", info=True)
        except Exception as ex:
            self._show_message(f"Error deleting session: {ex}", error=True)

    def _on_search(self, e):
        """Filter threads based on search query"""
        query = e.control.value.lower()

        for thread_id, thread_item in self.threads.items():
            if query:
                visible = query in thread_item.title.lower() or query in thread_item.preview.lower()
                thread_item.visible = visible
            else:
                thread_item.visible = True

        self.update()

    def update_thread_preview(self, thread_id: str, preview: str):
        """Update the preview text for a thread"""
        if thread_id in self.threads:
            self.threads[thread_id].update_preview(preview)
            self.update()

    def _on_knowledge_select(self, e):
        """Handle knowledge selection from dropdown and link to current session."""
        selected_knowledge_id = e.control.value
        if self.selected_thread_id and self.extension_manager:
            try:
                link_session_to_knowledge(self.selected_thread_id, selected_knowledge_id)
                self.update_thread_preview(
                    self.selected_thread_id,
                    f"Knowledge: {selected_knowledge_id[:8]}... | Phase: {self.extension_manager.engine.phase}/47",
                )
                if hasattr(self, "on_thread_select"):
                    self.on_thread_select(self.selected_thread_id)
                self._show_message("Knowledge switched.", info=True)
            except Exception as ex:
                self._show_message(f"Error switching knowledge: {ex}", error=True)

    def refresh_knowledge_list(self):
        """Refresh the list of available knowledge packages from backend."""
        self._init_knowledge_list()
        if self.knowledge_dropdown:
            self.knowledge_dropdown.options = [
                ft.dropdown.Option(kid, f"{kid[:8]}...") for kid in self.knowledge_ids
            ]
            self.knowledge_dropdown.value = self.knowledge_ids[0] if self.knowledge_ids else None
            self.knowledge_dropdown.update()

    def _show_message(self, msg, error=False, info=False):
        # Show a Flet snackbar for user feedback
        color = (
            GyroTheme.ERROR
            if error
            else (GyroTheme.TEXT_SECONDARY if info else GyroTheme.TEXT_ON_ACCENT)
        )
        bgcolor = GyroTheme.ERROR if error else (GyroTheme.SURFACE if info else GyroTheme.ACCENT)
        snackbar = ft.SnackBar(
            content=ft.Text(msg, color=color),
            bgcolor=bgcolor,
            open=True,
            duration=3000,
        )
        if hasattr(self, "page") and self.page:
            self.page.snack_bar = snackbar
            self.page.update()
        else:
            self._snackbar = snackbar
            self.update()


class ThreadItem(ft.UserControl):
    """Individual thread item with Apple-like design"""

    def __init__(
        self,
        thread_id: str,
        title: str,
        preview: str,
        timestamp: datetime,
        on_click: Callable,
        on_delete: Callable,
    ):
        super().__init__()
        self.thread_id = thread_id
        self.title = title
        self.preview = preview
        self.timestamp = timestamp
        self.on_click = on_click
        self.on_delete = on_delete
        self.selected = False

    def build(self):
        # Delete button (shown on hover)
        self.delete_btn = ft.IconButton(
            icon=ft.icons.DELETE_OUTLINE,
            icon_size=16,
            icon_color=GyroTheme.TEXT_TERTIARY,
            visible=False,
            on_click=lambda _: self.on_delete(self.thread_id),
        )

        # Thread content
        self.content_text = ft.Text(
            self.title,
            size=14,
            weight=ft.FontWeight.W_500,
            color=GyroTheme.TEXT_PRIMARY,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )

        self.preview_text = ft.Text(
            self.preview,
            size=12,
            color=GyroTheme.TEXT_SECONDARY,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
        )

        self.timestamp_text = ft.Text(
            self._format_timestamp(), size=11, color=GyroTheme.TEXT_TERTIARY
        )

        content = ft.Column(
            controls=[
                self.content_text,
                self.preview_text,
                self.timestamp_text,
            ],
            spacing=2,
        )

        # Container with hover effects
        self.container = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Container(content=content, expand=True),
                    self.delete_btn,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=ft.padding.all(12),
            border_radius=8,
            on_click=lambda _: self.on_click(self.thread_id),
            on_hover=self._on_hover,
        )

        self._update_style()
        return self.container

    def _update_style(self):
        """Update visual style based on selection state"""
        if self.selected:
            self.container.bgcolor = GyroTheme.ACCENT
            self.content_text.color = GyroTheme.TEXT_ON_ACCENT
            self.preview_text.color = "#FFFFFF80"
            self.timestamp_text.color = "#FFFFFF60"
        else:
            self.container.bgcolor = None
            self.content_text.color = GyroTheme.TEXT_PRIMARY
            self.preview_text.color = GyroTheme.TEXT_SECONDARY
            self.timestamp_text.color = GyroTheme.TEXT_TERTIARY

    def _on_hover(self, e):
        """Handle hover effect"""
        if e.data == "true" and not self.selected:
            self.container.bgcolor = GyroTheme.HOVER_BG
            self.delete_btn.visible = True
        else:
            if not self.selected:
                self.container.bgcolor = None
            self.delete_btn.visible = False
        self.update()

    def _format_timestamp(self):
        """Format timestamp for display"""
        now = datetime.now()
        diff = now - self.timestamp

        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours}h ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes}m ago"
        else:
            return "Just now"

    def update_preview(self, preview: str):
        """Update the preview text"""
        self.preview = preview
        self.preview_text.value = preview
        self.update()

    @property
    def selected(self):
        return self._selected

    @selected.setter
    def selected(self, value: bool):
        self._selected = value
        if hasattr(self, "container"):
            self._update_style()


class FolderItem(ft.UserControl):
    """Folder item for organizing threads"""

    def __init__(self, folder_id: str, name: str):
        super().__init__()
        self.folder_id = folder_id
        self.name = name

    def build(self):
        return ft.Container(
            content=ft.Text(self.name, size=14, weight=ft.FontWeight.W_500),
            padding=ft.padding.all(12),
            bgcolor=GyroTheme.SURFACE,
            border_radius=8,
        )
