# src/frontend/components/gyro_threads_panel.py
import flet as ft
from typing import Callable, Optional, Dict
import uuid
from datetime import datetime
from ..assets.styles.theme import GyroTheme


class GyroThreadsPanel(ft.UserControl):
    """Thread list with folder support and Apple-like design"""

    def __init__(self, on_thread_select: Callable, extension_manager=None):
        super().__init__()
        self.on_thread_select = on_thread_select
        self.extension_manager = extension_manager
        self.threads: Dict[str, ThreadItem] = {}
        self.folders: Dict[str, FolderItem] = {}
        self.selected_thread_id: Optional[str] = None

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
        """Load existing GyroSI sessions from storage"""
        try:
            # Get current session info
            current_session_id = self.extension_manager.get_session_id()
            knowledge_id = self.extension_manager.get_knowledge_id()

            # Create thread for current session
            thread_item = ThreadItem(
                thread_id=current_session_id,
                title="Current Session",
                preview=f"Knowledge: {knowledge_id[:8]}...",
                timestamp=datetime.now(),
                on_click=self._on_thread_click,
                on_delete=self._on_thread_delete,
            )
            self.threads[current_session_id] = thread_item
            self.thread_list.controls.append(thread_item)

            # Select current session
            self._on_thread_click(current_session_id)

        except Exception as e:
            # Fallback to sample threads if loading fails
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
                # Create new GyroSI session
                from core.gyro_api import initialize_session

                new_session_id = initialize_session()

                thread_item = ThreadItem(
                    thread_id=new_session_id,
                    title="New Conversation",
                    preview="Start typing...",
                    timestamp=datetime.now(),
                    on_click=self._on_thread_click,
                    on_delete=self._on_thread_delete,
                )

                self.threads[new_session_id] = thread_item
                self.thread_list.controls.insert(0, thread_item)
                self.update()

                # Select the new thread
                self._on_thread_click(new_session_id)
            else:
                # Fallback for demo mode
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

                # Select the new thread
                self._on_thread_click(thread_id)

        except Exception as e:
            # Show error in UI
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
        self.on_thread_select(thread_id)

    def _on_thread_delete(self, thread_id: str):
        """Handle thread deletion"""
        try:
            if thread_id in self.threads:
                thread_item = self.threads[thread_id]
                self.thread_list.controls.remove(thread_item)
                del self.threads[thread_id]

                # If this was the selected thread, clear selection
                if self.selected_thread_id == thread_id:
                    self.selected_thread_id = None

                self.update()

                # TODO: Implement proper session cleanup via gyro_api
                # from core.gyro_api import shutdown_session
                # shutdown_session(thread_id)

        except Exception as e:
            print(f"Error deleting thread: {e}")

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
