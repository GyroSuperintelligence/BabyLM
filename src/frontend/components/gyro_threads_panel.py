# src/frontend/components/gyro_threads_panel.py
import flet as ft
from typing import Callable, Optional, List, Dict
import uuid
from datetime import datetime
from ..assets.styles.theme import GyroTheme


class GyroThreadsPanel(ft.Control):
    """Thread list with folder support and Apple-like design"""

    def __init__(self, on_thread_select: Callable):
        super().__init__()
        self.on_thread_select = on_thread_select
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

        # Add sample threads
        self._add_sample_threads()

        return ft.Column(controls=[header, self.thread_list], spacing=0, expand=True)

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

    async def _create_new_thread(self, e):
        """Create a new conversation thread"""
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
        await self.update_async()

        # Select the new thread
        await self._on_thread_click(thread_id)

    async def _on_thread_click(self, thread_id: str):
        """Handle thread selection"""
        # Update selection state
        if self.selected_thread_id:
            self.threads[self.selected_thread_id].selected = False

        self.selected_thread_id = thread_id
        self.threads[thread_id].selected = True

        await self.update_async()
        await self.on_thread_select(thread_id)

    async def _on_thread_delete(self, thread_id: str):
        """Handle thread deletion"""
        if thread_id in self.threads:
            thread_item = self.threads[thread_id]
            self.thread_list.controls.remove(thread_item)
            del self.threads[thread_id]
            await self.update_async()

    async def _on_search(self, e):
        """Filter threads based on search query"""
        query = e.control.value.lower()

        for thread_id, thread_item in self.threads.items():
            if query:
                visible = query in thread_item.title.lower() or query in thread_item.preview.lower()
                thread_item.visible = visible
            else:
                thread_item.visible = True

        await self.update_async()


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
        content = ft.Column(
            controls=[
                ft.Text(
                    self.title,
                    size=14,
                    weight=ft.FontWeight.W_500,
                    color=GyroTheme.TEXT_PRIMARY,
                    max_lines=1,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
                ft.Text(
                    self.preview,
                    size=12,
                    color=GyroTheme.TEXT_SECONDARY,
                    max_lines=1,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
                ft.Text(self._format_timestamp(), size=11, color=GyroTheme.TEXT_TERTIARY),
            ],
            spacing=2,
        )

        # Container with hover effects
        self.container = ft.Container(
            content=ft.Row(
                controls=[content, self.delete_btn], alignment=ft.MainAxisAlignment.SPACE_BETWEEN
            ),
            padding=ft.padding.all(12),
            border_radius=8,
            on_hover=self._on_hover,
            on_click=lambda _: self.on_click(self.thread_id),
            animate=ft.animation.Animation(200, ft.AnimationCurve.EASE_IN_OUT),
        )

        self._update_style()
        return self.container

    def _update_style(self):
        """Update container style based on selection state"""
        if self.selected:
            self.container.bgcolor = GyroTheme.ACCENT
            self.container.border = ft.border.all(1, GyroTheme.ACCENT)
        else:
            self.container.bgcolor = None
            self.container.border = None

    def _on_hover(self, e):
        """Handle hover state"""
        if e.data == "true":
            if not self.selected:
                self.container.bgcolor = GyroTheme.HOVER_BG
            self.delete_btn.visible = True
        else:
            if not self.selected:
                self.container.bgcolor = None
            self.delete_btn.visible = False
        self.update()

    def _format_timestamp(self):
        """Format timestamp in Apple style"""
        now = datetime.now()
        diff = now - self.timestamp

        if diff.days > 7:
            return self.timestamp.strftime("%b %d")
        elif diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "Just now"


class FolderItem(ft.UserControl):
    """Folder item with nesting support"""

    def __init__(self, folder_id: str, name: str, parent_id: Optional[str] = None):
        super().__init__()
        self.folder_id = folder_id
        self.name = name
        self.parent_id = parent_id
        self.expanded = True
        self.children: List[ThreadItem] = []
        self.subfolders: List[FolderItem] = []

    def build(self):
        # Folder header
        self.expand_icon = ft.Icon(
            ft.icons.KEYBOARD_ARROW_DOWN if self.expanded else ft.icons.KEYBOARD_ARROW_RIGHT,
            size=16,
            color=GyroTheme.TEXT_SECONDARY,
        )

        header = ft.Container(
            content=ft.Row(
                controls=[
                    self.expand_icon,
                    ft.Icon(ft.icons.FOLDER, size=16, color=GyroTheme.ACCENT),
                    ft.Text(
                        self.name, size=13, weight=ft.FontWeight.W_500, color=GyroTheme.TEXT_PRIMARY
                    ),
                ],
                spacing=5,
            ),
            padding=ft.padding.symmetric(horizontal=12, vertical=8),
            on_click=self._toggle_expand,
            on_hover=self._on_hover,
            border_radius=6,
        )

        # Children container
        self.children_container = ft.Column(
            controls=self.children + self.subfolders, visible=self.expanded, spacing=2
        )

        # Add indentation for nested items
        if self.parent_id:
            self.children_container.controls = [
                ft.Container(content=child, padding=ft.padding.only(left=20))
                for child in self.children_container.controls
            ]

        return ft.Column(controls=[header, self.children_container], spacing=0)

    def _toggle_expand(self, e):
        """Toggle folder expansion"""
        self.expanded = not self.expanded
        self.expand_icon.name = (
            ft.icons.KEYBOARD_ARROW_DOWN if self.expanded else ft.icons.KEYBOARD_ARROW_RIGHT
        )
        self.children_container.visible = self.expanded
        self.update()

    def _on_hover(self, e):
        """Handle hover state"""
        container = e.control
        if e.data == "true":
            container.bgcolor = GyroTheme.HOVER_BG
        else:
            container.bgcolor = None
        container.update()


class DraggableThreadItem(ThreadItem):
    """Thread item with drag and drop support"""

    def build(self):
        base_control = super().build()

        return ft.Draggable(
            group="thread",
            content=base_control,
            content_feedback=ft.Container(
                content=ft.Text(self.title, size=14),
                bgcolor=GyroTheme.ACCENT,
                padding=10,
                border_radius=8,
                opacity=0.8,
            ),
        )


class DroppableFolder(FolderItem):
    """Folder with drop target support"""

    def build(self):
        base_control = super().build()

        return ft.DragTarget(
            group="thread",
            content=base_control,
            on_accept=self._on_drop,
            on_hover=self._on_drag_hover,
        )

    def _on_drop(self, e):
        """Handle dropped thread"""
        # Move thread to this folder
        thread_id = e.data
        # Implementation for moving thread

    def _on_drag_hover(self, e):
        """Visual feedback during drag"""
        if e.data == "true":
            self.container.border = ft.border.all(2, GyroTheme.ACCENT)
        else:
            self.container.border = None
        self.update()


class ThreadContextMenu(ft.PopupMenuButton):
    """Context menu for thread operations"""

    def __init__(self, thread_id: str, on_rename: Callable, on_delete: Callable):
        super().__init__(
            items=[
                ft.PopupMenuItem(
                    text="Rename", icon=ft.icons.EDIT, on_click=lambda _: on_rename(thread_id)
                ),
                ft.PopupMenuItem(
                    text="Move to Folder",
                    icon=ft.icons.FOLDER,
                    on_click=lambda _: self._show_folder_dialog(thread_id),
                ),
                ft.PopupMenuItem(),  # Divider
                ft.PopupMenuItem(
                    text="Delete", icon=ft.icons.DELETE, on_click=lambda _: on_delete(thread_id)
                ),
            ]
        )
