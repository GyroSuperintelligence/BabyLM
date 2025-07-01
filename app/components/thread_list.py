"""
thread_list.py - Thread list sidebar component

Displays conversation threads with search and management.
"""

import flet as ft
from typing import Optional, Callable, List
from datetime import datetime

from state import AppState, Thread


class ThreadListItem(ft.UserControl):
    """Individual thread item in the list."""

    def __init__(self, thread: Thread, is_selected: bool, on_click: Callable[[str], None]):
        super().__init__()
        self.thread = thread
        self.is_selected = is_selected
        self.on_click = on_click

    def build(self):
        # Format last update time
        now = datetime.now()
        diff = now - self.thread.updated_at

        if diff.days > 0:
            time_str = self.thread.updated_at.strftime("%b %d")
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            time_str = f"{hours}h ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            time_str = f"{minutes}m ago"
        else:
            time_str = "Just now"

        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Text(
                                self.thread.title,
                                size=14,
                                weight=(
                                    ft.FontWeight.W_500
                                    if self.is_selected
                                    else ft.FontWeight.NORMAL
                                ),
                                color="#FFFFFF" if self.is_selected else "#E5E5E7",
                                max_lines=1,
                                overflow=ft.TextOverflow.ELLIPSIS,
                                expand=True,
                            ),
                            ft.Text(time_str, size=11, color="#8E8E93"),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Text(f"{self.thread.message_count} messages", size=12, color="#8E8E93"),
                ],
                spacing=2,
            ),
            padding=ft.padding.all(12),
            bgcolor="#0A84FF" if self.is_selected else None,
            border_radius=8,
            on_click=lambda _: self.on_click(self.thread.id),
            animate=ft.animation.Animation(150, ft.AnimationCurve.EASE_IN_OUT),
        )


class ThreadList(ft.UserControl):
    """
    Thread list sidebar with search and new thread button.
    """

    def __init__(
        self,
        state: AppState,
        on_thread_select: Callable[[str], None],
        on_new_thread: Callable[[], None],
    ):
        super().__init__()
        self.state = state
        self.on_thread_select = on_thread_select
        self.on_new_thread = on_new_thread
        self.search_query = ""

    def build(self):
        # Search field
        search_field = ft.TextField(
            hint_text="Search threads...",
            prefix_icon=ft.icons.SEARCH,
            border_radius=8,
            height=40,
            bgcolor="#000000",
            border_color="#38383A",
            focused_border_color="#0A84FF",
            hint_style=ft.TextStyle(color="#8E8E93"),
            text_style=ft.TextStyle(color="#FFFFFF", size=14),
            on_change=self._on_search_change,
            filled=True,
        )

        # New thread button
        new_thread_button = ft.ElevatedButton(
            text="New Thread",
            icon=ft.icons.ADD,
            on_click=lambda _: self.on_new_thread(),
            style=ft.ButtonStyle(
                color="#FFFFFF",
                bgcolor="#0A84FF",
                shape=ft.RoundedRectangleBorder(radius=8),
                padding=ft.padding.symmetric(horizontal=16, vertical=8),
            ),
            height=40,
        )

        # Thread list
        thread_items = self._build_thread_items()

        thread_list_view = ft.ListView(
            controls=thread_items, spacing=4, padding=ft.padding.symmetric(vertical=10), expand=True
        )

        # Layout
        return ft.Container(
            content=ft.Column(
                controls=[
                    # Header
                    ft.Container(
                        content=ft.Text(
                            "Threads", size=20, weight=ft.FontWeight.W_600, color="#FFFFFF"
                        ),
                        padding=ft.padding.all(16),
                    ),
                    # Search
                    ft.Container(content=search_field, padding=ft.padding.symmetric(horizontal=16)),
                    # New thread button
                    ft.Container(content=new_thread_button, padding=ft.padding.all(16)),
                    # Thread list
                    ft.Container(
                        content=thread_list_view,
                        expand=True,
                        padding=ft.padding.symmetric(horizontal=8),
                    ),
                ],
                spacing=0,
            ),
            bgcolor="#1C1C1E",
            expand=True,
        )

    def _build_thread_items(self) -> List[ft.Control]:
        """Build filtered thread items."""
        threads = list(self.state.threads.values())

        # Filter by search query
        if self.search_query:
            query_lower = self.search_query.lower()
            threads = [t for t in threads if query_lower in t.title.lower()]

        # Sort by updated time (newest first)
        threads.sort(key=lambda t: t.updated_at, reverse=True)

        # Build items
        items = []
        for thread in threads:
            is_selected = thread.id == self.state.current_thread_id
            items.append(
                ThreadListItem(
                    thread=thread, is_selected=is_selected, on_click=self.on_thread_select
                )
            )

        # Empty state
        if not items:
            items.append(
                ft.Container(
                    content=ft.Text(
                        "No threads found" if self.search_query else "No threads yet",
                        size=14,
                        color="#8E8E93",
                        text_align=ft.TextAlign.CENTER,
                    ),
                    padding=ft.padding.all(20),
                    alignment=ft.alignment.center,
                )
            )

        return items

    def _on_search_change(self, e):
        """Handle search query change."""
        self.search_query = e.control.value
        self.update()

    def update(self):
        """Update the thread list."""
        # Rebuild the entire control to reflect state changes
        if hasattr(self, "controls") and self.controls:
            column = getattr(self.controls[0], "content", None)
            if column and hasattr(column, "controls") and len(column.controls) > 3:
                thread_container = column.controls[3]
                if hasattr(thread_container, "content") and hasattr(
                    thread_container.content, "controls"
                ):
                    thread_container.content.controls = self._build_thread_items()
        super().update()
