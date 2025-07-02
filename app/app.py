"""
app.py - Main app initialization for GyroSI Baby LM

Sets up the Flet app, routing, and global state.
"""

import flet as ft
from typing import Optional, Dict, Any
import os
import sys
from pathlib import Path

# Ensure parent directory is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from state import AppState
from utils.theme import apply_theme
from views.chat_view import ChatView
from views.settings_view import SettingsView
from views.dev_view import DevView
from components.top_bar import TopBar
from components.bottom_bar import BottomBar
from config import PAGE_TITLE, PAGE_WIDTH, PAGE_HEIGHT, PAGE_MIN_WIDTH, PAGE_MIN_HEIGHT, DATA_DIR


class GyroSIApp:
    """Main application class for GyroSI Baby LM."""

    def __init__(self):
        """Initialize app state."""
        self.state = AppState(base_path=DATA_DIR)
        self.current_view: Optional[ft.UserControl] = None
        self.page: Optional[ft.Page] = None

    def init_app(self, page: ft.Page):
        """Initialize the Flet app."""
        self.page = page

        # Set page properties
        page.title = PAGE_TITLE
        page.window_width = PAGE_WIDTH
        page.window_height = PAGE_HEIGHT
        page.window_min_width = PAGE_MIN_WIDTH
        page.window_min_height = PAGE_MIN_HEIGHT
        page.window_maximized = False
        page.window_resizable = True

        # Apply theme
        apply_theme(page)

        # Initialize components
        self.top_bar = TopBar(
            state=self.state,
            on_settings_click=self.show_settings_view,
            on_dev_click=self.show_dev_view,
            page=page,
        )

        self.bottom_bar = BottomBar(state=self.state, page=page)

        # Set up content area
        self.content_area = ft.Container(expand=True, content=None, padding=0)

        # Create main layout
        page.add(
            ft.Column([self.top_bar, self.content_area, self.bottom_bar], spacing=0, expand=True)
        )

        # Show initial view
        self.show_chat_view()

        # Try to initialize or load agent
        self._init_agent()

    def _init_agent(self):
        """Initialize the agent or create a new one, persisting UUID."""
        from s4_intelligence.g1_intelligence_in import initialize_system, create_agent

        try:
            # Ensure system is initialized
            initialize_system(base_path=self.state.base_path)

            # Path to persist agent UUID
            uuid_path = os.path.join(self.state.base_path, "agent_uuid.txt")
            agent_uuid = None
            if os.path.exists(uuid_path):
                with open(uuid_path, "r") as f:
                    agent_uuid = f.read().strip()

            if not agent_uuid:
                agent_uuid = create_agent(base_path=self.state.base_path)
                with open(uuid_path, "w") as f:
                    f.write(agent_uuid)

                self.state.set_agent(agent_uuid)
        except Exception as e:
            self.state.error_message = f"Failed to initialize system: {str(e)}"

    def set_view(self, view: ft.UserControl):
        """Set the current view in the content area."""
        self.current_view = view
        self.content_area.content = view
        if self.page:
            self.page.update()

    def show_chat_view(self):
        """Show the main chat view."""
        if self.page is None:
            return
        chat_view = ChatView(state=self.state, page=self.page)
        self.set_view(chat_view)

    def show_settings_view(self):
        """Show the settings view."""
        if self.page is None:
            return
        settings_view = SettingsView(state=self.state, page=self.page, on_back=self.show_chat_view)
        self.set_view(settings_view)

    def show_dev_view(self):
        """Show the developer tools view."""
        if self.page is None:
            return
        dev_view = DevView(state=self.state, page=self.page, on_back=self.show_chat_view)
        self.set_view(dev_view)


def main(page: ft.Page):
    """Main entry point for the Flet app."""
    app = GyroSIApp()
    app.init_app(page)
