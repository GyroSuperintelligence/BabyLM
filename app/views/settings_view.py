"""
settings_view.py - Settings management view for GyroSI Baby LM

Handles general settings, agent management, and format operations.
"""

import flet as ft
from typing import Optional, Callable, Dict, Any
import json
from datetime import datetime
import os

from state import AppState
from components.common import Section, SettingRow, ActionButton


class SettingsView(ft.UserControl):
    """
    Settings view with sections for general settings, agent, and agency management.
    """

    def __init__(self, state: AppState, page: ft.Page, on_back: Callable[[], None]):
        super().__init__()
        self.state = state
        self.page = page
        self.on_back = on_back

    def build(self):
        # Header with back button
        header = ft.Container(
            content=ft.Row(
                controls=[
                    ft.IconButton(
                        icon=ft.icons.ARROW_BACK,
                        icon_color="#0A84FF",
                        on_click=lambda _: self.on_back(),
                    ),
                    ft.Text("Settings", size=20, weight=ft.FontWeight.W_600, color="#FFFFFF"),
                ]
            ),
            padding=ft.padding.all(10),
            border=ft.border.only(bottom=ft.BorderSide(1, "#38383A")),
        )

        # General settings section
        general_section = Section(
            title="General",
            controls=[
                SettingRow(
                    title="Encryption",
                    subtitle="Protect agent data with encryption",
                    control=ft.CupertinoSwitch(
                        value=self.state.settings.get("encryption_enabled", True),
                        active_color="#30D158",
                        on_change=lambda e: self._update_setting(
                            "encryption_enabled", e.control.value
                        ),
                    ),
                ),
                SettingRow(
                    title="Auto-save",
                    subtitle="Automatically save conversations",
                    control=ft.CupertinoSwitch(
                        value=self.state.settings.get("auto_save", True),
                        active_color="#30D158",
                        on_change=lambda e: self._update_setting("auto_save", e.control.value),
                    ),
                ),
                SettingRow(
                    title="Developer Mode",
                    subtitle="Show developer tools and metrics",
                    control=ft.CupertinoSwitch(
                        value=self.state.settings.get("show_dev_info", False),
                        active_color="#30D158",
                        on_change=lambda e: self._update_setting("show_dev_info", e.control.value),
                    ),
                ),
                SettingRow(
                    title="Message History",
                    subtitle=f"Keep last {self.state.settings.get('max_recent_messages', 250)} messages",
                    control=ft.TextButton(text="Change", on_click=self._show_history_dialog),
                ),
            ],
        )

        # Agent management section
        agent_section = Section(
            title="Agent Management",
            controls=[
                SettingRow(
                    title="Current Agent",
                    subtitle=(
                        self.state.agent_uuid[:8] + "..."
                        if self.state.agent_uuid
                        else "No agent selected"
                    ),
                    control=ft.TextButton(text="View", on_click=self._show_agent_info),
                ),
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ActionButton(
                                text="New Agent", icon=ft.icons.ADD, on_click=self._create_new_agent
                            ),
                            ActionButton(
                                text="Switch Agent",
                                icon=ft.icons.SWAP_HORIZ,
                                on_click=self._switch_agent,
                            ),
                        ],
                        spacing=10,
                    ),
                    padding=ft.padding.symmetric(horizontal=20, vertical=10),
                ),
            ],
        )

        # Scroll view with all sections
        return ft.Column(
            controls=[
                header,
                ft.Container(
                    content=ft.Column(
                        controls=[general_section, agent_section],
                        spacing=20,
                        scroll=ft.ScrollMode.AUTO,
                    ),
                    expand=True,
                    padding=ft.padding.all(20),
                ),
            ],
            spacing=0,
            expand=True,
        )

    def _update_setting(self, key: str, value: Any):
        """Update a setting value."""
        self.state.settings[key] = value
        self.state._notify_updates()
        if key == "show_dev_info" and self.page is not None:
            # Notify parent to update UI
            self.page.update()

    def _show_history_dialog(self, e):
        """Show dialog to change message history limit."""
        current_value = self.state.settings.get("max_recent_messages", 250)

        value_field = ft.TextField(
            value=str(current_value),
            label="Maximum messages to keep",
            keyboard_type=ft.KeyboardType.NUMBER,
            border_radius=12,
            bgcolor="#1C1C1E",
            border_color="#38383A",
            focused_border_color="#0A84FF",
        )

        def save_value(e):
            try:
                val = value_field.value
                if val is not None:
                    new_value = int(val)
                    if new_value > 0:
                        self._update_setting("max_recent_messages", new_value)
                        self.update()
            except ValueError:
                pass

            if self.page is not None:
                dlg = self.page.dialog
                if dlg is not None:
                    dlg.visible = False
                self.page.update()

        dialog = ft.AlertDialog(
            title=ft.Text("Message History", color="#FFFFFF"),
            content=ft.Container(content=value_field, width=300, padding=ft.padding.all(10)),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: self._close_dialog(dialog)),
                ft.TextButton("Save", on_click=save_value),
            ],
            content_padding=ft.padding.all(10),
        )

        dialog_container = ft.Container(content=dialog, bgcolor="#1C1C1E")

        if self.page is not None:
            self.page.dialog = dialog
            self.page.overlay.append(dialog_container)
            self.page.update()

    def _show_agent_info(self, e):
        """Show current agent information."""
        if not self.state.current_agent:
            return

        info = self.state.current_agent.get_state()

        content = ft.Column(
            controls=[
                ft.Text(f"UUID: {self.state.agent_uuid}", size=12, color="#8E8E93"),
                ft.Text(f"Cycles: {info.get('cycle_index', 0)}", size=12, color="#8E8E93"),
                ft.Text(
                    f"Phase: {info.get('governance', {}).get('phase', 0)}", size=12, color="#8E8E93"
                ),
            ],
            spacing=5,
        )

        dialog = ft.AlertDialog(
            title=ft.Text("Agent Information", color="#FFFFFF"),
            content=ft.Container(content=content, width=300, padding=ft.padding.all(10)),
            actions=[ft.TextButton("Close", on_click=lambda e: self._close_dialog(dialog))],
            content_padding=ft.padding.all(10),
        )

        dialog_container = ft.Container(content=dialog, bgcolor="#1C1C1E")

        if self.page is not None:
            self.page.dialog = dialog
            self.page.overlay.append(dialog_container)
            self.page.update()

    def _create_new_agent(self, e):
        """Create a new agent."""
        from s4_intelligence.g1_intelligence_in import create_agent

        try:
            new_uuid = create_agent(base_path=self.state.base_path)
            self.state.set_agent(new_uuid)
            self.state.status_message = f"Created new agent: {new_uuid[:8]}..."
            self.on_back()  # Return to chat
        except Exception as ex:
            self.state.error_message = f"Failed to create agent: {str(ex)}"

    def _switch_agent(self, e):
        """Switch to a different agent."""
        # In a real app, this would show a list of available agents
        # For now, just show an input dialog

        uuid_field = ft.TextField(
            label="Agent UUID",
            hint_text="Enter agent UUID",
            border_radius=12,
            bgcolor="#1C1C1E",
            border_color="#38383A",
            focused_border_color="#0A84FF",
        )

        def switch(e):
            try:
                uuid_value = uuid_field.value
                if uuid_value is not None and uuid_value.strip():
                    self.state.set_agent(uuid_value)
                    self.state.status_message = f"Switched to agent: {uuid_value[:8]}..."

                    if self.page is not None:
                        dlg = self.page.dialog
                        if dlg is not None:
                            dlg.visible = False
                        self.page.update()

                    self.on_back()
            except Exception as ex:
                self.state.error_message = f"Failed to switch agent: {str(ex)}"

        dialog = ft.AlertDialog(
            title=ft.Text("Switch Agent", color="#FFFFFF"),
            content=ft.Container(content=uuid_field, width=300, padding=ft.padding.all(10)),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: self._close_dialog(dialog)),
                ft.TextButton("Switch", on_click=switch),
            ],
            content_padding=ft.padding.all(10),
        )

        dialog_container = ft.Container(content=dialog, bgcolor="#1C1C1E")

        if self.page is not None:
            self.page.dialog = dialog
            self.page.overlay.append(dialog_container)
            self.page.update()

    def _close_dialog(self, dialog):
        """Close a dialog."""
        if self.page is not None:
            dlg = self.page.dialog
            if dlg is not None:
                dlg.visible = False
            self.page.update()
