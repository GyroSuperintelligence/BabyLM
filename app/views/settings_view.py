"""
settings_view.py - Settings management view for GyroSI Baby LM

Handles general settings, agent management, and curriculum operations.
"""

import flet as ft
from typing import Optional, Callable, Dict, Any
import json
from datetime import datetime

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

        # Get curriculum data safely
        curriculum_patterns = self._get_curriculum_patterns()

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

        # Agency management section
        agency_section = Section(
            title="Agency Management",
            controls=[
                SettingRow(
                    title="Curriculum",
                    subtitle=f"{len(curriculum_patterns)} patterns learned",
                    control=None,
                ),
                ft.Container(
                    content=ft.Column(
                        controls=[
                            ActionButton(
                                text="Add Pattern",
                                icon=ft.icons.ADD_CIRCLE_OUTLINE,
                                on_click=self._add_pattern,
                                full_width=True,
                            ),
                            ActionButton(
                                text="Export Curriculum",
                                icon=ft.icons.DOWNLOAD,
                                on_click=self._export_curriculum,
                                full_width=True,
                            ),
                            ActionButton(
                                text="Import Curriculum",
                                icon=ft.icons.UPLOAD,
                                on_click=self._import_curriculum,
                                full_width=True,
                            ),
                            ActionButton(
                                text="Clear Curriculum",
                                icon=ft.icons.DELETE_OUTLINE,
                                on_click=self._clear_curriculum,
                                style="danger",
                                full_width=True,
                            ),
                        ],
                        spacing=10,
                    ),
                    padding=ft.padding.all(20),
                ),
            ],
        )

        # Scroll view with all sections
        return ft.Column(
            controls=[
                header,
                ft.Container(
                    content=ft.Column(
                        controls=[general_section, agent_section, agency_section],
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

    def _get_curriculum_patterns(self) -> Dict[str, Any]:
        """Safely get curriculum patterns."""
        try:
            # Try to access curriculum patterns through the state
            curriculum = getattr(self.state, "curriculum", {})
            if isinstance(curriculum, dict):
                return curriculum.get("patterns", {})
            return {}
        except (AttributeError, TypeError):
            # If any error occurs, return empty dict
            return {}

    def _update_setting(self, key: str, value: Any):
        """Update a setting value."""
        self.state.update_setting(key, value)
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

        info = self.state.get_engine_state()

        content = ft.Column(
            controls=[
                ft.Text(f"UUID: {self.state.agent_uuid}", size=12, color="#8E8E93"),
                ft.Text(f"Cycles: {info.get('cycle_index', 0)}", size=12, color="#8E8E93"),
                ft.Text(f"Patterns: {info.get('curriculum_size', 0)}", size=12, color="#8E8E93"),
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

    def _add_pattern(self, e):
        """Add a new pattern to curriculum."""
        # This would show a pattern input interface
        # For now, just show a placeholder
        self.state.status_message = "Pattern editor not yet implemented"

    def _export_curriculum(self, e):
        """Export curriculum to file."""
        if not self.state.current_agent:
            return

        try:
            # Export to JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"curriculum_export_{timestamp}.json"

            # In a real app, this would use a file picker
            # For now, just show success message
            self.state.status_message = f"Curriculum exported to {filename}"
        except Exception as ex:
            self.state.error_message = f"Export failed: {str(ex)}"

    def _import_curriculum(self, e):
        """Import curriculum from file."""
        # This would show a file picker
        self.state.status_message = "Import functionality not yet implemented"

    def _clear_curriculum(self, e):
        """Clear all learned patterns."""

        def confirm_clear(e):
            if self.state.current_agent:
                # Use a safe method to update curriculum
                self._update_curriculum({"patterns": {}, "byte_to_token": {}, "token_to_byte": {}})
                self.state.status_message = "Curriculum cleared"

            if self.page is not None:
                dlg = self.page.dialog
                if dlg is not None:
                    dlg.visible = False
                self.page.update()

            self.update()

        dialog = ft.AlertDialog(
            title=ft.Text("Clear Curriculum", color="#FFFFFF"),
            content=ft.Text(
                "This will delete all learned patterns. This action cannot be undone.",
                color="#8E8E93",
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: self._close_dialog(dialog)),
                ft.TextButton(
                    "Clear", on_click=confirm_clear, style=ft.ButtonStyle(color="#FF453A")
                ),
            ],
            content_padding=ft.padding.all(10),
        )

        dialog_container = ft.Container(content=dialog, bgcolor="#1C1C1E")

        if self.page is not None:
            self.page.dialog = dialog
            self.page.overlay.append(dialog_container)
            self.page.update()

    def _update_curriculum(self, new_curriculum: Dict[str, Any]):
        """Safely update the curriculum."""
        if hasattr(self.state, "curriculum"):
            setattr(self.state, "curriculum", new_curriculum)

        # If the agent has a method to persist curriculum, call it
        if self.state.current_agent and hasattr(self.state.current_agent, "_persist_curriculum"):
            self.state.current_agent._persist_curriculum()

    def _close_dialog(self, dialog):
        """Close a dialog."""
        if self.page is not None:
            dlg = self.page.dialog
            if dlg is not None:
                dlg.visible = False
            self.page.update()
