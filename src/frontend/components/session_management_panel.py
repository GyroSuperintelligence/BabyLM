import flet as ft
import os
import json
import shutil
from pathlib import Path
from ..assets.styles.theme import GyroTheme
from typing import cast, Optional, List, Dict, Any, Sequence


class SessionManagementPanel(ft.AlertDialog):
    # Constants
    SESSION_ROOT = Path("data/sessions")
    ID_DISPLAY_LENGTH = 8
    SNACKBAR_DURATION = 3000

    def __init__(self, page: Optional[ft.Page] = None):
        self.page = page
        self.session_list: List[Dict[str, Any]] = []
        self.session_table_container = ft.Container()

        super().__init__(
            modal=True,
            title=ft.Text("Session Management", size=20, weight=ft.FontWeight.W_600),
            content=ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Row(
                            [
                                ft.Text("All Sessions:", weight=ft.FontWeight.W_500),
                                ft.IconButton(
                                    icon=ft.icons.UPLOAD_FILE,
                                    tooltip="Import Session",
                                    on_click=self._import_session,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        ),
                        self.session_table_container,
                    ],
                    spacing=15,
                ),
                width=800,
                padding=20,
            ),
            actions=[
                ft.TextButton("Close", on_click=self._close),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            shape=ft.RoundedRectangleBorder(radius=12),
        )

        self._load_and_display_sessions()

    def _load_session_list(self) -> List[Dict[str, Any]]:
        """Load session list with error recovery for individual sessions."""
        if not self.SESSION_ROOT.exists():
            return []

        session_list = []
        for session_dir in self.SESSION_ROOT.iterdir():
            if not session_dir.is_dir():
                continue

            meta_file = session_dir / "session.meta.json"
            if not meta_file.exists():
                continue

            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                session_list.append(
                    {
                        "id": session_dir.name,
                        "meta": meta,
                    }
                )
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load session {session_dir.name}: {e}")
                continue

        return session_list

    def _build_session_table(self) -> ft.Column:
        """Build properly aligned session table."""
        if not self.session_list:
            return ft.Column(
                [
                    ft.Text(
                        "No sessions found.",
                        color=GyroTheme.TEXT_SECONDARY,
                        text_align=ft.TextAlign.CENTER,
                    )
                ]
            )

        # Header - Use fallback color if SURFACE_VARIANT doesn't exist
        header_bgcolor = getattr(GyroTheme, "SURFACE_VARIANT", ft.colors.GREY_100)

        header = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Container(ft.Text("ID", weight=ft.FontWeight.W_600), width=80),
                    ft.Container(ft.Text("Name", weight=ft.FontWeight.W_600), width=200),
                    ft.Container(ft.Text("Phase", weight=ft.FontWeight.W_600), width=80),
                    ft.Container(ft.Text("Knowledge", weight=ft.FontWeight.W_600), width=100),
                    ft.Container(ft.Text("Status", weight=ft.FontWeight.W_600), width=80),
                    ft.Container(ft.Text("Actions", weight=ft.FontWeight.W_600), width=200),
                ],
                spacing=10,
            ),
            bgcolor=header_bgcolor,
            padding=ft.padding.symmetric(horizontal=10, vertical=8),
            border_radius=8,
        )

        # Data rows
        rows: List[ft.Control] = [header]  # Explicit typing as List[ft.Control]
        for session in self.session_list:
            row = self._build_session_row(session)
            rows.append(row)

        return ft.Column(rows, spacing=4)

    def _build_session_row(self, session: Dict[str, Any]) -> ft.Container:
        """Build individual session row with proper alignment."""
        meta = session["meta"]
        session_id = session["id"]
        id_short = session_id[: self.ID_DISPLAY_LENGTH]
        name = meta.get("name", f"Session {id_short}")
        phase = str(meta.get("phase", "?"))
        knowledge = str(meta.get("knowledge_id", "-"))[: self.ID_DISPLAY_LENGTH]
        archived = meta.get("archived", False)

        # Action buttons
        actions = ft.Row(
            [
                ft.IconButton(
                    icon=ft.icons.EDIT,
                    tooltip="Rename",
                    on_click=lambda e, sid=session_id: self._rename_session(sid),
                    icon_size=16,
                ),
                ft.IconButton(
                    icon=ft.icons.UNARCHIVE if archived else ft.icons.ARCHIVE,
                    tooltip="Unarchive" if archived else "Archive",
                    on_click=lambda e, sid=session_id: self._toggle_archive_session(sid),
                    icon_size=16,
                ),
                ft.IconButton(
                    icon=ft.icons.DELETE,
                    tooltip="Delete",
                    icon_color=GyroTheme.ERROR,
                    on_click=lambda e, sid=session_id: self._delete_session(sid),
                    icon_size=16,
                ),
                ft.IconButton(
                    icon=ft.icons.DOWNLOAD,
                    tooltip="Export",
                    on_click=lambda e, sid=session_id: self._export_session(sid),
                    icon_size=16,
                ),
            ],
            spacing=5,
        )

        return ft.Container(
            content=ft.Row(
                controls=[
                    ft.Container(
                        ft.Text(id_short, size=13, color=GyroTheme.TEXT_PRIMARY), width=80
                    ),
                    ft.Container(ft.Text(name, size=13, color=GyroTheme.TEXT_SECONDARY), width=200),
                    ft.Container(ft.Text(phase, size=12, color=GyroTheme.TEXT_SECONDARY), width=80),
                    ft.Container(
                        ft.Text(knowledge, size=12, color=GyroTheme.TEXT_TERTIARY), width=100
                    ),
                    ft.Container(
                        ft.Text(
                            "Archived" if archived else "Active",
                            size=12,
                            color=GyroTheme.TEXT_TERTIARY if archived else GyroTheme.TEXT_SECONDARY,
                        ),
                        width=80,
                    ),
                    ft.Container(actions, width=200),
                ],
                spacing=10,
            ),
            padding=ft.padding.symmetric(horizontal=10, vertical=4),
            border_radius=4,
            bgcolor=ft.colors.TRANSPARENT,
        )

    def _load_and_display_sessions(self):
        """Load sessions and update display."""
        self.session_list = self._load_session_list()
        self.session_table_container.content = self._build_session_table()
        if hasattr(self.session_table_container, "update"):
            self.session_table_container.update()

    def _show_confirmation_dialog(
        self,
        title: str,
        content: str,
        action_text: str,
        action_callback,
        is_destructive: bool = False,
    ):
        """Show a confirmation dialog."""
        if not self.page:
            return

        confirm_dialog = ft.AlertDialog(
            title=ft.Text(title, weight=ft.FontWeight.W_600),
            content=ft.Text(content),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: self._close_current_dialog()),
                ft.FilledButton(
                    action_text,
                    on_click=lambda e: self._execute_and_close(action_callback),
                    style=ft.ButtonStyle(
                        bgcolor=GyroTheme.ERROR if is_destructive else GyroTheme.ACCENT,
                        color="#fff",
                    ),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self._show_dialog(confirm_dialog)

    def _show_input_dialog(self, title: str, label: str, initial_value: str, action_callback):
        """Show an input dialog."""
        if not self.page:
            return

        input_field: ft.TextField = ft.TextField(
            label=label,
            value=initial_value,
            autofocus=True,
            on_submit=lambda e: self._execute_and_close(
                lambda: action_callback(self._safe_strip(input_field.value))
            ),
        )

        input_dialog = ft.AlertDialog(
            title=ft.Text(title, weight=ft.FontWeight.W_600),
            content=input_field,
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: self._close_current_dialog()),
                ft.FilledButton(
                    "Save",
                    on_click=lambda e: self._execute_and_close(
                        lambda: action_callback(self._safe_strip(input_field.value))
                    ),
                    style=ft.ButtonStyle(bgcolor=GyroTheme.ACCENT, color="#fff"),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self._show_dialog(input_dialog)

    def _safe_strip(self, value: Optional[str]) -> str:
        """Safely strip a string value that might be None."""
        return value.strip() if value is not None else ""

    def _show_dialog(self, dialog: ft.AlertDialog):
        """Show a dialog properly."""
        if self.page:
            cast(ft.AlertDialog, dialog).open = True
            self.page.dialog = dialog
            self.page.update()

    def _close_current_dialog(self):
        """Close the current dialog."""
        if self.page and self.page.dialog:
            if hasattr(self.page.dialog, "open"):
                cast(ft.AlertDialog, self.page.dialog).open = False
            self.page.dialog = None
            self.page.update()

    def _execute_and_close(self, callback):
        """Execute callback and close dialog."""
        try:
            callback()
        finally:
            self._close_current_dialog()

    def _rename_session(self, session_id: str):
        """Show rename dialog."""
        # Find current name
        current_name = ""
        for session in self.session_list:
            if session["id"] == session_id:
                current_name = session["meta"].get(
                    "name", f"Session {session_id[:self.ID_DISPLAY_LENGTH]}"
                )
                break

        self._show_input_dialog(
            "Rename Session",
            "Session Name",
            current_name,
            lambda new_name: self._do_rename_session(session_id, new_name),
        )

    def _do_rename_session(self, session_id: str, new_name: str):
        """Perform session rename."""
        if not new_name:
            self._show_message("Session name cannot be empty.", error=True)
            return

        try:
            meta_path = self.SESSION_ROOT / session_id / "session.meta.json"
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            meta["name"] = new_name

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            self._show_message(
                f"Renamed session: {session_id[:self.ID_DISPLAY_LENGTH]}...", info=True
            )
            self._refresh_sessions()

        except (IOError, json.JSONDecodeError) as e:
            self._show_message(f"Error renaming session: {str(e)}", error=True)

    def _toggle_archive_session(self, session_id: str):
        """Show archive/unarchive confirmation."""
        # Check current archive status
        archived = False
        for session in self.session_list:
            if session["id"] == session_id:
                archived = session["meta"].get("archived", False)
                break

        action = "Unarchive" if archived else "Archive"
        message = f"{action} session {session_id[:self.ID_DISPLAY_LENGTH]}...?"

        self._show_confirmation_dialog(
            f"{action} Session?",
            message,
            action,
            lambda: self._do_toggle_archive_session(session_id, not archived),
        )

    def _do_toggle_archive_session(self, session_id: str, archive: bool):
        """Perform archive/unarchive."""
        try:
            meta_path = self.SESSION_ROOT / session_id / "session.meta.json"
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            meta["archived"] = archive

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            action = "Archived" if archive else "Unarchived"
            self._show_message(
                f"{action} session: {session_id[:self.ID_DISPLAY_LENGTH]}...", info=True
            )
            self._refresh_sessions()

        except (IOError, json.JSONDecodeError) as e:
            self._show_message(f"Error updating session: {str(e)}", error=True)

    def _delete_session(self, session_id: str):
        """Show delete confirmation."""
        self._show_confirmation_dialog(
            "Delete Session?",
            f"Are you sure you want to delete session {session_id[:self.ID_DISPLAY_LENGTH]}...? This cannot be undone.",
            "Delete",
            lambda: self._do_delete_session(session_id),
            is_destructive=True,
        )

    def _do_delete_session(self, session_id: str):
        """Perform session deletion."""
        try:
            session_path = self.SESSION_ROOT / session_id
            shutil.rmtree(session_path)
            self._show_message(
                f"Deleted session: {session_id[:self.ID_DISPLAY_LENGTH]}...", info=True
            )
            self._refresh_sessions()

        except OSError as e:
            self._show_message(f"Error deleting session: {str(e)}", error=True)

    def _import_session(self, e):
        """Import session using file picker."""
        if not self.page:
            self._show_message("Import not available.", error=True)
            return

        picker = ft.FilePicker(on_result=self._handle_import_result)
        self.page.overlay.append(picker)
        self.page.update()

        picker.pick_files(dialog_title="Select session file to import", allowed_extensions=["gyro"])

    def _handle_import_result(self, e: ft.FilePickerResultEvent):
        """Handle import file selection - Proper typing."""
        try:
            self._do_import_session(e)
        except Exception as error:
            self._show_message(f"Error during import: {str(error)}", error=True)

    def _do_import_session(self, e: ft.FilePickerResultEvent):
        """Handle import file selection."""
        # Clean up file picker
        if self.page and self.page.overlay:
            self.page.overlay.clear()
            self.page.update()

        if not e.files:
            self._show_message("Import cancelled.", info=True)
            return

        file_path = e.files[0].path
        if not file_path:
            self._show_message("Invalid file selected.", error=True)
            return

        try:
            ext_mgr = getattr(self.page, "extension_manager", None)
            if ext_mgr is None:
                self._show_message("System not initialized.", error=True)
                return
            new_session_id = ext_mgr.import_session(file_path)
            self._show_message(
                f"Imported session: {new_session_id[:self.ID_DISPLAY_LENGTH]}...", info=True
            )
            self._refresh_sessions()
        except Exception as error:
            self._show_message(f"Error importing session: {str(error)}", error=True)

    def _export_session(self, session_id: str):
        """Export session using file picker."""
        if not self.page:
            self._show_message("Export not available.", error=True)
            return

        picker = ft.FilePicker(on_result=lambda e: self._handle_export_result(session_id, e))
        self.page.overlay.append(picker)
        self.page.update()

        picker.save_file(
            dialog_title="Save session export",
            file_name=f"session_{session_id[:self.ID_DISPLAY_LENGTH]}.session.gyro",
            allowed_extensions=["gyro"],
        )

    def _handle_export_result(self, session_id: str, e: ft.FilePickerResultEvent):
        """Handle export file selection - Proper typing."""
        try:
            self._do_export_session(session_id, e)
        except Exception as error:
            self._show_message(f"Error during export: {str(error)}", error=True)

    def _do_export_session(self, session_id: str, e: ft.FilePickerResultEvent):
        """Handle export file selection."""
        # Clean up file picker
        if self.page and self.page.overlay:
            self.page.overlay.clear()
            self.page.update()

        if not e.path:
            self._show_message("Export cancelled.", info=True)
            return

        try:
            ext_mgr = getattr(self.page, "extension_manager", None)
            if ext_mgr is None:
                self._show_message("System not initialized.", error=True)
                return
            ext_mgr.export_session(session_id, e.path)
            self._show_message(f"Exported to: {Path(e.path).name}", info=True)
        except Exception as error:
            self._show_message(f"Error exporting session: {str(error)}", error=True)

    def _refresh_sessions(self):
        """Refresh session list and update display."""
        self._load_and_display_sessions()
        if self.page:
            self.page.update()

    def _show_message(self, msg: str, error: bool = False, info: bool = False):
        """Show message via snackbar with proper styling."""
        if not self.page:
            return

        if error:
            bgcolor = GyroTheme.ERROR
            color = "#fff"
        elif info:
            bgcolor = GyroTheme.ACCENT
            color = "#fff"
        else:
            # Use fallback color if SURFACE_VARIANT doesn't exist
            bgcolor = getattr(GyroTheme, "SURFACE_VARIANT", ft.colors.GREY_100)
            color = GyroTheme.TEXT_PRIMARY

        snackbar = ft.SnackBar(
            content=ft.Text(msg, color=color),
            bgcolor=bgcolor,
            duration=self.SNACKBAR_DURATION,
        )

        self.page.snack_bar = snackbar
        self.page.update()

    def _close(self, e):
        """Close the main dialog."""
        self.open = False
        if self.page:
            self.page.dialog = None
            self.page.update()
