import flet as ft
import os
import json
from pathlib import Path
from ..assets.styles.theme import GyroTheme
from typing import cast


class KnowledgeManagementPanel(ft.AlertDialog):
    def __init__(self, page=None, import_picker=None, export_picker=None):
        self.page = page
        self.import_picker = import_picker
        self.export_picker = export_picker
        self.knowledge_list = self._load_knowledge_list()
        super().__init__(
            modal=True,
            title=ft.Text("Knowledge Management", size=20, weight=ft.FontWeight.W_600),
            content=ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Row(
                            [
                                ft.Text("All Knowledge Packages:", weight=ft.FontWeight.W_500),
                                ft.IconButton(
                                    icon=ft.icons.UPLOAD_FILE,
                                    tooltip="Import Knowledge",
                                    on_click=self._import_knowledge,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        ),
                        self._build_knowledge_table(),
                    ],
                    spacing=15,
                ),
                width=700,
                padding=20,
            ),
            actions=[
                ft.TextButton("Close", on_click=self._close),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            shape=ft.RoundedRectangleBorder(radius=12),
        )

    def _load_knowledge_list(self):
        knowledge_root = Path("data/knowledge")
        knowledge_list = []
        for d in knowledge_root.iterdir():
            if d.is_dir() and (d / "knowledge.meta.json").exists():
                try:
                    with open(d / "knowledge.meta.json", "r") as f:
                        meta = json.load(f)
                    knowledge_list.append(
                        {
                            "id": d.name,
                            "meta": meta,
                        }
                    )
                except Exception:
                    continue
        return knowledge_list

    def _build_knowledge_table(self):
        rows = []
        for k in self.knowledge_list:
            meta = k["meta"]
            id_short = k["id"][:8]
            created = meta.get("created_ts", "?")
            parent = meta.get("parent_knowledge_id", "-")
            step_count = meta.get("step_count", "?")
            row = ft.Row(
                controls=[
                    ft.Text(id_short, size=13, color=GyroTheme.TEXT_PRIMARY),
                    ft.Text(str(created), size=12, color=GyroTheme.TEXT_SECONDARY),
                    ft.Text(
                        str(parent)[:8] if parent else "-", size=12, color=GyroTheme.TEXT_TERTIARY
                    ),
                    ft.Text(str(step_count), size=12, color=GyroTheme.TEXT_SECONDARY),
                    ft.IconButton(
                        icon=ft.icons.FILE_COPY,
                        tooltip="Fork",
                        on_click=lambda e, kid=k["id"]: self._fork_knowledge(kid),
                    ),
                    ft.IconButton(
                        icon=ft.icons.DOWNLOAD,
                        tooltip="Export",
                        on_click=lambda e, kid=k["id"]: self._export_knowledge(kid),
                    ),
                    ft.IconButton(
                        icon=ft.icons.DELETE,
                        tooltip="Delete",
                        icon_color=GyroTheme.ERROR,
                        on_click=lambda e, kid=k["id"]: self._delete_knowledge(kid),
                    ),
                ],
                spacing=10,
            )
            rows.append(row)
        header = ft.Row(
            controls=[
                ft.Text("ID", weight=ft.FontWeight.W_600),
                ft.Text("Created", weight=ft.FontWeight.W_600),
                ft.Text("Parent", weight=ft.FontWeight.W_600),
                ft.Text("Steps", weight=ft.FontWeight.W_600),
                ft.Text("Fork"),
                ft.Text("Export"),
                ft.Text("Delete"),
            ],
            spacing=10,
        )
        controls = [header] + rows
        controls_cast = cast(list[ft.Control], controls)
        return ft.Column(controls=controls_cast, spacing=8)  # type: ignore

    def _fork_knowledge(self, knowledge_id):
        # Confirmation dialog for forking
        confirm_dialog = ft.AlertDialog(
            title=ft.Text("Fork Knowledge?", weight=ft.FontWeight.W_600),
            content=ft.Text(
                f"Fork knowledge package {knowledge_id[:8]}...? This will create a new package with a new ID."
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: self._close_dialog()),
                ft.FilledButton(
                    "Fork",
                    on_click=lambda e: self._do_fork_knowledge(knowledge_id),
                    style=ft.ButtonStyle(bgcolor=GyroTheme.ACCENT, color="#fff"),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        if self.page:
            self.page.dialog = confirm_dialog
            self.page.update()

    def _do_fork_knowledge(self, knowledge_id):
        # Actually fork the knowledge package (copy dir, update meta)
        try:
            import shutil, uuid, time
            from pathlib import Path

            src = Path("data/knowledge") / knowledge_id
            new_id = str(uuid.uuid4())
            dest = Path("data/knowledge") / new_id
            shutil.copytree(src, dest)
            meta_path = dest / "knowledge.meta.json"
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta["knowledge_id"] = new_id
            meta["parent_knowledge_id"] = knowledge_id
            meta["created_ts"] = time.time()
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            self._show_message(f"Forked knowledge: {new_id[:8]}...", info=True)
            self._refresh()
        except Exception as ex:
            self._show_message(f"Error forking: {ex}", error=True)
        self._close_dialog()

    def _import_knowledge(self, e):
        # Use app-level file picker
        if hasattr(self, "import_picker") and self.import_picker:
            self.import_picker.on_result = self._do_import_knowledge
            self.import_picker.pick_files(allowed_extensions=[".gyro"])
        else:
            self._show_message("Import not available.", error=True)

    def _do_import_knowledge(self, e):
        if not e.files or not e.files[0].path:
            self._show_message("Import cancelled.", info=True)
            return
        try:
            # Call backend import logic (assume ExtensionManager is accessible via self.page)
            from core.extension_manager import ExtensionManager

            ext_mgr = getattr(self.page, "extension_manager", None)
            if ext_mgr is None:
                self._show_message("System not initialized.", error=True)
                return
            new_knowledge_id = ext_mgr.import_knowledge(e.files[0].path)
            self._show_message(f"Imported knowledge: {new_knowledge_id[:8]}...", info=True)
            self._refresh()
        except Exception as ex:
            self._show_message(f"Error importing: {ex}", error=True)

    def _export_knowledge(self, knowledge_id):
        # Use app-level file picker
        if hasattr(self, "export_picker") and self.export_picker:
            self.export_picker.on_result = lambda e: self._do_export_knowledge(knowledge_id, e)
            self.export_picker.save_file(
                file_name=f"knowledge_{knowledge_id[:8]}.gyro", allowed_extensions=[".gyro"]
            )
        else:
            self._show_message("Export not available.", error=True)

    def _do_export_knowledge(self, knowledge_id, e):
        if not e.path:
            self._show_message("Export cancelled.", info=True)
            return
        try:
            # Call backend export logic (assume ExtensionManager is accessible via self.page)
            from core.extension_manager import ExtensionManager

            ext_mgr = getattr(self.page, "extension_manager", None)
            if ext_mgr is None:
                self._show_message("System not initialized.", error=True)
                return
            ext_mgr.export_knowledge(e.path)
            self._show_message(f"Exported to: {e.path}", info=True)
        except Exception as ex:
            self._show_message(f"Error exporting: {ex}", error=True)

    def _delete_knowledge(self, knowledge_id):
        # Confirmation dialog for delete
        confirm_dialog = ft.AlertDialog(
            title=ft.Text("Delete Knowledge?", weight=ft.FontWeight.W_600),
            content=ft.Text(
                f"Are you sure you want to delete knowledge package {knowledge_id[:8]}...? This cannot be undone."
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda e: self._close_dialog()),
                ft.FilledButton(
                    "Delete",
                    on_click=lambda e: self._do_delete_knowledge(knowledge_id),
                    style=ft.ButtonStyle(bgcolor=GyroTheme.ERROR, color="#fff"),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        if self.page:
            self.page.dialog = confirm_dialog
            self.page.update()

    def _do_delete_knowledge(self, knowledge_id):
        try:
            import shutil
            from pathlib import Path

            shutil.rmtree(Path("data/knowledge") / knowledge_id)
            self._show_message(f"Deleted knowledge: {knowledge_id[:8]}...", info=True)
            self._refresh()
        except Exception as ex:
            self._show_message(f"Error deleting: {ex}", error=True)
        self._close_dialog()

    def _refresh(self):
        self.knowledge_list = self._load_knowledge_list()
        container = cast(ft.Container, self.content)
        column = cast(ft.Column, container.content)
        controls = column.controls
        if len(controls) > 1:
            controls[1] = self._build_knowledge_table()
        container.update()

    def _close_dialog(self):
        if self.page:
            if getattr(self.page, "dialog", None) is self:
                self.open = False
                self.page.dialog = None
            self.page.update()

    def _show_message(self, msg, error=False, info=False):
        color = (
            GyroTheme.ERROR if error else (GyroTheme.ACCENT if info else GyroTheme.TEXT_ON_ACCENT)
        )
        bgcolor = GyroTheme.ERROR if error else (GyroTheme.ACCENT if info else GyroTheme.ACCENT)
        snackbar = ft.SnackBar(
            content=ft.Text(msg, color=color),
            bgcolor=bgcolor,
            open=True,
            duration=3000,
        )
        if self.page:
            self.page.snack_bar = snackbar
            self.page.update()

    def _close(self, e):
        self.open = False
        if self.page:
            self.page.dialog = None
            self.page.update()
