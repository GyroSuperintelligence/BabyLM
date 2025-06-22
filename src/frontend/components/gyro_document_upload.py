# src/frontend/components/gyro_document_upload.py
import flet as ft
import asyncio
from typing import Callable, Optional
from pathlib import Path
from ..assets.styles.theme import GyroTheme


class GyroDocumentUpload(ft.UserControl):
    """Document upload area with Apple-like design"""

    # Unused variables for future debug: (none listed)

    def __init__(self, on_upload: Callable):
        super().__init__()
        self.on_upload = on_upload
        self.file_picker = ft.FilePicker(on_result=self._handle_file_pick)
        self.current_file: Optional[str] = None

    def build(self):
        # Upload area
        self.upload_area = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Icon(
                        ft.icons.CLOUD_UPLOAD_OUTLINED, size=40, color=GyroTheme.TEXT_SECONDARY
                    ),
                    ft.Text(
                        "Drop files here or click to upload",
                        size=14,
                        color=GyroTheme.TEXT_SECONDARY,
                        text_align=ft.TextAlign.CENTER,
                    ),
                    ft.Text(
                        "Supports .txt, .pdf, .docx",
                        size=12,
                        color=GyroTheme.TEXT_TERTIARY,
                        text_align=ft.TextAlign.CENTER,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
            ),
            padding=ft.padding.all(30),
            border=ft.border.all(2, GyroTheme.BORDER),
            border_radius=12,
            on_click=self._open_file_picker,
            on_hover=self._on_hover,
            alignment=ft.alignment.center,
        )

        # Progress indicator
        self.progress_bar = ft.ProgressBar(
            width=300, color=GyroTheme.ACCENT, bgcolor=GyroTheme.BORDER, visible=False
        )

        # File info
        self.file_info = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Icon(ft.icons.INSERT_DRIVE_FILE, size=20, color=GyroTheme.ACCENT),
                    ft.Text("", size=13, color=GyroTheme.TEXT_PRIMARY),
                    ft.IconButton(
                        icon=ft.icons.CLOSE,
                        icon_size=16,
                        icon_color=GyroTheme.TEXT_SECONDARY,
                        on_click=self._clear_file,
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=ft.padding.all(10),
            bgcolor=GyroTheme.SURFACE,
            border_radius=8,
            visible=False,
        )

        return ft.Column(
            controls=[self.upload_area, self.progress_bar, self.file_info, self.file_picker],
            spacing=15,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _on_hover(self, e):
        """Handle hover effect"""
        if e.data == "true":
            self.upload_area.bgcolor = GyroTheme.HOVER_BG
            self.upload_area.border = ft.border.all(2, GyroTheme.ACCENT)
        else:
            self.upload_area.bgcolor = None
            self.upload_area.border = ft.border.all(2, GyroTheme.BORDER)
        self.update()

    def _open_file_picker(self, e):
        """Open file picker dialog"""
        self.file_picker.pick_files(
            allowed_extensions=["txt", "pdf", "docx"], dialog_title="Select a document to upload"
        )

    async def _handle_file_pick(self, e: ft.FilePickerResultEvent):
        """Handle file selection"""
        if not e.files:
            return

        file = e.files[0]
        self.current_file = file.path

        # Show progress
        self.progress_bar.visible = True
        self.upload_area.visible = False
        await self.update_async()

        # Simulate upload progress
        for i in range(101):
            self.progress_bar.value = i / 100
            await self.update_async()
            await asyncio.sleep(0.01)

        # Show file info
        self.file_info.content.controls[1].value = file.name
        self.file_info.visible = True
        self.progress_bar.visible = False
        await self.update_async()

        # Notify parent
        await self.on_upload(file.path)

    async def _clear_file(self, e):
        """Clear uploaded file"""
        self.current_file = None
        self.file_info.visible = False
        self.upload_area.visible = True
        await self.update_async()
