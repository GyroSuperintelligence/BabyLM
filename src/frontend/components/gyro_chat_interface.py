# src/frontend/components/gyro_chat_interface.py

import flet as ft
from typing import Optional, List
from datetime import datetime
import asyncio
from ..assets.styles.theme import GyroTheme

class ChatMessage(ft.UserControl):
    """
    Individual chat message with Apple-like design.
    May require Flet >=0.13 for animation and border features.
    """
    def __init__(self, content: str, is_user: bool, timestamp: datetime, is_system: bool = False):
        super().__init__()
        self.content = content
        self.is_user = is_user
        self.timestamp = timestamp
        self.is_system = is_system

    def build(self):
        # Determine style and alignment
        if self.is_system:
            bgcolor = GyroTheme.BORDER
            border = ft.border.all(1, GyroTheme.BORDER)
            alignment = ft.alignment.center
            text_color = GyroTheme.TEXT_PRIMARY
            timestamp_color = GyroTheme.TEXT_TERTIARY
        elif self.is_user:
            bgcolor = GyroTheme.USER_MESSAGE_BG
            border = None
            alignment = ft.alignment.center_right
            text_color = GyroTheme.TEXT_ON_ACCENT
            timestamp_color = ft.colors.WHITE70  # May require compatible Flet version
        else:
            bgcolor = GyroTheme.ASSISTANT_MESSAGE_BG
            border = ft.border.all(1, GyroTheme.BORDER)
            alignment = ft.alignment.center_left
            text_color = GyroTheme.TEXT_PRIMARY
            timestamp_color = GyroTheme.TEXT_TERTIARY

        # Message content
        message_bubble = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        self.content,
                        size=14,
                        color=text_color,
                        selectable=True,
                    ),
                    ft.Text(
                        self.timestamp.strftime("%I:%M %p"),
                        size=11,
                        color=timestamp_color,
                    ),
                ],
                spacing=5,
            ),
            padding=ft.padding.all(12),
            border_radius=ft.border_radius.all(16),
            bgcolor=bgcolor,
            border=border,
            max_width=500,
        )

        return ft.Container(
            content=message_bubble,
            alignment=alignment,
            animate=ft.animation.Animation(300, ft.AnimationCurve.EASE_OUT),  # Animation may require Flet >=0.13
        )

class GyroChatInterface(ft.UserControl):
    """
    Main chat interface with Apple-like design.
    Uses Flet ListView, TextField, and IconButton components.
    """
    def __init__(self):
        super().__init__()
        self.session_id: Optional[str] = None
        self.messages: List[ChatMessage] = []
        self.is_processing = False

    def build(self):
        self.message_list = ft.ListView(
            spacing=15, padding=ft.padding.all(20), expand=True, auto_scroll=True
        )

        self.input_field = ft.TextField(
            hint_text="Type a message...",
            multiline=True,
            min_lines=1,
            max_lines=5,
            filled=True,
            bgcolor=GyroTheme.INPUT_BG,
            border_radius=20,
            border_color=GyroTheme.BORDER,
            focused_border_color=GyroTheme.ACCENT,
            text_size=14,
            on_submit=self._send_message,
            shift_enter=True,
            expand=True,
        )

        self.send_button = ft.IconButton(
            icon=ft.icons.SEND_ROUNDED,  # May require compatible Flet version
            icon_color=GyroTheme.ACCENT,
            icon_size=20,
            on_click=self._send_message,
            disabled=False,
            style=ft.ButtonStyle(
                shape=ft.CircleBorder(),
                bgcolor={
                    ft.ControlState.DISABLED: GyroTheme.BORDER,
                    ft.ControlState.DEFAULT: GyroTheme.ACCENT,
                },  # These states may require Flet >=0.12
            ),
        )

        self.processing_indicator = ft.ProgressRing(
            width=16, height=16, stroke_width=2, color=GyroTheme.ACCENT, visible=False
        )

        input_container = ft.Container(
            content=ft.Row(
                controls=[
                    self.input_field,
                    ft.Stack(
                        controls=[
                            self.send_button,
                            ft.Container(
                                content=self.processing_indicator,
                                alignment=ft.alignment.center,
                                width=40,
                                height=40,
                            ),
                        ]
                    ),
                ],
                spacing=10,
                vertical_alignment=ft.CrossAxisAlignment.END,
            ),
            padding=ft.padding.all(20),
            bgcolor=GyroTheme.SURFACE,
            border=ft.border.only(top=ft.BorderSide(1, GyroTheme.BORDER)),
        )

        return ft.Column(
            controls=[self.message_list, input_container],
            spacing=0,
            expand=True,
        )

    async def load_session(self, session_id: str):
        """
        Load chat history for a session.
        """
        self.session_id = session_id
        self.messages.clear()
        self.message_list.controls.clear()

        welcome_msg = ChatMessage(
            content="Hello! I'm GyroSI Baby ML. I'm ready to learn and chat with you.",
            is_user=False,
            timestamp=datetime.now(),
        )
        self.messages.append(welcome_msg)
        self.message_list.controls.append(welcome_msg)

        await self.update_async()

    async def _send_message(self, e):
        """
        Send a message through G3_BU_In.
        """
        if not self.input_field.value or self.is_processing:
            return

        message_text = self.input_field.value.strip()
        self.input_field.value = ""

        user_msg = ChatMessage(content=message_text, is_user=True, timestamp=datetime.now())
        self.messages.append(user_msg)
        self.message_list.controls.append(user_msg)

        await self._set_processing(True)
        await self.update_async()

        # Simulate processing
        await asyncio.sleep(1)

        assistant_msg = ChatMessage(
            content="I'm processing your message through my navigation cycle. This is where the GyroSI learning happens!",
            is_user=False,
            timestamp=datetime.now(),
        )
        self.messages.append(assistant_msg)
        self.message_list.controls.append(assistant_msg)

        await self._set_processing(False)
        await self.update_async()

    async def _set_processing(self, processing: bool):
        """
        Update processing state.
        """
        self.is_processing = processing
        self.send_button.visible = not processing
        self.processing_indicator.visible = processing
        self.input_field.disabled = processing

    async def process_document(self, file_path: str):
        """
        Process uploaded document through G2_BU_In.
        """
        doc_msg = ChatMessage(
            content=f"Processing document: {file_path}",
            is_user=False,
            is_system=True,
            timestamp=datetime.now(),
        )
        self.messages.append(doc_msg)
        self.message_list.controls.append(doc_msg)
        await self.update_async()
