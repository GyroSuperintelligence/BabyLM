# src/frontend/components/gyro_chat_interface.py

import flet as ft
from typing import Optional, List
from datetime import datetime
from ..assets.styles.theme import GyroTheme
from gyro_api import get_language_output


class ChatMessage(ft.UserControl):
    """Individual chat message with Apple-like design"""

    def __init__(self, content: str, is_user: bool, timestamp: datetime, is_system: bool = False):
        super().__init__()
        self.content = content
        self.is_user = is_user
        self.timestamp = timestamp
        self.is_system = is_system

    def build(self):
        # [Same as before - no changes needed]
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
            timestamp_color = "#FFFFFF70"
        else:
            bgcolor = GyroTheme.ASSISTANT_MESSAGE_BG
            border = ft.border.all(1, GyroTheme.BORDER)
            alignment = ft.alignment.center_left
            text_color = GyroTheme.TEXT_PRIMARY
            timestamp_color = GyroTheme.TEXT_TERTIARY

        message_bubble = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(self.content, size=14, color=text_color, selectable=True),
                    ft.Text(self.timestamp.strftime("%I:%M %p"), size=11, color=timestamp_color),
                ],
                spacing=5,
            ),
            padding=ft.padding.all(12),
            border_radius=ft.border_radius.all(16),
            bgcolor=bgcolor,
            border=border,
            width=500,
        )

        return ft.Container(content=message_bubble, alignment=alignment)


class GyroChatInterface(ft.UserControl):
    """Chat interface integrated with ExtensionManager"""

    def __init__(self, extension_manager):
        super().__init__()
        self.extension_manager = extension_manager
        self.session_id: Optional[str] = None
        self.messages: List[ChatMessage] = []
        self.is_processing = False
        self._snackbar = None
        self._pending_clear = False

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
            autofocus=True,
        )

        self.send_button = ft.IconButton(
            icon=ft.icons.SEND,
            icon_color=GyroTheme.ACCENT,
            icon_size=20,
            on_click=self._send_message,
            disabled=False,
            tooltip="Send message",
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

    def load_session(self, session_id: str):
        """Load chat history for a session"""
        self.session_id = session_id
        self.messages.clear()
        self.message_list.controls.clear()

        try:
            # Get current system state using correct memory tags
            phase = self.extension_manager.engine.phase
            knowledge_id = self.extension_manager.get_knowledge_id()

            # Get system health for additional info
            extension_count = len(self.extension_manager.extensions)

            welcome_msg = ChatMessage(
                content=f"Hello! I'm GyroSI Baby ML.\n"
                f"Phase: {phase}/47\n"
                f"Knowledge: {knowledge_id[:8]}...\n"
                f"Extensions: {extension_count} loaded\n"
                f"Ready to learn through navigation!",
                is_user=False,
                timestamp=datetime.now(),
            )
            self.messages.append(welcome_msg)
            self.message_list.controls.append(welcome_msg)
            self.update()
        except Exception as e:
            error_msg = ChatMessage(
                content=f"Error loading session: {str(e)}",
                is_user=False,
                is_system=True,
                timestamp=datetime.now(),
            )
            self.messages.append(error_msg)
            self.message_list.controls.append(error_msg)
            self.update()

    def update_language_output(self):
        if not self.session_id:
            return
        try:
            output_lines = get_language_output(self.session_id)
            for line in output_lines:
                assistant_msg = ChatMessage(
                    content=line,
                    is_user=False,
                    timestamp=datetime.now(),
                )
                self.messages.append(assistant_msg)
                self.message_list.controls.append(assistant_msg)
            self.update()
        except Exception as e:
            error_msg = ChatMessage(
                content=f"Error fetching language output: {str(e)}",
                is_user=False,
                is_system=True,
                timestamp=datetime.now(),
            )
            self.messages.append(error_msg)
            self.message_list.controls.append(error_msg)
            self.update()

    def _show_message(self, msg, error=False, info=False):
        color = (
            GyroTheme.ERROR
            if error
            else (GyroTheme.TEXT_SECONDARY if info else GyroTheme.TEXT_ON_ACCENT)
        )
        bgcolor = GyroTheme.ERROR if error else (GyroTheme.SURFACE if info else GyroTheme.ACCENT)
        snackbar = ft.SnackBar(
            content=ft.Text(msg, color=color),
            bgcolor=bgcolor,
            open=True,
            duration=3000,
        )
        if hasattr(self, "page") and self.page:
            self.page.snack_bar = snackbar
            self.page.update()
        else:
            self._snackbar = snackbar
            self.update()

    def _send_message(self, e):
        """Send message through G3→G2→G4→G5 cycle"""
        if not self.input_field.value or self.is_processing:
            self._show_message("Cannot send empty or duplicate message.", error=True)
            return

        message_text = self.input_field.value.strip()
        self.input_field.value = ""

        # Add user message
        user_msg = ChatMessage(content=message_text, is_user=True, timestamp=datetime.now())
        self.messages.append(user_msg)
        self.message_list.controls.append(user_msg)

        self._set_processing(True)
        self.update()

        try:
            # Process through real GyroSI system
            self._process_with_gyro_system(message_text)
            # Fetch and display new language output
            self.update_language_output()
            self._show_message("Message processed.", info=True)
        except Exception as ex:
            self._show_message(f"Error sending message: {ex}", error=True)
        finally:
            self._set_processing(False)
            self.update()

    def _process_with_gyro_system(self, text: str) -> str:
        """Process text through the complete GyroSI system"""
        try:
            # Store as G2 epigenetic event (user input)
            self.extension_manager.gyro_epigenetic_memory(
                "current.gyrotensor_com",
                {
                    "text": text,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": self.session_id,
                },
            )

            # Process text byte-by-byte through navigation cycle
            text_bytes = text.encode("utf-8")
            navigation_events = []

            initial_phase = self.extension_manager.engine.phase

            for byte_val in text_bytes:
                # Execute complete gyro_operation cycle
                result = self.extension_manager.gyro_operation(byte_val)
                if result:
                    navigation_events.append(result)

            final_phase = self.extension_manager.engine.phase

            # Get navigation log info
            nav_log_size = self.extension_manager.navigation_log.step_count

            # Generate intelligent response based on system state
            if navigation_events:
                response = f"🧠 Processed '{text}' through GyroSI navigation cycle.\n\n"
                response += f"📊 Navigation Events: {len(navigation_events)}\n"
                response += f"🔄 Phase: {initial_phase} → {final_phase}\n"
                response += f"📝 Navigation Log: {nav_log_size} total steps\n"
                response += (
                    f"🎯 Structural Resonance: {len(navigation_events)}/{len(text_bytes)} bytes\n\n"
                )

                # Show some recent navigation events
                if len(navigation_events) > 0:
                    recent_events = (
                        navigation_events[-3:] if len(navigation_events) >= 3 else navigation_events
                    )
                    response += f"Recent events: {recent_events}\n"

                response += "The system has learned from your input through structural resonance!"
            else:
                response = f"📝 Processed '{text}' ({len(text_bytes)} bytes)\n\n"
                response += f"🔄 Phase: {initial_phase} → {final_phase}\n"
                response += f"⚡ No structural resonance detected\n"
                response += f"📊 Navigation Log: {nav_log_size} total steps\n\n"
                response += "Input processed but no learning events generated."

            # Store response in structural memory
            self.extension_manager.gyro_structural_memory(
                "current.gyrotensor_nest",
                {
                    "input": text,
                    "response": response,
                    "navigation_events": len(navigation_events),
                    "timestamp": datetime.now().isoformat(),
                },
            )

            return response

        except Exception as e:
            error_response = f"❌ Error processing input: {str(e)}\n\nThe GyroSI system encountered an issue during navigation."

            # Store error in immunity memory
            self.extension_manager.gyro_immunity_memory(
                "current.gyrotensor_quant",
                {"error": str(e), "input": text, "timestamp": datetime.now().isoformat()},
            )

            return error_response

    def _set_processing(self, processing: bool):
        """Update processing state"""
        self.is_processing = processing
        self.send_button.visible = not processing
        self.processing_indicator.visible = processing
        self.input_field.disabled = processing

    def process_document(self, file_path: str):
        """Process document through G2_BU_In import adaptors"""
        try:
            doc_msg = ChatMessage(
                content=f"📄 Processing document: {file_path}",
                is_user=False,
                is_system=True,
                timestamp=datetime.now(),
            )
            self.messages.append(doc_msg)
            self.message_list.controls.append(doc_msg)
            self.update()

            # Read and process document through GyroSI
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Store document in G2 epigenetic memory
            self.extension_manager.gyro_epigenetic_memory(
                "current.gyrotensor_com",
                {
                    "file_path": file_path,
                    "content_length": len(content),
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Process document content through navigation
            initial_phase = self.extension_manager.engine.phase
            navigation_events = []

            # Process first 1000 bytes for demo (full document would be processed in chunks)
            sample_content = content[:1000].encode("utf-8")

            for byte_val in sample_content:
                result = self.extension_manager.gyro_operation(byte_val)
                if result:
                    navigation_events.append(result)

            final_phase = self.extension_manager.engine.phase
            nav_log_size = self.extension_manager.navigation_log.step_count

            result_msg = ChatMessage(
                content=f"✅ Document processed through GyroSI!\n\n"
                f"📊 Content: {len(content)} characters\n"
                f"🧠 Navigation Events: {len(navigation_events)}\n"
                f"🔄 Phase: {initial_phase} → {final_phase}\n"
                f"📝 Total Navigation Steps: {nav_log_size}\n\n"
                f"The document has been integrated into the knowledge system through structural resonance.",
                is_user=False,
                timestamp=datetime.now(),
            )
            self.messages.append(result_msg)
            self.message_list.controls.append(result_msg)
            self.update()
            self._show_message("Document processed.", info=True)

        except Exception as e:
            error_msg = ChatMessage(
                content=f"❌ Error processing document: {str(e)}",
                is_user=False,
                is_system=True,
                timestamp=datetime.now(),
            )
            self.messages.append(error_msg)
            self.message_list.controls.append(error_msg)
            self.update()
            self._show_message(f"Error processing document: {e}", error=True)

    # Optionally, add a clear chat history feature with confirmation
    def clear_chat_history(self):
        self._pending_clear = True
        confirm_dialog = ft.AlertDialog(
            title=ft.Text("Clear Chat History?", weight=ft.FontWeight.W_600),
            content=ft.Text(
                "Are you sure you want to clear the chat history for this session? This cannot be undone."
            ),
            actions=[
                ft.TextButton("Cancel", on_click=self._cancel_clear),
                ft.FilledButton(
                    "Clear",
                    on_click=self._confirm_clear,
                    style=ft.ButtonStyle(bgcolor=GyroTheme.ERROR, color="#fff"),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )
        if hasattr(self, "page") and self.page:
            self.page.dialog = confirm_dialog
            self.page.update()
        else:
            self._snackbar = confirm_dialog
            self.update()

    def _cancel_clear(self, e):
        if hasattr(self, "page") and self.page:
            self.page.dialog = None
            self.page.update()
        self._pending_clear = False

    def _confirm_clear(self, e):
        if hasattr(self, "page") and self.page:
            self.page.dialog = None
            self.page.update()
        self.messages.clear()
        self.message_list.controls.clear()
        self.update()
        self._show_message("Chat history cleared.", info=True)
