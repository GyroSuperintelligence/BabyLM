"""
Language egress extension - converts navigation patterns to text output.
This enables the system to respond with language like you requested.
"""

import threading
from typing import Dict, Any, Optional
from .base import GyroExtension


class ext_LanguageEgress(GyroExtension):
    """
    Converts navigation cycles into language output.
    Accumulates navigation events and emits text when patterns complete.
    """

    def __init__(self):
        """Initialize language egress system"""
        self.cycle_buffer = []  # Accumulates 48-phase cycles
        self.text_buffer = bytearray()  # Accumulates text bytes
        self.lock = threading.RLock()
        self.manager: Optional[Any] = None

        # Language detection patterns
        self.sentence_endings = {46, 33, 63, 10, 13}  # . ! ? \\n \\r
        self.word_separators = {32, 9, 10, 13}  # space, tab, newlines

    def get_extension_name(self) -> str:
        return "ext_LanguageEgress"

    def get_extension_version(self) -> str:
        return "1.0.0"

    def get_footprint_bytes(self) -> int:
        return len(self.cycle_buffer) + len(self.text_buffer)

    def process_navigation_event(self, nav_event: int, input_byte: Optional[int] = None):
        """
        Process navigation events and convert to language output.

        Args:
            nav_event: Packed navigation event byte
            input_byte: Original input byte that caused navigation
        """
        with self.lock:
            self.cycle_buffer.append(nav_event)

            # Check if we have a complete 48-phase cycle
            if len(self.cycle_buffer) >= 48:
                self._process_complete_cycle()

    def _process_complete_cycle(self) -> None:
        """Process a complete 48-phase navigation cycle"""
        if len(self.cycle_buffer) < 48:
            return

        # Extract the completed cycle
        cycle = self.cycle_buffer[:48]
        self.cycle_buffer = self.cycle_buffer[48:]

        # Convert navigation pattern to text bytes
        text_bytes = self._navigation_to_text(cycle)
        self.text_buffer.extend(text_bytes)

        # Check for complete sentences and emit
        self._emit_complete_sentences()

    def _navigation_to_text(self, navigation_cycle: list) -> bytes:
        """
        Convert 48-phase navigation cycle to text bytes.
        This is where the mechanical navigation becomes language.
        """
        # Simple mapping: navigation patterns -> UTF-8 bytes
        # In a full implementation, this would use learned patterns

        text_bytes = bytearray()

        # Group navigation events into character-generating patterns
        for i in range(0, len(navigation_cycle), 4):
            chunk = navigation_cycle[i : i + 4]

            # Convert 4 navigation events to one character
            char_code = self._chunk_to_char(chunk)
            if 32 <= char_code <= 126:  # Printable ASCII
                text_bytes.append(char_code)

        return bytes(text_bytes)

    def _chunk_to_char(self, nav_chunk: list) -> int:
        """Convert 4 navigation events to a character code"""
        if len(nav_chunk) < 4:
            return 32  # Space for incomplete chunks

        # Combine navigation events into character code
        char_code = 0
        for i, nav in enumerate(nav_chunk):
            # Extract operator types and combine
            op0 = nav & 0x0F
            op1 = (nav >> 4) & 0x0F
            char_code += (op0 + op1) * (2**i)

        # Map to printable ASCII range
        return 32 + (char_code % 95)  # ASCII 32-126

    def _emit_complete_sentences(self) -> None:
        """Emit complete sentences from text buffer"""
        if not self.text_buffer:
            return

        # Look for sentence endings
        last_sentence_end = -1
        for i, byte_val in enumerate(self.text_buffer):
            if byte_val in self.sentence_endings:
                last_sentence_end = i

        if last_sentence_end >= 0:
            # Extract complete sentence(s)
            sentence_bytes = self.text_buffer[: last_sentence_end + 1]
            self.text_buffer = self.text_buffer[last_sentence_end + 1 :]

            # Convert to text and emit
            try:
                text = sentence_bytes.decode("utf-8", errors="ignore")
                if text.strip():
                    self._emit_text(text.strip())
            except Exception:
                pass  # Skip invalid text

    def _emit_text(self, text: str) -> None:
        """
        Emit text through the structural memory system.
        This is how the system "speaks" - like you requested.
        """
        # Push text to structural memory for UI display
        if hasattr(self, "manager") and self.manager:
            try:
                self.manager.gyro_structural_memory("current.gyrotensor_nest.output", data=text)
            except Exception:
                # Fallback: direct output
                print(f"[GyroSI]: {text}")
        else:
            # Direct output when no manager available
            print(f"[GyroSI]: {text}")

    def get_learning_state(self) -> Dict[str, Any]:
        """Learning state for knowledge export"""
        return {
            "language_patterns": self._extract_language_patterns(),
            "character_mappings": self._get_character_mappings(),
        }

    def get_session_state(self) -> Dict[str, Any]:
        """Session state (non-exportable)"""
        return {"cycle_buffer": self.cycle_buffer.copy(), "text_buffer": bytes(self.text_buffer)}

    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore learning state"""
        # In full implementation, restore learned language patterns
        pass

    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore session state"""
        self.cycle_buffer = state.get("cycle_buffer", [])
        self.text_buffer = bytearray(state.get("text_buffer", b""))

    def get_pattern_filename(self) -> str:
        """Pattern filename for knowledge export"""
        return "ext_language_egress@1.0.0.patterns"

    def _extract_language_patterns(self) -> Dict[str, Any]:
        """Extract learned language patterns"""
        # Placeholder for learned patterns
        return {}

    def _get_character_mappings(self) -> Dict[str, Any]:
        """Get navigation->character mappings"""
        # Placeholder for learned character mappings
        return {}
