"""
ext_multi_resolution.py - Linguistic Boundary Detection

This extension detects linguistic boundaries (character, word, sentence) at
multiple resolutions using defect accumulation derived from the 48-step cycle.
"""

from typing import Dict, Any, Optional
from collections import deque

from extensions.base import GyroExtension


class ext_MultiResolution(GyroExtension):
    """
    Linguistic boundary detection with thresholds derived from 48-step cycle.
    FOOTPRINT: 3 bytes (defect accumulators)
    MAPPING: Analyzes navigation sequences from G2
    """

    def __init__(self):
        """Initialize multi-resolution processor."""
        # Thresholds derived from 48-step cycle mathematics
        self.CHAR_THRESHOLD = 6  # 48/8 - character boundary detection
        self.WORD_THRESHOLD = 12  # 48/4 - word boundary detection
        self.SENT_THRESHOLD = 48  # Full cycle - sentence boundary detection

        # Operation weights for defect accumulation
        self.OPERATION_WEIGHTS = [1, 2, 3, 3]  # Identity, Inverse, Forward, Backward

        # State classification per CORE-SPEC-05
        self.char_defect = 0  # Learning state (1 byte)
        self.word_defect = 0  # Learning state (1 byte)
        self.sent_defect = 0  # Learning state (1 byte)
        self.ui_boundaries = []  # Session state

        # Boundary detection history
        self._boundary_history = deque(maxlen=100)

        # Statistics
        self._boundary_stats = {
            "char_count": 0,
            "word_count": 0,
            "sent_count": 0,
            "total_defects": 0,
        }

    def ext_on_navigation_event(self, nav_event: int, input_byte: Optional[int] = None) -> None:
        """
        Process navigation event for boundary detection.

        Args:
            nav_event: Packed navigation byte (two 4-bit codes)
            input_byte:Original input byte that caused this navigation
        """
        # Extract operation codes
        op_0 = nav_event & 0x0F
        op_1 = (nav_event >> 4) & 0x0F

        # Get operation types (bits 3:1)
        op_type_0 = (op_0 >> 1) & 0x07
        op_type_1 = (op_1 >> 1) & 0x07

        # Accumulate defects based on operation weights
        weight_0 = self.OPERATION_WEIGHTS[min(op_type_0, 3)]
        weight_1 = self.OPERATION_WEIGHTS[min(op_type_1, 3)]
        total_weight = (weight_0 + weight_1) // 2

        self.char_defect += total_weight
        self.word_defect += total_weight
        self.sent_defect += total_weight
        self._boundary_stats["total_defects"] += total_weight

        # Check for character boundary
        if self.char_defect >= self.CHAR_THRESHOLD:
            self._record_boundary("char", self.char_defect, byte)
            self.char_defect = 0

        # Check for word boundary (with whitespace hint)
        if byte is not None and byte in [32, 9, 10, 13]:  # Space, tab, newline, return
            if self.word_defect >= self.WORD_THRESHOLD:
                self._record_boundary("word", self.word_defect, byte)
                self.word_defect = 0

        # Check for sentence boundary (with punctuation hint)
        if byte is not None and byte in [46, 33, 63]:  # Period, exclamation, question
            if self.sent_defect >= self.SENT_THRESHOLD:
                self._record_boundary("sentence", self.sent_defect, byte)
                self.sent_defect = 0

    def _record_boundary(
        self, boundary_type: str, defect_value: int, input_byte: Optional[int]
    ) -> None:
        """Record a detected boundary."""
        boundary_record = {
            "type": boundary_type,
            "defect": defect_value,
            "byte": byte,
            "char": chr(byte) if byte and 32 <= byte <= 126 else None,
        }

        self.ui_boundaries.append(boundary_record)
        self._boundary_history.append(boundary_record)

        # Update statistics
        if boundary_type == "char":
            self._boundary_stats["char_count"] += 1
        elif boundary_type == "word":
            self._boundary_stats["word_count"] += 1
        elif boundary_type == "sentence":
            self._boundary_stats["sent_count"] += 1

    def ext_get_boundary_analysis(self) -> Dict[str, Any]:
        """Get comprehensive boundary analysis."""
        return {
            "current_defects": {
                "char": self.char_defect,
                "word": self.word_defect,
                "sentence": self.sent_defect,
            },
            "progress": {
                "char": self.char_defect / self.CHAR_THRESHOLD,
                "word": self.word_defect / self.WORD_THRESHOLD,
                "sentence": self.sent_defect / self.SENT_THRESHOLD,
            },
            "statistics": self._boundary_stats.copy(),
            "recent_boundaries": list(self._boundary_history)[-10:],
        }

    def ext_predict_next_boundary(self) -> Dict[str, float]:
        """
        Predict distance to next boundary of each type.

        Returns:
            Dictionary with predicted defects needed for each boundary type
        """
        return {
            "char": max(0, self.CHAR_THRESHOLD - self.char_defect),
            "word": max(0, self.WORD_THRESHOLD - self.word_defect),
            "sentence": max(0, self.SENT_THRESHOLD - self.sent_defect),
        }

    def ext_get_text_structure(self) -> Dict[str, Any]:
        """
        Analyze text structure from boundary patterns.

        Returns:
            Text structure metrics
        """
        if self._boundary_stats["word_count"] == 0:
            avg_word_length = 0
        else:
            avg_word_length = (
                self._boundary_stats["char_count"] / self._boundary_stats["word_count"]
            )

        if self._boundary_stats["sent_count"] == 0:
            avg_sent_length = 0
        else:
            avg_sent_length = (
                self._boundary_stats["word_count"] / self._boundary_stats["sent_count"]
            )

        return {
            "average_word_length": avg_word_length,
            "average_sentence_length": avg_sent_length,
            "total_characters": self._boundary_stats["char_count"],
            "total_words": self._boundary_stats["word_count"],
            "total_sentences": self._boundary_stats["sent_count"],
        }

    # --- GyroExtension Interface Implementation ---

    def get_extension_name(self) -> str:
        return "ext_multi_resolution"

    def get_extension_version(self) -> str:
        return "1.0.0"

    def get_footprint_bytes(self) -> int:
        # Exactly 3 bytes for the three defect accumulators
        return 3

    def get_learning_state(self) -> Dict[str, Any]:
        """Learning state exported with knowledge per CORE-SPEC-05."""
        return {
            "char_defect": self.char_defect,
            "word_defect": self.word_defect,
            "sent_defect": self.sent_defect,
            "boundary_stats": self._boundary_stats.copy(),
        }

    def get_session_state(self) -> Dict[str, Any]:
        """Session state per CORE-SPEC-05 knowledge/session separation."""
        return {
            "ui_boundaries": self.ui_boundaries[-50:],  # Last 50 boundaries
            "boundary_history": list(self._boundary_history)[-20:],
        }

    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore learning state from knowledge package."""
        if "char_defect" in state:
            self.char_defect = state["char_defect"]
        if "word_defect" in state:
            self.word_defect = state["word_defect"]
        if "sent_defect" in state:
            self.sent_defect = state["sent_defect"]
        if "boundary_stats" in state:
            self._boundary_stats = state["boundary_stats"]

    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore session state."""
        if "ui_boundaries" in state:
            self.ui_boundaries = state["ui_boundaries"]
        if "boundary_history" in state:
            self._boundary_history.clear()
            for boundary in state["boundary_history"]:
                self._boundary_history.append(boundary)
