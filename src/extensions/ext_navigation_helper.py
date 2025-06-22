"""
ext_navigation_helper.py - Navigation Cycle Utilities

This extension provides utilities and helpers for navigation cycle operations.
"""

from typing import Dict, Any, List, Tuple, Optional
from collections import deque

from extensions.base import GyroExtension


class ext_NavigationHelper(GyroExtension):
    """
    Navigation cycle utilities and helpers.
    FOOTPRINT: 20-30 bytes (navigation cache)
    MAPPING: Provides navigation patterns and utilities
    """

    def __init__(self):
        """Initialize navigation helper."""
        # Navigation history for pattern analysis
        self._nav_history = deque(maxlen=48)  # One complete cycle

        # Phase boundaries from CORE-SPEC-08
        self._phase_boundaries = {
            "CS": [0, 12, 24, 36],
            "UNA_ONA": [3, 9, 15, 21, 27, 33, 39, 45],
            "NESTING": [6, 18, 30, 42],
        }

        # Navigation statistics
        self._nav_stats = {
            "total_cycles": 0,
            "resonance_count": 0,
            "operator_counts": {"stable": 0, "unstable": 0, "neutral": 0},
        }

        # Pattern detection cache
        self._pattern_cache = {}

    def record_navigation(self, phase: int, ops: Optional[Tuple[int, int]]) -> None:
        """
        Record a navigation event.

        Args:
            phase: Current phase
            ops: Operator codes if resonance occurred, None otherwise
        """
        nav_record = {"phase": phase, "ops": ops, "boundary_type": self._get_boundary_type(phase)}

        self._nav_history.append(nav_record)
        self._nav_stats["total_cycles"] += 1

        if ops:
            self._nav_stats["resonance_count"] += 1
            self._classify_operator(ops)

    def _get_boundary_type(self, phase: int) -> Optional[str]:
        """Determine what type of boundary a phase represents."""
        if phase in self._phase_boundaries["CS"]:
            return "CS"
        elif phase in self._phase_boundaries["UNA_ONA"]:
            return "UNA_ONA"
        elif phase in self._phase_boundaries["NESTING"]:
            return "NESTING"
        return None

    def _classify_operator(self, ops: Tuple[int, int]) -> None:
        """Classify and count operator types."""
        op_0, op_1 = ops

        if op_0 == 0 and op_1 == 3:  # Identity + Inverse
            self._nav_stats["operator_counts"]["stable"] += 1
        elif op_0 == 6 and op_1 == 7:  # Both Backward
            self._nav_stats["operator_counts"]["neutral"] += 1
        elif op_0 in [4, 6] and op_1 in [5, 7]:  # Forward/Backward
            self._nav_stats["operator_counts"]["unstable"] += 1

    def get_resonance_info(self) -> Dict[str, Any]:
        """Get information about recent resonance patterns."""
        recent_resonances = [r for r in self._nav_history if r["ops"] is not None]

        return {
            "total_resonances": self._nav_stats["resonance_count"],
            "recent_resonances": recent_resonances[-10:],
            "resonance_rate": self._nav_stats["resonance_count"]
            / max(1, self._nav_stats["total_cycles"]),
        }

    def get_operator_info(self) -> Dict[str, Any]:
        """Get information about operator activations."""
        return {
            "operator_counts": self._nav_stats["operator_counts"].copy(),
            "last_operators": self._get_last_operators(),
        }

    def _get_last_operators(self) -> Optional[Tuple[int, int]]:
        """Get the most recent operator activation."""
        for record in reversed(self._nav_history):
            if record["ops"]:
                return record["ops"]
        return None

    def predict_next_boundary(self, current_phase: int) -> Tuple[int, str]:
        """
        Predict the next phase boundary and its type.

        Returns:
            Tuple of (phase, boundary_type)
        """
        # Find next boundary phase
        all_boundaries = []
        for boundary_type, phases in self._phase_boundaries.items():
            for phase in phases:
                if phase > current_phase:
                    all_boundaries.append((phase, boundary_type))

        # If no boundaries ahead in current cycle, wrap to next cycle
        if not all_boundaries:
            for boundary_type, phases in self._phase_boundaries.items():
                for phase in phases:
                    all_boundaries.append((phase + 48, boundary_type))

        # Return the nearest boundary
        return min(all_boundaries, key=lambda x: x[0])

    def detect_navigation_patterns(self) -> List[Dict[str, Any]]:
        """Detect patterns in navigation history."""
        patterns = []

        # Look for repeated operator sequences
        if len(self._nav_history) >= 4:
            # Check for 2-step patterns
            for i in range(len(self._nav_history) - 3):
                seq1 = (self._nav_history[i]["ops"], self._nav_history[i + 1]["ops"])
                seq2 = (self._nav_history[i + 2]["ops"], self._nav_history[i + 3]["ops"])

                if seq1 == seq2 and seq1[0] is not None:
                    pattern = {"type": "2-step-repeat", "sequence": seq1, "positions": [i, i + 2]}
                    patterns.append(pattern)

        return patterns

    def get_cycle_progress(self, current_phase: int) -> Dict[str, Any]:
        """Get information about progress through the 48-step cycle."""
        return {
            "current_phase": current_phase,
            "cycle_position": current_phase / 48.0,
            "phases_to_complete": 48 - current_phase,
            "current_boundary": self._get_boundary_type(current_phase),
            "next_boundary": self.predict_next_boundary(current_phase),
        }

    # --- GyroExtension Interface Implementation ---

    def get_extension_name(self) -> str:
        return "ext_navigation_helper"

    def get_extension_version(self) -> str:
        return "1.0.0"

    def get_footprint_bytes(self) -> int:
        # Navigation history + stats + cache
        return 30

    def get_learning_state(self) -> Dict[str, Any]:
        """Return navigation statistics and patterns."""
        return {
            "navigation_stats": self._nav_stats.copy(),
            "detected_patterns": self.detect_navigation_patterns(),
        }

    def get_session_state(self) -> Dict[str, Any]:
        """Return recent navigation history."""
        return {"nav_history": list(self._nav_history)}

    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore navigation statistics."""
        if "navigation_stats" in state:
            self._nav_stats = state["navigation_stats"]

    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore navigation history."""
        if "nav_history" in state:
            self._nav_history.clear()
            for record in state["nav_history"]:
                self._nav_history.append(record)

    def ext_on_navigation_event(self, nav_event: int, input_byte: Optional[int] = None) -> None:
        """Process navigation events to update statistics."""
        # This would need phase information from the system
        # For now, just track that we received an event
        self._nav_stats["total_cycles"] += 1
