"""
ext_event_classifier.py - Event Classification

This extension classifies events as either learning events (contributing to
intelligence) or session events (UI/system events).
"""

from typing import Dict, Any
import hashlib

from extensions.base import GyroExtension


class ext_EventClassifier(GyroExtension):
    """
    Event classification and tagging.
    FOOTPRINT: 10-15 bytes (event tags)
    MAPPING: Identifies and classifies extension events
    """

    def __init__(self):
        """Initialize event classifier."""
        # Define learning event patterns
        self._learning_patterns = {
            "navigation_event",
            "pattern_detected",
            "boundary_crossed",
            "resonance_achieved",
            "compression_milestone",
            "semantic_cluster_formed",
        }

        # Track event statistics
        self._event_stats = {
            "total_events": 0,
            "learning_events": 0,
            "session_events": 0,
            "unknown_events": 0,
        }

        # Recent event cache for pattern detection
        self._recent_events = []
        self._max_recent = 100

    def is_learning_event(self, event_data: Any) -> bool:
        """
        Determine if an event contributes to learning/intelligence.

        Args:
            event_data: The event to classify

        Returns:
            True if this is a learning event, False if session event
        """
        self._event_stats["total_events"] += 1

        # Navigation events are always learning events
        if isinstance(event_data, dict):
            event_type = event_data.get("type", "")

            if event_type in self._learning_patterns:
                self._event_stats["learning_events"] += 1
                self._track_event(event_data, is_learning=True)
                return True

            # Check for navigation-related data
            if "navigation" in event_type or "nav_" in event_type:
                self._event_stats["learning_events"] += 1
                self._track_event(event_data, is_learning=True)
                return True

            # Check for pattern-related data
            if any(key in str(event_data) for key in ["pattern", "boundary", "resonance"]):
                self._event_stats["learning_events"] += 1
                self._track_event(event_data, is_learning=True)
                return True

        # Packed navigation bytes are learning events
        elif isinstance(event_data, int) and 0 <= event_data <= 255:
            self._event_stats["learning_events"] += 1
            self._track_event(event_data, is_learning=True)
            return True

        # Everything else is a session event
        self._event_stats["session_events"] += 1
        self._track_event(event_data, is_learning=False)
        return False

    def classify_event(self, event_data: Any) -> str:
        """
        Classify an event and return its category.

        Returns:
            Event category string
        """
        if self.is_learning_event(event_data):
            if isinstance(event_data, int):
                return "navigation"
            elif isinstance(event_data, dict):
                return event_data.get("type", "learning_other")
            else:
                return "learning_unknown"
        else:
            if isinstance(event_data, dict):
                return event_data.get("type", "session_other")
            else:
                return "session_unknown"

    def _track_event(self, event_data: Any, is_learning: bool) -> None:
        """Track event in recent history."""
        event_record = {
            "data": event_data,
            "is_learning": is_learning,
            "hash": self._hash_event(event_data),
        }

        self._recent_events.append(event_record)

        # Maintain size limit
        if len(self._recent_events) > self._max_recent:
            self._recent_events.pop(0)

    def _hash_event(self, event_data: Any) -> str:
        """Generate a hash for event deduplication."""
        if isinstance(event_data, int):
            data_str = str(event_data)
        elif isinstance(event_data, dict):
            data_str = str(sorted(event_data.items()))
        else:
            data_str = str(event_data)

        return hashlib.md5(data_str.encode()).hexdigest()[:8]

    def get_event_statistics(self) -> Dict[str, int]:
        """Get current event statistics."""
        return self._event_stats.copy()

    def get_recent_learning_events(self, count: int = 10) -> list:
        """Get recent learning events."""
        learning_events = [e for e in self._recent_events if e["is_learning"]]
        return learning_events[-count:]

    def register_learning_pattern(self, pattern: str) -> None:
        """Register a new learning event pattern."""
        self._learning_patterns.add(pattern)

    # --- GyroExtension Interface Implementation ---

    def get_extension_name(self) -> str:
        return "ext_event_classifier"

    def get_extension_version(self) -> str:
        return "1.0.0"

    def get_footprint_bytes(self) -> int:
        # Pattern set + stats + recent events cache
        return 15

    def get_learning_state(self) -> Dict[str, Any]:
        """Return learned patterns and statistics."""
        return {
            "learning_patterns": list(self._learning_patterns),
            "event_statistics": self._event_stats.copy(),
        }

    def get_session_state(self) -> Dict[str, Any]:
        """Return recent event cache."""
        return {"recent_events": self._recent_events[-20:]}  # Last 20 events

    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore learned patterns."""
        if "learning_patterns" in state:
            self._learning_patterns = set(state["learning_patterns"])

        if "event_statistics" in state:
            self._event_stats.update(state["event_statistics"])

    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore recent events."""
        if "recent_events" in state:
            self._recent_events = state["recent_events"]
