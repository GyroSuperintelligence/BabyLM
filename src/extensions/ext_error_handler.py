"""
ext_error_handler.py - Centralized Error Handling

This extension provides centralized error handling and recovery strategies
for the GyroSI system.
"""

from typing import Dict, Any
import traceback
import time
from collections import deque

from extensions.base import GyroExtension


class ext_ErrorHandler(GyroExtension):
    """
    Centralized error handling and recovery strategies.
    FOOTPRINT: 10-20 bytes (error state cache)
    MAPPING: Intercepts all GyroError exceptions for logging/recovery
    """

    def __init__(self):
        """Initialize error handler."""
        # Error history (limited size for memory efficiency)
        self._error_history = deque(maxlen=100)

        # Error statistics by type
        self._error_stats = {}

        # Recovery strategies
        self._recovery_strategies = {
            "GyroPhaseError": self._recover_phase_error,
            "GyroNavigationError": self._recover_navigation_error,
            "GyroImmutabilityError": self._recover_immutability_error,
            "GyroExtensionError": self._recover_extension_error,
        }

        # Current error state
        self._error_state = {"has_error": False, "last_error": None, "recovery_attempted": False}

    def handle_error(self, error: Exception) -> None:
        """
        Handle an error with logging and recovery attempt.

        Args:
            error: The exception to handle
        """
        # Record error
        error_record = {
            "timestamp": time.time(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

        self._error_history.append(error_record)
        self._update_stats(error_record["type"])

        # Update error state
        self._error_state["has_error"] = True
        self._error_state["last_error"] = error_record

        # Attempt recovery if strategy exists
        error_type = type(error).__name__
        if error_type in self._recovery_strategies:
            try:
                self._recovery_strategies[error_type](error)
                self._error_state["recovery_attempted"] = True
            except Exception as recovery_error:
                # Log recovery failure but don't propagate
                self.log_extension_error("error_handler", recovery_error)

    def log_extension_error(self, extension_name: str, error: Exception) -> None:
        """
        Log an error from a specific extension.

        Args:
            extension_name: Name of the extension that had the error
            error: The exception that occurred
        """
        error_record = {
            "timestamp": time.time(),
            "extension": extension_name,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }

        self._error_history.append(error_record)
        self._update_stats(f"Extension:{extension_name}")

    def _update_stats(self, error_type: str) -> None:
        """Update error statistics."""
        if error_type not in self._error_stats:
            self._error_stats[error_type] = 0
        self._error_stats[error_type] += 1

    def _recover_phase_error(self, error: Exception) -> None:
        """Recovery strategy for phase errors."""
        # Log recovery attempt
        self._log_recovery("Attempting phase error recovery")

        # Phase errors usually mean phase is out of bounds
        # Recovery: Reset to phase 0
        # (Actual implementation would need access to state helper)
        pass

    def _recover_navigation_error(self, error: Exception) -> None:
        """Recovery strategy for navigation errors."""
        self._log_recovery("Attempting navigation error recovery")

        # Navigation errors might mean corrupted log
        # Recovery: Mark for checkpoint at next opportunity
        pass

    def _recover_immutability_error(self, error: Exception) -> None:
        """Recovery strategy for immutability errors."""
        self._log_recovery("Attempting immutability error recovery")

        # Immutability errors mean we tried to modify immutable knowledge
        # Recovery: Force fork on next operation
        pass

    def _recover_extension_error(self, error: Exception) -> None:
        """Recovery strategy for extension errors."""
        self._log_recovery("Attempting extension error recovery")

        # Extension errors are isolated
        # Recovery: Disable the problematic extension
        pass

    def _log_recovery(self, message: str) -> None:
        """Log a recovery attempt."""
        self._error_history.append(
            {"timestamp": time.time(), "type": "RECOVERY", "message": message}
        )

    def get_error_report(self) -> Dict[str, Any]:
        """Get comprehensive error report."""
        return {
            "current_state": self._error_state.copy(),
            "statistics": self._error_stats.copy(),
            "recent_errors": list(self._error_history)[-10:],
            "total_errors": sum(self._error_stats.values()),
        }

    def clear_error_state(self) -> None:
        """Clear current error state after successful recovery."""
        self._error_state["has_error"] = False
        self._error_state["last_error"] = None
        self._error_state["recovery_attempted"] = False

    def has_critical_errors(self) -> bool:
        """Check if there are critical errors requiring attention."""
        # Critical if we have repeated errors of the same type
        for error_type, count in self._error_stats.items():
            if count > 10:  # More than 10 errors of same type
                return True

        # Critical if error rate is high
        if len(self._error_history) == self._error_history.maxlen:
            # Check time span of errors
            if self._error_history[-1]["timestamp"] - self._error_history[0]["timestamp"] < 60:
                # 100 errors in less than a minute
                return True

        return False

    # --- GyroExtension Interface Implementation ---

    def get_extension_name(self) -> str:
        return "ext_error_handler"

    def get_extension_version(self) -> str:
        return "1.0.0"

    def get_footprint_bytes(self) -> int:
        # Error state + limited history
        return 20

    def get_learning_state(self) -> Dict[str, Any]:
        """Return error patterns and statistics."""
        return {"error_statistics": self._error_stats.copy()}

    def get_session_state(self) -> Dict[str, Any]:
        """Return current error state and recent history."""
        return {
            "error_state": self._error_state.copy(),
            "recent_errors": list(self._error_history)[-5:],
        }

    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """Restore error statistics."""
        if "error_statistics" in state:
            self._error_stats = state["error_statistics"]

    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore error state."""
        if "error_state" in state:
            self._error_state = state["error_state"]

        if "recent_errors" in state:
            for error in state["recent_errors"]:
                self._error_history.append(error)
