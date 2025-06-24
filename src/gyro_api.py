"""
GyroSI Baby ML API

This module provides the primary API for interacting with the GyroSI Baby ML system.
It handles session management, processing operations, and knowledge management.

The API is designed to be simple and intuitive while providing access to the full
power of the GyroSI system.
"""

import os
import uuid
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Iterable
from pathlib import Path

from core.extension_manager import ExtensionManager
from core.gyro_errors import GyroSessionError, GyroError
from core.alignment_nav import NavigationLog

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Active session registry
_active_sessions: Dict[str, ExtensionManager] = {}

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================


def initialize_session(session_id: Optional[str] = None, knowledge_id: Optional[str] = None) -> str:
    """
    Initializes a new or existing GyroSI session.

    This is the primary entry point for starting work with the system. It creates
    the full stack (Engine, Extension Manager, Extensions) for a given session.

    Args:
        session_id: The UUID of an existing session to resume. If None, a new
                   session is created.
        knowledge_id: The UUID of a knowledge package to link to. If None,
                     a new, empty knowledge package is created.

    Returns:
        The active session_id.

    Raises:
        GyroSessionError: If initialization fails.
    """
    try:
        # Check if session already active
        if session_id and session_id in _active_sessions:
            return session_id

        # Create a new ExtensionManager instance
        # This constructor does all the heavy lifting of loading state
        # and initializing extensions
        manager = ExtensionManager(session_id, knowledge_id or "")
        active_id = manager.get_session_id()

        # Cache the manager instance for subsequent API calls
        _active_sessions[active_id] = manager

        return active_id

    except Exception as e:
        raise GyroSessionError(f"Failed to initialize session: {str(e)}") from e


def shutdown_session(session_id: str) -> None:
    """
    Shuts down an active session and cleans up resources.

    Args:
        session_id: The UUID of the session to shutdown.

    Raises:
        GyroSessionError: If the session is not active.
    """
    manager = _get_manager(session_id)

    try:
        manager.shutdown()
    finally:
        # Always remove from active sessions, even if shutdown fails
        if session_id in _active_sessions:
            del _active_sessions[session_id]


def list_active_sessions() -> List[str]:
    """
    Returns a list of all active session IDs.

    Returns:
        List of active session UUIDs.
    """
    return list(_active_sessions.keys())


def get_session_info(session_id: str) -> Dict[str, Any]:
    """
    Gets detailed information about an active session.

    Args:
        session_id: The UUID of the session.

    Returns:
        Dictionary containing session information.

    Raises:
        GyroSessionError: If the session is not active.
    """
    manager = _get_manager(session_id)

    return {
        "session_id": session_id,
        "knowledge_id": manager.get_knowledge_id(),
        "health": manager.get_system_health(),
    }


# ============================================================================
# PROCESSING OPERATIONS
# ============================================================================


def process_byte(session_id: str, input_byte: int) -> Tuple[int, int]:
    """
    Processes a single byte through the navigation cycle.
    Now ALWAYS returns operator codes (never None).
    """
    if not (0 <= input_byte <= 255):
        raise ValueError(f"Byte must be in range 0-255, got {input_byte}")

    manager = _get_manager(session_id)
    return manager.gyro_operation(input_byte)


def process_byte_stream(session_id: str, byte_stream: Iterable[int]) -> int:
    """
    Processes a stream of bytes.
    Every byte now generates a navigation event.
    """
    manager = _get_manager(session_id)

    navigation_count = 0
    for i, byte in enumerate(byte_stream):
        if not (0 <= byte <= 255):
            raise ValueError(f"Byte at position {i} must be in range 0-255, got {byte}")

        # Every byte now produces navigation
        manager.gyro_operation(byte)
        navigation_count += 1

    return navigation_count


def process_text(session_id: str, text: str) -> int:
    """
    Processes text through the navigation cycle.
    Each character is converted to its byte representation.
    """
    if not isinstance(text, str):
        raise ValueError(f"Text must be a string, got {type(text)}")

    byte_stream = (ord(char) for char in text)
    return process_byte_stream(session_id, byte_stream)


def process_file(session_id: str, file_path: Union[str, Path], chunk_size: int = 8192) -> int:
    """
    Processes a file through the navigation cycle.
    Every byte now generates a navigation event.
    """
    manager = _get_manager(session_id)
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    navigation_count = 0
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            for byte in chunk:
                # Every byte now produces navigation
                manager.gyro_operation(byte)
                navigation_count += 1

    return navigation_count


# ============================================================================
# KNOWLEDGE MANAGEMENT
# ============================================================================


def export_knowledge(session_id: str, output_path: Union[str, Path]) -> None:
    """
    Exports the current knowledge package to a .gyro bundle.

    Args:
        session_id: The ID of the active session.
        output_path: Path where the .gyro bundle will be saved.

    Raises:
        GyroSessionError: If the session is not active.
        GyroError: If export fails.
    """
    manager = _get_manager(session_id)
    manager.export_knowledge(str(output_path))


def import_knowledge(bundle_path: Union[str, Path], new_session: bool = True) -> str:
    """
    Imports a knowledge package from a .gyro bundle.

    Args:
        bundle_path: Path to the .gyro bundle file.
        new_session: If True, creates a new session for the imported knowledge.
                    If False, returns just the knowledge ID.

    Returns:
        The session ID if new_session=True, otherwise the knowledge ID.

    Raises:
        FileNotFoundError: If the bundle file does not exist.
        GyroError: If the bundle is invalid or incompatible.
    """
    bundle_path = Path(bundle_path)

    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle file not found: {bundle_path}")

    if new_session:
        # Create a temporary manager just for import
        temp_manager = ExtensionManager()
        knowledge_id = temp_manager.import_knowledge(str(bundle_path))
        temp_manager.shutdown()

        # Create new session with imported knowledge
        return initialize_session(knowledge_id=knowledge_id)
    else:
        # Just import and return knowledge ID
        temp_manager = ExtensionManager()
        knowledge_id = temp_manager.import_knowledge(str(bundle_path))
        temp_manager.shutdown()
        return knowledge_id


def fork_knowledge(session_id: str, new_session: bool = False) -> str:
    """
    Forks the current knowledge package to create a new learning path.

    The current session will automatically be linked to the new fork unless
    new_session=True.

    Args:
        session_id: The ID of the active session.
        new_session: If True, creates a new session linked to the fork instead
                    of re-linking the current one.

    Returns:
        The UUID of the newly created knowledge package.

    Raises:
        GyroSessionError: If the session is not active.
    """
    manager = _get_manager(session_id)
    new_knowledge_id = manager.fork_knowledge(new_session=new_session)

    if new_session:
        # Create and return new session ID
        return initialize_session(knowledge_id=new_knowledge_id)
    else:
        return new_knowledge_id


def link_session_to_knowledge(session_id: str, knowledge_id: str) -> None:
    """
    Links an active session to a different knowledge package.

    This resets the session phase to 0 and switches to the new knowledge context.

    Args:
        session_id: The ID of the active session.
        knowledge_id: The UUID of the knowledge package to link to.

    Raises:
        GyroSessionError: If the session is not active.
        GyroError: If the knowledge package does not exist.
    """
    manager = _get_manager(session_id)
    manager.link_to_knowledge(knowledge_id)


# ============================================================================
# QUERY OPERATIONS
# ============================================================================


def query_memory(session_id: str, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Queries the system's memory for relevant information.

    Args:
        session_id: The ID of the active session.
        query: The search query.
        max_results: Maximum number of results to return.

    Returns:
        List of memory entries matching the query.

    Raises:
        GyroSessionError: If the session is not active.
    """
    manager = _get_manager(session_id)
    # This method doesn't exist yet - return empty list for now
    return []


def get_navigation_history(session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Gets the navigation history for a session.

    Args:
        session_id: The ID of the active session.
        limit: Maximum number of entries to return.

    Returns:
        List of navigation history entries.

    Raises:
        GyroSessionError: If the session is not active.
    """
    manager = _get_manager(session_id)
    # This method doesn't exist yet - return empty list for now
    return []


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def validate_system_integrity() -> Tuple[bool, Dict[str, Any]]:
    """
    Performs system-wide integrity validation.

    This checks all active sessions and the overall system configuration.

    Returns:
        Tuple of (system_valid, results dictionary).
    """
    results: Dict[str, Any] = {
        "system_valid": True,
        "active_sessions": len(_active_sessions),
        "sessions": {},
    }

    # Check each active session
    for session_id, manager in _active_sessions.items():
        try:
            health = manager.get_system_health()
            results["sessions"][session_id] = {"valid": True, "health": health}
        except Exception as e:
            results["sessions"][session_id] = {"valid": False, "error": str(e)}
            results["system_valid"] = False

    return results["system_valid"], results


def cleanup_inactive_sessions() -> int:
    """
    Removes any sessions that are no longer valid.
    Returns number of sessions cleaned up.
    """
    to_remove = []

    for session_id, manager in _active_sessions.items():
        try:
            # Try to get health - if it fails, session is invalid
            manager.get_system_health()
        except Exception:
            to_remove.append(session_id)

    for session_id in to_remove:
        try:
            _active_sessions[session_id].shutdown()
        except Exception:
            pass
        del _active_sessions[session_id]

    return len(to_remove)


def _get_manager(session_id: str) -> ExtensionManager:
    """
    Internal helper to get a manager instance with validation.

    Args:
        session_id: The session ID to look up.

    Returns:
        The ExtensionManager instance.

    Raises:
        GyroSessionError: If session not found.
    """
    if session_id not in _active_sessions:
        raise GyroSessionError(
            f"Session '{session_id}' is not active. "
            f"Active sessions: {list(_active_sessions.keys())}"
        )
    return _active_sessions[session_id]


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================


# Ensure data directories exist on module import
def _ensure_data_directories():
    """Create required data directories if they don't exist."""
    data_root = Path("data")
    (data_root / "sessions").mkdir(parents=True, exist_ok=True)
    (data_root / "knowledge").mkdir(parents=True, exist_ok=True)

    # Create .gitkeep files
    (data_root / "sessions" / ".gitkeep").touch(exist_ok=True)
    (data_root / "knowledge" / ".gitkeep").touch(exist_ok=True)


# Initialize on import
_ensure_data_directories()


# ============================================================================
# PUBLIC API SUMMARY
# ============================================================================
__all__ = [
    # Session management
    "initialize_session",
    "shutdown_session",
    "list_active_sessions",
    "get_session_info",
    # Processing operations
    "process_byte",
    "process_byte_stream",
    "process_text",
    "process_file",
    # Knowledge management
    "export_knowledge",
    "import_knowledge",
    "fork_knowledge",
    "link_session_to_knowledge",
    # Query operations
    "query_memory",
    "get_navigation_history",
    "get_language_output",
    # Utility functions
    "validate_system_integrity",
    "cleanup_inactive_sessions",
]


def get_language_output(session_id: str, last_n: int = 10) -> list:
    """
    Get recent language output from the system.
    This is how you retrieve what the system has 'said'.
    """
    session_path = Path("data/sessions") / session_id / "output.log"

    if not session_path.exists():
        return []

    with open(session_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Return last N lines
    return [line.strip() for line in lines[-last_n:] if line.strip()]
