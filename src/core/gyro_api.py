"""
gyro_api.py - Tier 1: The Public API Facade

This module provides the high-level, stable, and user-facing functions for
managing and interacting with GyroSI sessions. It is the designated entry point
for any external application.

Its primary responsibilities are:
- Managing the lifecycle of active sessions (initialization, shutdown).
- Providing simple, command-like functions (e.g., export_knowledge, process_stream).
- Delegating all complex operations to the appropriate ExtensionManager instance.

References:
- CORE-SPEC-05: Baby ML Structure
- CORE-SPEC-06: Baby ML Interface Definitions
- CORE-SPEC-07: Baseline Implementation Specifications
"""

from typing import Optional, Dict, List, Iterable, Union, Tuple, Any
from pathlib import Path

# The ExtensionManager is the workhorse orchestrated by this API
from core.extension_manager import ExtensionManager
from core.gyro_errors import GyroSessionError


# A simple in-memory cache to hold manager instances for active sessions
# This prevents re-initializing the entire system on every call
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
        manager = ExtensionManager(session_id, knowledge_id)
        active_id = manager.get_session_id()

        # Cache the manager instance for subsequent API calls
        _active_sessions[active_id] = manager

        return active_id

    except Exception as e:
        raise GyroSessionError(f"Failed to initialize session: {str(e)}") from e


def shutdown_session(session_id: str) -> None:
    """
    Gracefully shuts down a session, ensuring all state is saved.

    Args:
        session_id: The ID of the session to shut down.

    Raises:
        GyroSessionError: If the session is not active.
    """
    if session_id not in _active_sessions:
        raise GyroSessionError(f"Session '{session_id}' is not active or does not exist.")

    manager = _active_sessions[session_id]

    try:
        # Delegate shutdown logic to the manager
        manager.shutdown()
    except Exception as e:
        raise GyroSessionError(f"Failed to shutdown session: {str(e)}") from e
    finally:
        # Always remove from cache, even if shutdown had issues
        del _active_sessions[session_id]


def list_active_sessions() -> List[str]:
    """
    Returns a list of currently active session IDs.

    Returns:
        List of session UUID strings.
    """
    return list(_active_sessions.keys())


def get_session_info(session_id: str) -> Dict[str, any]:
    """
    Returns information about an active session.

    Args:
        session_id: The ID of the session to query.

    Returns:
        Dictionary containing session metadata.

    Raises:
        GyroSessionError: If the session is not active.
    """
    manager = _get_manager(session_id)

    return {
        "session_id": session_id,
        "knowledge_id": manager.get_knowledge_id(),
        "phase": manager.engine.phase,
        "nav_log_size": manager.navigation_log.step_count,
        "health": manager.get_system_health(),
    }


# ============================================================================
# PROCESSING OPERATIONS
# ============================================================================


def process_byte(session_id: str, input_byte: int) -> Optional[Tuple[int, int]]:
    """
    Processes a single byte through the navigation cycle.

    Args:
        session_id: The ID of the active session to use.
        input_byte:The byte value (0-255) to process.

    Returns:
        Tuple of (op_0, op_1) if navigation occurred, None otherwise.

    Raises:
        GyroSessionError: If the session is not active.
        ValueError: If byte is not in range 0-255.
    """
    if not (0 <= input_byte <= 255):
        raise ValueError(f"Byte must be in range 0-255, got {input_byte}")

    manager = _get_manager(session_id)
    return manager.gyro_operation(input_byte)


def process_byte_stream(session_id: str, byte_stream: Iterable[int]) -> int:
    """
    Processes a stream of bytes through the specified session's navigation cycle.

    This is the main operational function for feeding data into the system.

    Args:
        session_id: The ID of the active session to use.
        byte_stream: An iterable of bytes (0-255) to process.

    Returns:
        The total number of resonant navigation events that were recorded.

    Raises:
        GyroSessionError: If the session_id is not active.
        ValueError: If any byte is not in range 0-255.
    """
    manager = _get_manager(session_id)

    resonance_count = 0
    for i, byte in enumerate(byte_stream):
        if not (0 <= byte <= 255):
            raise ValueError(f"Byte at position {i} must be in range 0-255, got {byte}")

        ops = manager.gyro_operation(byte)
        if ops:
            resonance_count += 1

    return resonance_count


def process_text(session_id: str, text: str, encoding: str = "utf-8") -> int:
    """
    Processes a text string through the navigation cycle.

    Args:
        session_id: The ID of the active session to use.
        text: The text string to process.
        encoding: The text encoding to use (default: utf-8).

    Returns:
        The total number of resonant navigation events.

    Raises:
        GyroSessionError: If the session is not active.
        UnicodeEncodeError: If text cannot be encoded with specified encoding.
    """
    byte_stream = text.encode(encoding)
    return process_byte_stream(session_id, byte_stream)


def process_file(session_id: str, file_path: Union[str, Path], chunk_size: int = 8192) -> int:
    """
    Processes a file through the navigation cycle.

    Args:
        session_id: The ID of the active session to use.
        file_path: Path to the file to process.
        chunk_size: Size of chunks to read at a time (default: 8KB).

    Returns:
        The total number of resonant navigation events.

    Raises:
        GyroSessionError: If the session is not active.
        FileNotFoundError: If the file does not exist.
        IOError: If file cannot be read.
    """
    manager = _get_manager(session_id)
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    resonance_count = 0
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            for byte in chunk:
                ops = manager.gyro_operation(byte)
                if ops:
                    resonance_count += 1

    return resonance_count


# ============================================================================
# KNOWLEDGE MANAGEMENT
# ============================================================================


def export_knowledge(session_id: str, output_path: Union[str, Path]) -> None:
    """
    Exports the knowledge package currently linked to the session.

    Creates a .gyro bundle containing the complete learned intelligence,
    suitable for sharing or archival.

    Args:
        session_id: The ID of the active session.
        output_path: The file path to save the .gyro bundle.

    Raises:
        GyroSessionError: If the session is not active.
        IOError: If the bundle cannot be written.
    """
    manager = _get_manager(session_id)
    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Delegate to manager
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


def query_memory(session_id: str, tag: str) -> any:
    """
    Query any of the five memory systems using TAG syntax.

    This provides direct access to the G1-G5 memory interfaces for advanced usage.

    Args:
        session_id: The ID of the active session.
        tag: TAG expression per CORE-SPEC-04 grammar.
             Format: <temporal>.<invariant>[.<context>]

    Returns:
        The requested data from the memory system.

    Raises:
        GyroSessionError: If the session is not active.
        GyroTagError: If the TAG expression is invalid.
    """
    manager = _get_manager(session_id)

    # Determine which memory system based on tag
    # This is a simplified routing - full implementation would parse the tag
    if "genetic" in tag or any(
        inv in tag
        for inv in [
            "gyrotensor_id",
            "gyrotensor_com",
            "gyrotensor_nest",
            "gyrotensor_add",
            "gyrotensor_quant",
        ]
    ):
        return manager.gyro_genetic_memory(tag)
    elif "epigenetic" in tag:
        return manager.gyro_epigenetic_memory(tag)
    elif "structural" in tag:
        return manager.gyro_structural_memory(tag)
    elif "somatic" in tag:
        return manager.gyro_somatic_memory(tag)
    elif "immunity" in tag:
        return manager.gyro_immunity_memory(tag)
    else:
        # Default to genetic memory
        return manager.gyro_genetic_memory(tag)


def get_navigation_history(
    session_id: str, count: int = 100, reverse: bool = True
) -> List[Tuple[int, int]]:
    """
    Retrieves recent navigation events from the session.

    Args:
        session_id: The ID of the active session.
        count: Maximum number of events to retrieve.
        reverse: If True, returns newest first. If False, oldest first.

    Returns:
        List of (op_0, op_1) tuples representing navigation events.

    Raises:
        GyroSessionError: If the session is not active.
    """
    manager = _get_manager(session_id)

    events = []
    for packed_byte in manager.navigation_log.iter_steps(reverse=reverse):
        op_0 = packed_byte & 0x0F
        op_1 = (packed_byte >> 4) & 0x0F
        events.append((op_0, op_1))

        if len(events) >= count:
            break

    return events


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
    results = {"system_valid": True, "active_sessions": len(_active_sessions), "sessions": {}}

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

    Returns:
        Number of sessions cleaned up.
    """
    to_remove = []

    for session_id, manager in _active_sessions.items():
        try:
            # Try to get health - if it fails, session is invalid
            manager.get_system_health()
        except Exception as e:
            to_remove.append(session_id)

    for session_id in to_remove:
        try:
            _active_sessions[session_id].shutdown()
        except Exception as e:
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
    # Utility functions
    "validate_system_integrity",
    "cleanup_inactive_sessions",
]
