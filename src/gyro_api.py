from typing import Tuple, Iterable, Union
from pathlib import Path
from core.extension_manager import ExtensionManager
from core.gyro_errors import GyroSessionError

# Ensure _get_manager is defined or imported
# from .some_module import _get_manager  # Uncomment and fix as needed


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


def GenomeEncrypt(genome: bytes, key: bytes) -> bytes:
    """
    Pure encryption function for offline use.
    Implements the GenomeEncrypt utility from CORE-SPEC-06.
    """
    from extensions.ext_cryptographer import ext_Cryptographer

    crypto = ext_Cryptographer(key)
    return crypto.encrypt(genome)


def GenomeDecrypt(cipher: bytes, key: bytes) -> bytes:
    """
    Pure decryption function for offline use.
    Implements the GenomeDecrypt utility from CORE-SPEC-06.
    """
    from extensions.ext_cryptographer import ext_Cryptographer

    crypto = ext_Cryptographer(key)
    return crypto.decrypt(cipher)


def get_language_output(session_id: str, last_n: int = 10) -> list:
    """
    Get recent language output from the system.
    This is how you retrieve what the system has "said".
    """
    manager = _get_manager(session_id)
    from pathlib import Path

    session_path = Path("data/sessions") / session_id / "output.log"

    if not session_path.exists():
        return []

    with open(session_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Return last N lines
    return [line.strip() for line in lines[-last_n:] if line.strip()]


def import_knowledge(bundle_path: Union[str, Path], new_session: bool = True) -> str:
    """Import knowledge package from a .gyro bundle."""
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


def _get_manager(session_id: str):
    """
    Internal helper to get a manager instance with validation.
    """
    if session_id not in _active_sessions:
        raise GyroSessionError(
            f"Session '{session_id}' is not active. "
            f"Active sessions: {list(_active_sessions.keys())}"
        )
    return _active_sessions[session_id]


# Update __all__ to include new functions
__all__ = [
    # ... existing exports ...
    "GenomeEncrypt",
    "GenomeDecrypt",
    "get_language_output",
    "import_knowledge",
    "cleanup_inactive_sessions",
]
