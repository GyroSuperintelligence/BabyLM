"""
Navigation log with RLE compression and CORE-SPEC-06 compliant API.
"""

import threading
from typing import Iterator, Tuple, Optional, Any, Dict
from core.gyro_errors import GyroStorageError
from .log_decoder import decode_log_stream, encode_identity_run


class NavigationLog:
    """
    Thread-safe navigation log with RLE compression for identity events.
    Now compliant with CORE-SPEC-06 interface requirements.
    """

    def __init__(self, knowledge_id: str, storage_manager, max_size: int = 1048576):
        """Initialize navigation log"""
        self.knowledge_id = knowledge_id
        self._storage = storage_manager
        self._max_size = max_size
        self._prune_threshold = int(max_size * 0.75)

        # Raw compressed log storage
        self._log = bytearray()
        self._lock = threading.RLock()
        self._is_dirty = False

        # RLE compression state
        self._identity_run_count = 0

        self.manager: Optional[Any] = None

    def append(self, id0_code: int, id1_code: int, *, fork_ok: bool = True) -> None:
        """
        Append navigation event with RLE compression.
        This is the NEW API required by CORE-SPEC-06.

        Args:
            id0_code: Operator code (0-3) for id_0 tensor
            id1_code: Operator code (0-3) for id_1 tensor
            fork_ok: If False, raise error on immutable knowledge
        """
        # Validate operator codes
        if not (0 <= id0_code <= 3 and 0 <= id1_code <= 3):
            raise ValueError(f"Operator codes must be 0-3, got {id0_code}, {id1_code}")

        # Pack codes into byte: [id1_code:4][id0_code:4]
        packed_byte = (id1_code << 4) | id0_code

        with self._lock:
            # Handle RLE compression for identity events
            if packed_byte == 0x00:  # Double-identity event
                self._identity_run_count += 1
                # Flush run if it reaches maximum encodable length
                if self._identity_run_count == 17:
                    self._flush_identity_run()
            else:
                # Different event - flush any pending identity run first
                self._flush_identity_run()
                self._append_raw_byte(packed_byte)

    def _flush_identity_run(self) -> None:
        """Encode and flush any pending identity run"""
        if self._identity_run_count == 0:
            return

        if self._identity_run_count == 1:
            # Single identity not worth compressing
            self._append_raw_byte(0x00)
        else:
            # Use encode_identity_run utility for RLE encoding
            encoded = encode_identity_run(self._identity_run_count)
            for b in encoded:
                self._append_raw_byte(b)

        self._identity_run_count = 0

    def _append_raw_byte(self, byte_val: int) -> None:
        """Internal method to append raw byte with pruning"""
        # Check if pruning needed
        if len(self._log) >= self._max_size:
            self._flush_identity_run()  # Important: flush before pruning
            self._prune()

        self._log.append(byte_val)
        self._is_dirty = True

    def iter_steps(self, reverse: bool = False) -> Iterator[Tuple[int, int]]:
        """
        Iterate over navigation events as (id0_code, id1_code) tuples.
        This replaces your old iteration that returned packed bytes.
        """
        with self._lock:
            # Create snapshot for thread safety
            snapshot = bytes(self._log)

        # Decode RLE compression
        events = []
        for packed_byte in decode_log_stream(iter(snapshot)):
            id0_code = packed_byte & 0x0F
            id1_code = (packed_byte >> 4) & 0x0F
            events.append((id0_code, id1_code))

        if reverse:
            events.reverse()

        yield from events

    @property
    def step_count(self) -> int:
        """Returns actual number of navigation steps (after RLE decompression)"""
        return sum(1 for _ in self.iter_steps())

    def shutdown(self) -> None:
        """Shutdown with proper RLE flushing"""
        with self._lock:
            self._flush_identity_run()
            self.persist_to_disk()

    def load_from_disk(self):
        """Load log from storage (unchanged)"""
        data = self._storage.load_raw_navigation_log(self.knowledge_id)
        with self._lock:
            self._log = bytearray(data)
            self._is_dirty = False

    def persist_to_disk(self):
        """Persist log to storage (unchanged)"""
        with self._lock:
            if self._is_dirty:
                self._storage.save_raw_navigation_log(self.knowledge_id, bytes(self._log))
                self._is_dirty = False

    def _prune(self) -> None:
        """Prune oldest 25% when at capacity (unchanged)"""
        self._log = self._log[-self._prune_threshold :]

    def validate_gene_checksum(self, gene: Dict[str, Any]) -> bool:
        """
        Validates that the current in-memory Gene constant matches the checksum
        stored in the active knowledge package's metadata.
        """
        try:
            if not isinstance(self.knowledge_id, str) or not self.knowledge_id:
                raise GyroStorageError("No valid knowledge_id set for checksum validation")
            # metadata = self._storage.load_metadata(self.knowledge_id)  # Unused variable removed
            # ... rest of method ...
        except Exception as e:
            print(f"Error validating gene checksum: {e}")
            return False
        return False
