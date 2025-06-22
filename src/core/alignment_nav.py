"""
alignment_nav.py - The Navigation Log

This module defines the NavigationLog class, the sole mutable record of experience
in the GyroSI system. It represents the gyrotensor_quant (Genome) as an
append-only log of navigation events.

The class is responsible for its own persistence, pruning, and thread-safety,
delegating physical I/O to a provided StorageManager instance.
"""

import threading
from typing import Optional, Iterator, Tuple, Any

# This object needs a handle to the storage extension to persist itself.
from ..extensions.ext_storage_manager import ext_StorageManager
from .gyro_errors import GyroNavigationError


class NavigationLog:
    """
    An append-only, thread-safe navigation log that manages its own persistence
    and bounded growth. It implements the mechanics of gyrotensor_quant.
    """
    def __init__(
        self,
        knowledge_id: str,
        storage_manager: ext_StorageManager,
        max_size: int = 1_048_576  # 1MB default per CORE-SPEC-07
    ):
        """
        Initializes a NavigationLog instance for a specific knowledge package.

        Args:
            knowledge_id: The UUID of the knowledge package this log belongs to.
            storage_manager: An instance of the storage manager for persistence.
            max_size: The maximum size of the log in bytes before pruning.
        """
        self._knowledge_id = knowledge_id
        self._storage = storage_manager
        self._max_size = max_size
        self._prune_threshold = int(max_size * 0.75)  # 75% retention

        # The in-memory representation of the log.
        # This is loaded from disk by the storage manager.
        self._log: bytearray = bytearray()
        self._lock = threading.RLock()
        
        # A dirty flag to optimize persistence.
        self._is_dirty = False

    @property
    def step_count(self) -> int:
        """Returns the current number of navigation steps (bytes) in the log."""
        with self._lock:
            return len(self._log)
            
    def load_from_disk(self):
        """
        Loads the log's contents from storage. This is called by the
        ExtensionManager during session initialization.
        """
        # Implementation will call self._storage.load_raw_navigation_log()
        pass

    def persist_to_disk(self):
        """
        Persists the log's contents to storage if it has changed.
        This can be called periodically or at session shutdown.
        """
        with self._lock:
            if self._is_dirty:
                # Implementation will call self._storage.save_raw_navigation_log()
                self._is_dirty = False

    def append(self, packed_byte: int) -> None:
        """
        Appends a new navigation event (a packed byte) to the log.

        This is the primary write method. It handles thread-safety, pruning,
        and marks the log as dirty for persistence.

        Args:
            packed_byte: The 8-bit integer containing two 4-bit operator codes.
        """
        with self._lock:
            if len(self._log) >= self._max_size:
                self._prune()

            self._log.append(packed_byte)
            self._is_dirty = True

    def _prune(self) -> None:
        """
        Internal method to prune the oldest 25% of the log when at capacity.
        This is called automatically by append().
        """
        # This is a locked context, so no need for another lock.
        self._log = self._log[-self._prune_threshold:]

    def iter_steps(self, reverse: bool = False) -> Iterator[int]:
        """
        Provides a thread-safe iterator over the navigation log.
        It operates on a snapshot of the log to prevent modification issues.

        Args:
            reverse: If True, iterates from newest to oldest event.

        Returns:
            An iterator of packed-byte integers.
        """
        with self._lock:
            snapshot = self._log[:]  # Create a copy for safe iteration
        
        if reverse:
            yield from reversed(snapshot)
        else:
            yield from snapshot

    def get_as_tensor(self) -> "torch.Tensor":
        """
        Decodes the entire log and reconstructs the gyrotensor_quant tensor.
        This is a computationally expensive operation used for analysis or
        when G1 needs to represent the full decoded state.
        """
        # Implementation would import torch, convert bytearray to a tensor,
        # and unpack the nibbles into the (N, 2) shape.
        pass

# Define the public surface of this module.
__all__ = [
    'NavigationLog',
]