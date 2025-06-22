"""
alignment_nav.py - Navigation Log Implementation

The NavigationLog class manages the mutable navigation history.
"""

import threading
from typing import Optional, Iterator, Dict, Any
from pathlib import Path


class NavigationLog:
    """
    Thread-safe navigation log with bounded growth and intelligent pruning.
    Implements 75% retention strategy for optimal extension context preservation.
    """
    
    def __init__(self, knowledge_id: str, storage_manager, max_size: int = 1048576):
        """Initialize navigation log for a knowledge package."""
        self.knowledge_id = knowledge_id
        self._storage = storage_manager
        self._max_size = max_size
        self._prune_threshold = int(max_size * 0.75)
        
        # In-memory log
        self._log = bytearray()
        self._lock = threading.RLock()
        self._is_dirty = False
    
    @property
    def step_count(self) -> int:
        """Returns current number of navigation steps."""
        with self._lock:
            return len(self._log)
    
    def load_from_disk(self):
        """Load log contents from storage."""
        data = self._storage.load_raw_navigation_log(self.knowledge_id)
        with self._lock:
            self._log = bytearray(data)
            self._is_dirty = False
    
    def persist_to_disk(self):
        """Persist log contents to storage if dirty."""
        with self._lock:
            if self._is_dirty:
                self._storage.save_raw_navigation_log(self.knowledge_id, bytes(self._log))
                self._is_dirty = False
    
    def append(self, packed_byte: int) -> None:
        """Append navigation event with pruning."""
        with self._lock:
            if len(self._log) >= self._max_size:
                self._prune()
            
            self._log.append(packed_byte)
            self._is_dirty = True
    
    def _prune(self) -> None:
        """Prune oldest 25% when at capacity."""
        self._log = self._log[-self._prune_threshold:]
    
    def iter_steps(self, reverse: bool = False) -> Iterator[int]:
        """Thread-safe iteration over navigation log."""
        with self._lock:
            snapshot = self._log[:]
        
        if reverse:
            yield from reversed(snapshot)
        else:
            yield from snapshot