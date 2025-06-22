"""
ext_fork_manager.py - System Extension for Knowledge Forking

This extension implements the fork-on-write immutability contract for
knowledge packages. It ensures that published/imported knowledge is never
modified, and that new learning always occurs on a new, traceable fork.
"""

from typing import Optional

# This extension needs to interact with the Storage Manager and NavigationLog.
from .ext_storage_manager import ext_StorageManager
from ..core.alignment_nav import NavigationLog
from .base import GyroExtension


class ext_ForkManager(GyroExtension):
    """
    Manages the fork-on-write lifecycle of knowledge packages.
    """
    def __init__(self, storage_manager: ext_StorageManager):
        """
        Initializes the fork manager.

        Args:
            storage_manager: A handle to the storage manager for file operations.
        """
        self._storage = storage_manager
        
        # Load metadata to check if the current knowledge package is immutable.
        # A package might be considered immutable if it was imported, or if
        # a specific flag is set in its metadata.
        metadata = self._storage.load_metadata(self._storage.knowledge_id)
        self._is_immutable = metadata.get("immutable", False)

    def is_current_knowledge_immutable(self) -> bool:
        """Checks if the currently linked knowledge package is immutable."""
        return self._is_immutable

    def ensure_writable(self, current_nav_log: NavigationLog) -> NavigationLog:
        """
        The cornerstone method of this extension.

        Checks if the current knowledge is immutable. If it is, this method
        triggers a full fork and returns the NEW NavigationLog object for the
        forked knowledge. If not immutable, it returns the original log.

        This guarantees the ExtensionManager always has a writable target.

        Args:
            current_nav_log: The NavigationLog object for the current knowledge.

        Returns:
            A writable NavigationLog object (either the original or a new one).
        """
        if not self.is_current_knowledge_immutable():
            return current_nav_log
        
        # If we are here, a fork is required.
        new_knowledge_id = self.fork()
        
        # The ExtensionManager will need to be notified that its context has
        # changed. This method returns the new log to facilitate that.
        # It's crucial that the storage manager's context is also updated.
        self._storage.switch_knowledge_context(new_knowledge_id)
        
        # Create and return a new, empty NavigationLog object for the new fork.
        new_nav_log = NavigationLog(
            knowledge_id=new_knowledge_id,
            storage_manager=self._storage
        )
        new_nav_log.load_from_disk() # Load any (likely empty) state for the new fork.
        
        # The fork is no longer inherently immutable until published/exported.
        self._is_immutable = False
        
        return new_nav_log

    def fork(self) -> str:
        """
        Executes the physical forking process by delegating to the storage manager.
        It handles metadata updates for provenance.

        Returns:
            The UUID of the newly created knowledge package.
        """
        source_id = self._storage.knowledge_id
        
        # 1. Delegate physical directory forking to the storage manager.
        new_id = self._storage.fork_knowledge_directory(source_id)

        # 2. Update metadata for the new fork to establish provenance.
        new_metadata = self._storage.load_metadata(new_id)
        new_metadata["parent_knowledge_id"] = source_id
        new_metadata["fork_reason"] = "fork_on_write"
        new_metadata["immutable"] = False # New forks are writable by default.
        self._storage.save_metadata(new_id, new_metadata)

        return new_id

    # --- GyroExtension Interface Compliance ---
    
    def get_extension_name(self) -> str:
        return "ext_fork_manager"

    # ... and other required methods from the base class.