"""
ext_fork_manager.py - System Extension for Knowledge Forking

This extension implements the fork-on-write immutability contract for
knowledge packages.
"""

from typing import Optional, Dict, Any

from .base import GyroExtension


class ext_ForkManager(GyroExtension):
    """
    Manages the fork-on-write lifecycle of knowledge packages.
    FOOTPRINT: 20-30 bytes (fork state)
    MAPPING: Manages isolated forks for knowledge editing
    """
    
    def __init__(self, storage_manager):
        """
        Initialize fork manager.
        
        Args:
            storage_manager: Handle to the storage manager for file operations
        """
        self._storage = storage_manager
        self._fork_state = {
            'is_immutable': False,
            'fork_count': 0,
            'last_fork_id': None
        }
        
        # Check if current knowledge is immutable
        self._update_immutability_status()
    
    def _update_immutability_status(self):
        """Update immutability status from knowledge metadata."""
        metadata = self._storage.load_metadata(self._storage.knowledge_id)
        self._fork_state['is_immutable'] = metadata.get("immutable", False)
    
    def is_current_knowledge_immutable(self) -> bool:
        """Check if the currently linked knowledge package is immutable."""
        return self._fork_state['is_immutable']
    
    def ensure_writable(self, current_nav_log) -> Any:
        """
        Ensures we have a writable navigation log.
        
        If current knowledge is immutable, triggers a fork and returns
        the new NavigationLog object. Otherwise returns the original.
        
        Args:
            current_nav_log: The current NavigationLog object
            
        Returns:
            A writable NavigationLog object
        """
        if not self.is_current_knowledge_immutable():
            return current_nav_log
        
        # Fork is required
        new_knowledge_id = self.fork()
        
        # Update storage context
        self._storage.switch_knowledge_context(new_knowledge_id)
        
        # Create new navigation log for the fork
        # This would need to import NavigationLog
        # For now, we'll assume the caller handles this
        current_nav_log.knowledge_id = new_knowledge_id
        
        # Update immutability status
        self._fork_state['is_immutable'] = False
        
        return current_nav_log
    
    def fork(self) -> str:
        """
        Execute the forking process.
        
        Returns:
            UUID of the newly created knowledge package
        """
        source_id = self._storage.knowledge_id
        
        # Delegate physical forking to storage manager
        new_id = self._storage.fork_knowledge_directory(source_id)
        
        # Update fork tracking
        self._fork_state['fork_count'] += 1
        self._fork_state['last_fork_id'] = new_id
        
        return new_id
    
    def mark_immutable(self, knowledge_id: Optional[str] = None) -> None:
        """Mark a knowledge package as immutable."""
        if knowledge_id is None:
            knowledge_id = self._storage.knowledge_id
        
        metadata = self._storage.load_metadata(knowledge_id)
        metadata["immutable"] = True
        self._storage.save_metadata(knowledge_id, metadata)
        
        if knowledge_id == self._storage.knowledge_id:
            self._fork_state['is_immutable'] = True
    
    # --- GyroExtension Interface Implementation ---
    
    def get_extension_name(self) -> str:
        return "ext_fork_manager"
    
    def get_extension_version(self) -> str:
        return "1.0.0"
    
    def get_footprint_bytes(self) -> int:
        # Fork state dictionary overhead
        return 30
    
    def get_learning_state(self) -> Dict[str, Any]:
        """Fork manager has no learning state."""
        return {}
    
    def get_session_state(self) -> Dict[str, Any]:
        """Return fork tracking state."""
        return self._fork_state.copy()
    
    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """No learning state to restore."""
        pass
    
    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore fork tracking state."""
        self._fork_state.update(state)