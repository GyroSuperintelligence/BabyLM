"""
ext_state_helper.py - State Management Utilities

This extension provides state management utilities and helpers for
coordinating state across the system.
"""

from typing import Dict, Any, Optional
import time

from .base import GyroExtension


class ext_StateHelper(GyroExtension):
    """
    State management utilities and helpers.
    FOOTPRINT: 50-100 bytes (state cache)
    MAPPING: Provides state access patterns for other extensions
    """
    
    def __init__(self, storage_manager):
        """
        Initialize state helper.
        
        Args:
            storage_manager: Handle to storage manager
        """
        self._storage = storage_manager
        self._state_cache = {
            'phase': 0,
            'nav_log_max_size': 1048576,  # 1MB default
            'last_checkpoint': time.time(),
            'checkpoint_interval': 100000,  # Operations between checkpoints
            'dirty_flags': set()
        }
        
        # Load initial state
        self._load_initial_state()
    
    def _load_initial_state(self):
        """Load initial state from storage."""
        # Load phase
        self._state_cache['phase'] = self._storage.load_phase()
        
        # Load other session state if exists
        # This would load from session metadata
        pass
    
    def load_session_state(self) -> Dict[str, Any]:
        """
        Load complete session state.
        
        Returns:
            Dictionary containing all session state
        """
        state = {
            'phase': self._storage.load_phase(),
            'knowledge_id': self._storage.knowledge_id,
            'nav_log_max_size': self._state_cache['nav_log_max_size'],
            'navigation_log': None  # Will be loaded separately
        }
        
        return state
    
    def persist_phase(self, phase: int) -> None:
        """Persist phase to storage."""
        self._state_cache['phase'] = phase
        self._storage.save_phase(phase)
        self._mark_dirty('phase')
    
    def persist_all_state(self, phase: int, knowledge_id: str) -> None:
        """
        Persist all state to storage.
        
        Args:
            phase: Current phase counter
            knowledge_id: Current knowledge package ID
        """
        # Persist phase
        self.persist_phase(phase)
        
        # Update knowledge link if changed
        if knowledge_id != self._storage.knowledge_id:
            self.update_knowledge_link(knowledge_id)
        
        # Save session metadata
        session_meta = {
            'session_id': self._storage.session_id,
            'knowledge_id': knowledge_id,
            'phase': phase,
            'last_saved': time.time(),
            'nav_log_max_size': self._state_cache['nav_log_max_size']
        }
        
        with open(self._storage.session_meta_path, 'w') as f:
            import json
            json.dump(session_meta, f, indent=2)
        
        # Clear dirty flags
        self._state_cache['dirty_flags'].clear()
        self._state_cache['last_checkpoint'] = time.time()
    
    def update_knowledge_link(self, knowledge_id: str) -> None:
        """Update the active knowledge link."""
        self._storage.save_knowledge_link(knowledge_id)
        self._mark_dirty('knowledge_link')
    
    def should_checkpoint(self, operation_count: int) -> bool:
        """
        Determine if a checkpoint should be performed.
        
        Args:
            operation_count: Number of operations since last checkpoint
            
        Returns:
            True if checkpoint is needed
        """
        # Check operation count
        if operation_count >= self._state_cache['checkpoint_interval']:
            return True
        
        # Check time elapsed
        elapsed = time.time() - self._state_cache['last_checkpoint']
        if elapsed > 300:  # 5 minutes
            return True
        
        return False
    
    def create_snapshot(self) -> Dict[str, Any]:
        """Create a complete state snapshot."""
        return {
            'timestamp': time.time(),
            'phase': self._state_cache['phase'],
            'knowledge_id': self._storage.knowledge_id,
            'session_id': self._storage.session_id,
            'cache': self._state_cache.copy()
        }
    
    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore from a state snapshot."""
        if 'phase' in snapshot:
            self.persist_phase(snapshot['phase'])
        
        if 'knowledge_id' in snapshot:
            self.update_knowledge_link(snapshot['knowledge_id'])
        
        if 'cache' in snapshot:
            self._state_cache.update(snapshot['cache'])
    
    def _mark_dirty(self, component: str) -> None:
        """Mark a component as having unsaved changes."""
        self._state_cache['dirty_flags'].add(component)
    
    def has_unsaved_changes(self) -> bool:
        """Check if there are any unsaved changes."""
        return len(self._state_cache['dirty_flags']) > 0
    
    # --- GyroExtension Interface Implementation ---
    
    def get_extension_name(self) -> str:
        return "ext_state_helper"
    
    def get_extension_version(self) -> str:
        return "1.0.0"
    
    def get_footprint_bytes(self) -> int:
        # State cache overhead
        return 100
    
    def get_learning_state(self) -> Dict[str, Any]:
        """State helper has no learning state."""
        return {}
    
    def get_session_state(self) -> Dict[str, Any]:
        """Return state cache."""
        return self._state_cache.copy()
    
    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """No learning state to restore."""
        pass
    
    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore state cache."""
        self._state_cache.update(state)