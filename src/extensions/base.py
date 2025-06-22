"""
base.py - Base class for all GyroSI extensions.

Defines the interface that all extensions must implement to participate
in the GyroSI ecosystem.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class GyroExtension(ABC):
    """
    Base contract for extensions operating through invariant substrate.
    
    All extensions must:
    1. Have a canonical name starting with 'ext_'
    2. Track their memory footprint
    3. Separate learning state (exportable) from session state (non-exportable)
    4. Process navigation events without modifying core state
    """
    
    @abstractmethod
    def get_extension_name(self) -> str:
        """
        Returns the canonical name of this extension.
        Must start with 'ext_' per CORE-SPEC-04.
        """
        pass
    
    @abstractmethod
    def get_extension_version(self) -> str:
        """
        Returns the semantic version of this extension.
        Format: 'major.minor.patch' (e.g., '1.0.0')
        """
        pass
    
    @abstractmethod
    def get_footprint_bytes(self) -> int:
        """
        Returns the current memory footprint in bytes.
        This must be accurate for system validation.
        """
        pass
    
    @abstractmethod
    def get_learning_state(self) -> Dict[str, Any]:
        """
        Returns state that should be exported with knowledge packages.
        This is the accumulated intelligence/patterns learned.
        """
        pass
    
    @abstractmethod
    def get_session_state(self) -> Dict[str, Any]:
        """
        Returns session-local state that is NOT exported.
        This includes caches, UI preferences, temporary data.
        """
        pass
    
    @abstractmethod
    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """
        Restores learning state from a knowledge package.
        Called when importing or loading knowledge.
        """
        pass
    
    @abstractmethod
    def set_session_state(self, state: Dict[str, Any]) -> None:
        """
        Restores session state.
        Called when resuming a session.
        """
        pass
    
    def ext_on_navigation_event(self, nav_event: int, input_byte: Optional[int] = None) -> None:
        """
        Process a navigation event. This is called by the ExtensionManager
        whenever a navigation event occurs.
        
        Args:
            nav_event: The packed navigation byte (two 4-bit codes)
            input_byte: The original input byte that caused this navigation
        """
        # Default implementation does nothing
        # Extensions override this if they need to process events
        pass
    
    def validate_footprint(self) -> bool:
        """
        Validates that the extension's actual memory usage matches declared footprint.
        Can be overridden for custom validation logic.
        """
        # Default implementation always passes
        # Extensions with strict footprint requirements should override
        return True
    
    def get_pattern_filename(self) -> str:
        """
        Returns the filename pattern for storing this extension's learning state.
        Format: ext_<name>@<version>.<type>
        """
        name = self.get_extension_name()
        version = self.get_extension_version()
        # Remove 'ext_' prefix for cleaner filenames
        clean_name = name[4:] if name.startswith('ext_') else name
        return f"ext_{clean_name}@{version}.patterns"
    
    def shutdown(self) -> None:
        """
        Called when the extension is being shut down.
        Override to perform cleanup tasks.
        """
        pass
    
    def persist_state(self) -> None:
        """
        Called to persist current state to storage.
        Override if the extension needs custom persistence logic.
        """
        pass
    
    def load_state(self) -> None:
        """
        Called to load saved state from storage.
        Override if the extension needs custom loading logic.
        """
        pass