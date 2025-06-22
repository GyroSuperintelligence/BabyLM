"""
ext_storage_manager.py - System Extension for Storage Management

This extension is responsible for all direct file system interactions, including
reading and writing session state, knowledge packages, and navigation logs.

It acts as the persistence layer for the entire GyroSI system, ensuring that
all data is stored and retrieved according to the directory structure defined
in CORE-SPEC-05.
"""

import os
import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

# This extension will need access to core data structures to know how to
# serialize/deserialize them.
from ..core.alignment_nav import NavigationLog
from ..core.gyro_errors import GyroStorageError, GyroIntegrityError

# The base class for all extensions per CORE-SPEC-06.
from .base import GyroExtension


class ext_StorageManager(GyroExtension):
    """
    Handles all file I/O operations for knowledge and session persistence.
    It is the single source of truth for file paths and disk operations.
    """
    def __init__(self, session_id: Optional[str], knowledge_id: Optional[str]):
        """
        Initializes the storage manager for a specific session and knowledge context.

        Args:
            session_id: The UUID of the session to manage. If None, a new one is created.
            knowledge_id: The UUID of the knowledge to link. If None, a new one is created.
        """
        # 1. Determine and set session and knowledge IDs.
        self.session_id = session_id or str(uuid.uuid4())
        self.knowledge_id = knowledge_id # Can be None initially

        # 2. Define root paths based on CORE-SPEC-05 structure.
        self.sessions_root = Path("data/sessions")
        self.knowledge_root = Path("data/knowledge")

        # 3. Define specific paths for the current session.
        self.session_path = self.sessions_root / self.session_id
        self.knowledge_link_path = self.session_path / "active_knowledge.link"
        self.phase_path = self.session_path / "phase.bin"
        # ... etc.

        # 4. Ensure directories exist on initialization.
        self._ensure_paths()

        # 5. If knowledge_id was not provided, load it from the link file or create a new one.
        if not self.knowledge_id:
            self.knowledge_id = self.load_knowledge_link() or self._create_new_knowledge_package()
            self.save_knowledge_link(self.knowledge_id)
        
        self.knowledge_path = self.knowledge_root / self.knowledge_id

    def _ensure_paths(self):
        """Creates all necessary directories if they don't exist."""
        self.session_path.mkdir(parents=True, exist_ok=True)
        (self.session_path / "ui_state").mkdir(exist_ok=True)
        # Knowledge path is created on-demand by _create_new_knowledge_package

    def _create_new_knowledge_package(self, parent_id: Optional[str] = None) -> str:
        """Creates the directory structure for a new, empty knowledge package."""
        new_id = str(uuid.uuid4())
        knowledge_path = self.knowledge_root / new_id
        knowledge_path.mkdir(parents=True)
        (knowledge_path / "navigation_log").mkdir()
        (knowledge_path / "extensions").mkdir()
        
        # Create metadata file
        meta = { "knowledge_id": new_id, "parent_knowledge_id": parent_id, ... }
        self.save_metadata(new_id, meta)
        
        return new_id

    # --- Session State I/O ---

    def load_phase(self) -> int:
        """Reads the phase counter from phase.bin."""
        pass

    def save_phase(self, phase: int) -> None:
        """Writes the phase counter to phase.bin."""
        pass

    def load_knowledge_link(self) -> Optional[str]:
        """Reads the knowledge UUID from active_knowledge.link."""
        pass

    def save_knowledge_link(self, knowledge_id: str) -> None:
        """Atomically writes the knowledge UUID to active_knowledge.link."""
        pass

    # --- Knowledge Package I/O ---

    def load_navigation_log(self, knowledge_id: str) -> NavigationLog:
        """
        Loads the navigation log for a given knowledge package.
        This involves reading the main log file and any shards.
        """
        pass

    def save_navigation_log(self, knowledge_id: str, nav_log: NavigationLog) -> None:
        """
        Persists the state of a NavigationLog object to disk.
        This is called by the NavigationLog object itself.
        """
        pass

    def load_metadata(self, knowledge_id: str) -> Dict[str, Any]:
        """Loads and returns the content of knowledge.meta.json."""
        pass

    def save_metadata(self, knowledge_id: str, metadata: Dict[str, Any]) -> None:
        """Saves metadata to knowledge.meta.json."""
        pass

    def load_extension_data(self, knowledge_id: str, filename: str) -> Optional[bytes]:
        """Loads raw data for a specific extension from the knowledge package."""
        pass

    def save_extension_data(self, knowledge_id: str, filename: str, data: bytes) -> None:
        """Saves raw data for a specific extension to the knowledge package."""
        pass

    # --- High-Level Operations (Used by ForkManager, etc.) ---

    def fork_knowledge_directory(self, source_id: str) -> str:
        """
        Performs the physical file-system forking of a knowledge package.
        - Creates a new knowledge directory.
        - Hard-links immutable contents (like shards).
        - Copies mutable contents (like metadata).
        
        Returns:
            The UUID of the new knowledge package directory.
        """
        pass

    def build_export_bundle(self, knowledge_id: str, output_path: str) -> None:
        """
        Creates a compressed .gyro bundle from a knowledge directory.
        """
        pass

    def unpack_import_bundle(self, bundle_path: str) -> str:
        """
        Unpacks a .gyro bundle into a new knowledge directory.
        
        Returns:
            The UUID of the new knowledge package directory.
        """
        pass

    # --- Integrity and Validation ---
    
    def validate_gene_checksum(self, gene: Dict[str, torch.Tensor]) -> bool:
        """
        Validates that the current in-memory Gene constant matches the checksum
        stored in the active knowledge package's metadata.
        """
        pass

    # --- GyroExtension Interface Compliance ---
    # This is a system-critical extension, so its own state is minimal.
    
    def get_extension_name(self) -> str:
        return "ext_storage_manager"

    def get_learning_state(self) -> Dict[str, Any]:
        """Storage manager itself has no learning state."""
        return {}

    def get_session_state(self) -> Dict[str, Any]:
        """Its state is its configuration (paths), which is session-local."""
        return {
            "session_id": self.session_id,
            "knowledge_id": self.knowledge_id
        }

    # ... and other required methods from the base class.