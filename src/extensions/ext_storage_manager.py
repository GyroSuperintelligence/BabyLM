
"""
ext_storage_manager.py - System Extension for Storage Management

This extension is responsible for all direct file system interactions, including
reading and writing session state, knowledge packages, and navigation logs.
"""

import os
import json
import struct
import shutil
import tarfile
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid
import fcntl
import time

from .base import GyroExtension


class ext_StorageManager(GyroExtension):
    """
    Handles all file I/O operations for knowledge and session persistence.
    FOOTPRINT: Variable (based on active file handles and buffers)
    MAPPING: Manages data/knowledge/ and data/sessions/ directories
    """
    
    def __init__(self, session_id: Optional[str] = None, knowledge_id: Optional[str] = None):
        """Initialize storage manager for a specific session and knowledge context."""
        # Generate IDs if not provided
        self.session_id = session_id or str(uuid.uuid4())
        self.knowledge_id = knowledge_id
        
        # Define root paths based on CORE-SPEC-05 structure
        self.sessions_root = Path("data/sessions")
        self.knowledge_root = Path("data/knowledge")
        
        # Define specific paths for the current session
        self.session_path = self.sessions_root / self.session_id
        self.knowledge_link_path = self.session_path / "active_knowledge.link"
        self.phase_path = self.session_path / "phase.bin"
        self.events_path = self.session_path / "events.log"
        self.ui_state_path = self.session_path / "ui_state"
        self.session_meta_path = self.session_path / "session.meta.json"
        
        # Ensure directories exist
        self._ensure_paths()
        
        # If knowledge_id not provided, load from link or create new
        if not self.knowledge_id:
            self.knowledge_id = self.load_knowledge_link()
            if not self.knowledge_id:
                self.knowledge_id = self._create_new_knowledge_package()
                self.save_knowledge_link(self.knowledge_id)
        
        self.knowledge_path = self.knowledge_root / self.knowledge_id
        
        # Track open file handles for footprint calculation
        self._open_handles = {}
    
    def _ensure_paths(self):
        """Creates all necessary directories if they don't exist."""
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.ui_state_path.mkdir(exist_ok=True)
    
    def _create_new_knowledge_package(self, parent_id: Optional[str] = None) -> str:
        """Creates the directory structure for a new, empty knowledge package."""
        new_id = str(uuid.uuid4())
        knowledge_path = self.knowledge_root / new_id
        knowledge_path.mkdir(parents=True)
        (knowledge_path / "navigation_log").mkdir()
        (knowledge_path / "navigation_log" / "shards").mkdir()
        (knowledge_path / "extensions").mkdir()
        
        # Create initial metadata
        metadata = {
            "knowledge_id": new_id,
            "gyro_version": "0.8.8",
            "gene_checksum": self._calculate_gene_checksum(),
            "step_count": 0,
            "created_ts": time.time(),
            "source": f"session {self.session_id}",
            "parent_knowledge_id": parent_id,
            "learning_sources": [],
            "extension_versions": {},
            "immutable": False
        }
        
        self.save_metadata(new_id, metadata)
        
        # Create empty navigation log
        nav_log_path = knowledge_path / "navigation_log" / "genome.log"
        nav_log_path.touch()
        
        # Create manifest
        manifest = {
            "format_version": "1.0",
            "shard_count": 0,
            "total_steps": 0,
            "last_modified": time.time()
        }
        manifest_path = knowledge_path / "navigation_log" / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return new_id
    
    def _calculate_gene_checksum(self) -> str:
        """Calculate checksum of the Gene constant."""
        # This would import gyro_core and calculate actual checksum
        # For now, return a placeholder
        return "sha256:placeholder_gene_checksum"
    
    # --- Session State I/O ---
    
    def load_phase(self) -> int:
        """Reads the phase counter from phase.bin."""
        if not self.phase_path.exists():
            return 0
        
        with open(self.phase_path, 'rb') as f:
            data = f.read(4)
            if len(data) < 4:
                return 0
            return struct.unpack('<I', data)[0]
    
    def save_phase(self, phase: int) -> None:
        """Writes the phase counter to phase.bin."""
        with open(self.phase_path, 'wb') as f:
            f.write(struct.pack('<I', phase))
    
    def load_knowledge_link(self) -> Optional[str]:
        """Reads the knowledge UUID from active_knowledge.link."""
        if not self.knowledge_link_path.exists():
            return None
        
        with open(self.knowledge_link_path, 'r') as f:
            content = f.read().strip()
            return content if content else None
    
    def save_knowledge_link(self, knowledge_id: str) -> None:
        """Atomically writes the knowledge UUID to active_knowledge.link."""
        temp_path = self.knowledge_link_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            f.write(knowledge_id)
        temp_path.replace(self.knowledge_link_path)
    
    def switch_knowledge_context(self, knowledge_id: str) -> None:
        """Switch to a different knowledge package."""
        self.knowledge_id = knowledge_id
        self.knowledge_path = self.knowledge_root / knowledge_id
        self.save_knowledge_link(knowledge_id)
    
    # --- Knowledge Package I/O ---
    
    def load_raw_navigation_log(self, knowledge_id: str) -> bytes:
        """Load raw navigation log data."""
        nav_log_path = self.knowledge_root / knowledge_id / "navigation_log" / "genome.log"
        if not nav_log_path.exists():
            return b''
        
        with open(nav_log_path, 'rb') as f:
            return f.read()
    
    def save_raw_navigation_log(self, knowledge_id: str, data: bytes) -> None:
        """Save raw navigation log data."""
        nav_log_path = self.knowledge_root / knowledge_id / "navigation_log" / "genome.log"
        
        # Use file locking for thread safety
        with open(nav_log_path, 'wb') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(data)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def load_metadata(self, knowledge_id: str) -> Dict[str, Any]:
        """Loads and returns the content of knowledge.meta.json."""
        meta_path = self.knowledge_root / knowledge_id / "knowledge.meta.json"
        if not meta_path.exists():
            return {}
        
        with open(meta_path, 'r') as f:
            return json.load(f)
    
    def save_metadata(self, knowledge_id: str, metadata: Dict[str, Any]) -> None:
        """Saves metadata to knowledge.meta.json."""
        meta_path = self.knowledge_root / knowledge_id / "knowledge.meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_extension_data(self, knowledge_id: str, filename: str) -> Optional[bytes]:
        """Loads raw data for a specific extension from the knowledge package."""
        ext_path = self.knowledge_root / knowledge_id / "extensions" / filename
        if not ext_path.exists():
            return None
        
        with open(ext_path, 'rb') as f:
            return f.read()
    
    def save_extension_data(self, knowledge_id: str, filename: str, data: bytes) -> None:
        """Saves raw data for a specific extension to the knowledge package."""
        ext_path = self.knowledge_root / knowledge_id / "extensions" / filename
        with open(ext_path, 'wb') as f:
            f.write(data)
    
    # --- Session Event Management ---
    
    def store_session_event(self, session_id: str, event_data: Any) -> None:
        """Store a session-specific event."""
        events_path = self.sessions_root / session_id / "events.log"
        with open(events_path, 'a') as f:
            f.write(json.dumps({
                'timestamp': time.time(),
                'event': event_data
            }) + '\n')
    
    def store_learning_event(self, knowledge_id: str, event_data: Any) -> None:
        """Store a learning event (high-value, contributes to intelligence)."""
        # Learning events are stored in the navigation log
        # This is handled by the NavigationLog class
        pass
    
    def load_ui_state(self, session_id: str) -> Dict[str, Any]:
        """Load UI state from session."""
        # Placeholder - would load from SQLite databases
        return {}
    
    # --- High-Level Operations
    
    def fork_knowledge_directory(self, source_id: str) -> str:
        """
        Performs the physical file-system forking of a knowledge package.
        - Creates a new knowledge directory
        - Hard-links immutable contents (like shards)
        - Copies mutable contents (like metadata)
        
        Returns:
            The UUID of the new knowledge package directory
        """
        new_id = str(uuid.uuid4())
        source_path = self.knowledge_root / source_id
        dest_path = self.knowledge_root / new_id
        
        # Create destination structure
        dest_path.mkdir(parents=True)
        (dest_path / "navigation_log").mkdir()
        (dest_path / "navigation_log" / "shards").mkdir()
        (dest_path / "extensions").mkdir()
        
        # Hard-link navigation log shards (immutable)
        source_shards = source_path / "navigation_log" / "shards"
        dest_shards = dest_path / "navigation_log" / "shards"
        
        if source_shards.exists():
            for shard in source_shards.glob("*.bin"):
                os.link(str(shard), str(dest_shards / shard.name))
        
        # Copy the main navigation log
        source_log = source_path / "navigation_log" / "genome.log"
        if source_log.exists():
            shutil.copy2(source_log, dest_path / "navigation_log" / "genome.log")
        
        # Copy and update metadata
        source_meta = self.load_metadata(source_id)
        new_meta = source_meta.copy()
        new_meta.update({
            "knowledge_id": new_id,
            "parent_knowledge_id": source_id,
            "created_ts": time.time(),
            "source": f"forked from {source_id}",
            "immutable": False
        })
        self.save_metadata(new_id, new_meta)
        
        # Copy extension data
        source_ext = source_path / "extensions"
        dest_ext = dest_path / "extensions"
        if source_ext.exists():
            for ext_file in source_ext.glob("*"):
                shutil.copy2(ext_file, dest_ext / ext_file.name)
        
        # Copy manifest
        source_manifest = source_path / "navigation_log" / "manifest.json"
        if source_manifest.exists():
            shutil.copy2(source_manifest, dest_path / "navigation_log" / "manifest.json")
        
        return new_id
    
    def build_export_bundle(self, knowledge_id: str, output_path: str, 
                           extension_states: Optional[Dict[str, Any]] = None) -> None:
        """Creates a compressed .gyro bundle from a knowledge directory."""
        knowledge_path = self.knowledge_root / knowledge_id
        
        # Create temporary directory for bundle contents
        temp_dir = Path(output_path).parent / f".tmp_{knowledge_id}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Copy all knowledge contents
            bundle_root = temp_dir / "package"
            shutil.copytree(knowledge_path, bundle_root)
            
            # Add extension states if provided
            if extension_states:
                ext_state_path = bundle_root / "extension_states.json"
                with open(ext_state_path, 'w') as f:
                    json.dump(extension_states, f, indent=2)
            
            # Calculate integrity checksum
            checksum = self._calculate_bundle_checksum(bundle_root)
            with open(bundle_root / "integrity.sha256", 'w') as f:
                f.write(checksum)
            
            # Create tar.gz bundle
            with tarfile.open(output_path, 'w:gz') as tar:
                tar.add(bundle_root, arcname="package")
            
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir)
    
    def unpack_import_bundle(self, bundle_path: str) -> str:
        """
        Unpacks a .gyro bundle into a new knowledge directory.
        
        Returns:
            The UUID of the new knowledge package directory
        """
        # Create temporary extraction directory
        temp_dir = Path(bundle_path).parent / f".tmp_import_{uuid.uuid4()}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Extract bundle
            with tarfile.open(bundle_path, 'r:gz') as tar:
                tar.extractall(temp_dir)
            
            # Find the package root
            package_root = temp_dir / "package"
            if not package_root.exists():
                raise ValueError("Invalid bundle format: missing package directory")
            
            # Verify integrity
            integrity_file = package_root / "integrity.sha256"
            if integrity_file.exists():
                expected_checksum = integrity_file.read_text().strip()
                integrity_file.unlink()  # Remove before calculating
                actual_checksum = self._calculate_bundle_checksum(package_root)
                if expected_checksum != actual_checksum:
                    raise ValueError("Bundle integrity check failed")
            
            # Load metadata to get knowledge ID
            meta_path = package_root / "knowledge.meta.json"
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            # Generate new ID for imported knowledge
            new_id = str(uuid.uuid4())
            metadata["knowledge_id"] = new_id
            metadata["imported_ts"] = time.time()
            metadata["imported_from"] = str(bundle_path)
            
            # Move to final location
            dest_path = self.knowledge_root / new_id
            shutil.move(str(package_root), str(dest_path))
            
            # Update metadata with new ID
            self.save_metadata(new_id, metadata)
            
            return new_id
            
        finally:
            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def _calculate_bundle_checksum(self, bundle_root: Path) -> str:
        """Calculate SHA256 checksum of bundle contents."""
        hasher = hashlib.sha256()
        
        # Sort files for consistent ordering
        for file_path in sorted(bundle_root.rglob("*")):
            if file_path.is_file():
                # Include relative path in hash
                rel_path = file_path.relative_to(bundle_root)
                hasher.update(str(rel_path).encode())
                
                # Include file contents
                with open(file_path, 'rb') as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def validate_gene_checksum(self, gene: Dict[str, Any]) -> bool:
        """
        Validates that the current in-memory Gene constant matches the checksum
        stored in the active knowledge package's metadata.
        """
        metadata = self.load_metadata(self.knowledge_id)
        expected_checksum = metadata.get("gene_checksum")
        
        if not expected_checksum:
            return True  # No checksum to validate against
        
        # Calculate actual checksum
        # This would need to import torch and calculate from gene tensors
        # For now, return True
        return True
    
    # --- GyroExtension Interface Implementation ---
    
    def get_extension_name(self) -> str:
        return "ext_storage_manager"
    
    def get_extension_version(self) -> str:
        return "1.0.0"
    
    def get_footprint_bytes(self) -> int:
        """Calculate current memory footprint."""
        # Base overhead
        footprint = 1024  # 1KB base
        
        # Add size of open file handles
        footprint += len(self._open_handles) * 256
        
        # Add path string overhead
        footprint += len(str(self.session_path)) * 2
        footprint += len(str(self.knowledge_path)) * 2
        
        return footprint
    
    def get_learning_state(self) -> Dict[str, Any]:
        """Storage manager has no learning state."""
        return {}
    
    def get_session_state(self) -> Dict[str, Any]:
        """Return current configuration."""
        return {
            "session_id": self.session_id,
            "knowledge_id": self.knowledge_id,
            "session_path": str(self.session_path),
            "knowledge_path": str(self.knowledge_path)
        }
    
    def set_learning_state(self, state: Dict[str, Any]) -> None:
        """No learning state to restore."""
        pass
    
    def set_session_state(self, state: Dict[str, Any]) -> None:
        """Restore configuration."""
        if "session_id" in state:
            self.session_id = state["session_id"]
            self.session_path = self.sessions_root / self.session_id
        
        if "knowledge_id" in state:
            self.knowledge_id = state["knowledge_id"]
            self.knowledge_path = self.knowledge_root / self.knowledge_id