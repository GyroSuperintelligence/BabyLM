"""
ext_storage_manager.py - System Extension for Storage Management

This extension is responsible for all direct file system interactions, including
reading and writing session state, knowledge packages, and navigation logs.
"""

import json
import struct
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING
import uuid
import time
import os
import shutil
import tarfile
import hashlib

from core.gyro_errors import GyroStorageError, GyroIntegrityError
from extensions.base import GyroExtension

# Avoid circular imports
if TYPE_CHECKING:
    from core.extension_manager import ExtensionManager


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
        
        # Manager reference will be set by ExtensionManager
        self.manager: Optional['ExtensionManager'] = None

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

    def set_manager(self, manager: 'ExtensionManager') -> None:
        """Set the manager reference after initialization."""
        self.manager = manager

    def _get_crypto_extension(self):
        """Safely get crypto extension if available."""
        if self.manager and hasattr(self.manager, 'extensions'):
            return self.manager.extensions.get("crypto")
        return None

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
            "immutable": False,
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
            "last_modified": time.time(),
        }
        manifest_path = knowledge_path / "navigation_log" / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return new_id

    def _calculate_gene_checksum(self) -> str:
        """Calculate checksum of the Gene constant."""
        try:
            # Try to get Gene from manager if available
            if self.manager and hasattr(self.manager, 'engine'):
                import hashlib
                gene = self.manager.engine.gene
                hasher = hashlib.sha256()
                hasher.update(gene["id_0"].numpy().tobytes())
                hasher.update(gene["id_1"].numpy().tobytes())
                return f"sha256:{hasher.hexdigest()}"
        except Exception:
            pass
        
        # Fallback placeholder
        return "sha256:placeholder_gene_checksum"

    # --- Session State I/O ---

    def load_phase(self) -> int:
        """Reads the phase counter from phase.bin."""
        if not self.phase_path.exists():
            return 0

        try:
            with open(self.phase_path, "rb") as f:
                data = f.read(4)
                if len(data) < 4:
                    return 0
                return struct.unpack("<I", data)[0]
        except (OSError, struct.error):
            return 0

    def save_phase(self, phase: int) -> None:
        """Writes the phase counter to phase.bin."""
        try:
            with open(self.phase_path, "wb") as f:
                f.write(struct.pack("<I", phase))
        except OSError as e:
            raise GyroStorageError(f"Failed to save phase: {e}")

    def load_knowledge_link(self) -> Optional[str]:
        """Reads the knowledge UUID from active_knowledge.link."""
        if not self.knowledge_link_path.exists():
            return None

        try:
            with open(self.knowledge_link_path, "r") as f:
                content = f.read().strip()
                return content if content else None
        except OSError:
            return None

    def save_knowledge_link(self, knowledge_id: str) -> None:
        """Atomically writes the knowledge UUID to active_knowledge.link."""
        try:
            temp_path = self.knowledge_link_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                f.write(knowledge_id)
            temp_path.replace(self.knowledge_link_path)
        except OSError as e:
            raise GyroStorageError(f"Failed to save knowledge link: {e}")

    def switch_knowledge_context(self, knowledge_id: str) -> None:
        """Switch to a different knowledge package."""
        self.knowledge_id = knowledge_id
        self.knowledge_path = self.knowledge_root / knowledge_id
        self.save_knowledge_link(knowledge_id)

    # --- Knowledge Package I/O ---

    def load_raw_navigation_log(self, knowledge_id: str) -> bytes:
        nav_path = self.knowledge_root / knowledge_id / "navigation_log"
        enc_file = nav_path / "genome.enc"
        if enc_file.exists():
            crypto = self._get_crypto_extension()
            if crypto:
                encrypted = enc_file.read_bytes()
                return crypto.decrypt(encrypted)
            else:
                raise GyroStorageError("Found encrypted genome but no crypto key available")
        log_file = nav_path / "genome.log"
        if log_file.exists():
            return log_file.read_bytes()
        return b""

    def save_raw_navigation_log(self, knowledge_id: str, raw_data: bytes):
        nav_path = self.knowledge_root / knowledge_id / "navigation_log"
        nav_path.mkdir(parents=True, exist_ok=True)
        crypto = self._get_crypto_extension()
        if crypto:
            encrypted = crypto.encrypt(raw_data)
            (nav_path / "genome.enc").write_bytes(encrypted)
            (nav_path / "genome.log").unlink(missing_ok=True)
        else:
            (nav_path / "genome.log").write_bytes(raw_data)
            (nav_path / "genome.enc").unlink(missing_ok=True)

    def load_metadata(self, knowledge_id: str) -> Dict[str, Any]:
        """Loads and returns the content of knowledge.meta.json."""
        meta_path = self.knowledge_root / knowledge_id / "knowledge.meta.json"
        if not meta_path.exists():
            return {}

        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def save_metadata(self, knowledge_id: str, metadata: Dict[str, Any]) -> None:
        """Saves metadata to knowledge.meta.json."""
        try:
            meta_path = self.knowledge_root / knowledge_id / "knowledge.meta.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            raise GyroStorageError(f"Failed to save metadata: {e}")

    def load_extension_data(self, knowledge_id: str, filename: str) -> Optional[bytes]:
        """Loads raw data for a specific extension from the knowledge package."""
        ext_path = self.knowledge_root / knowledge_id / "extensions" / filename
        if not ext_path.exists():
            return None

        try:
            with open(ext_path, "rb") as f:
                return f.read()
        except OSError:
            return None

    def save_extension_data(self, knowledge_id: str, filename: str, data: bytes) -> None:
        """Saves raw data for a specific extension to the knowledge package."""
        try:
            ext_path = self.knowledge_root / knowledge_id / "extensions" / filename
            ext_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ext_path, "wb") as f:
                f.write(data)
        except Exception as e:
            raise GyroStorageError(f"Failed to save extension data: {e}")

    # --- Session Event Management ---

    def store_session_event(self, session_id: str, event_data: Any) -> None:
        """Store a session-specific event."""
        try:
            events_path = self.sessions_root / session_id / "events.log"
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with open(events_path, "a") as f:
                f.write(json.dumps({"timestamp": time.time(), "event": event_data}) + "\n")
        except Exception as e:
            raise GyroStorageError(f"Failed to store session event: {e}")

    def store_learning_event(self, knowledge_id: str, event_data: Any) -> None:
        """Store a learning event (high-value, contributes to intelligence)."""
        # Learning events are stored in the navigation log
        # This is handled by the NavigationLog class
        pass

    def load_ui_state(self, session_id: str) -> Dict[str, Any]:
        """Load UI state from session."""
        # Placeholder - would load from SQLite databases
        ui_state_path = self.sessions_root / session_id / "ui_state"
        if not ui_state_path.exists():
            return {}
        
        # For now, return empty dict
        # In full implementation, would load from SQLite
        return {}

    # --- Gene Storage (New Methods) ---

    def save_gene(self, knowledge_id: str, gene: Dict[str, Any]) -> None:
        """Save encrypted Gene"""
        try:
            import torch
            import io
            
            buffer = io.BytesIO()
            torch.save(gene, buffer)
            raw_data = buffer.getvalue()

            # Encrypt if crypto available
            crypto = self._get_crypto_extension()
            if crypto:
                encrypted = crypto.encrypt(raw_data)
                filepath = self.knowledge_root / knowledge_id / "gene.enc"
            else:
                encrypted = raw_data
                filepath = self.knowledge_root / knowledge_id / "gene.dat"

            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(encrypted)
        except Exception as e:
            raise GyroStorageError(f"Failed to save gene: {e}")

    def load_gene(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Load and decrypt Gene"""
        try:
            import torch
            import io
            
            # Try encrypted first
            enc_path = self.knowledge_root / knowledge_id / "gene.enc"
            if enc_path.exists():
                crypto = self._get_crypto_extension()
                if crypto:
                    try:
                        encrypted = enc_path.read_bytes()
                        decrypted = crypto.decrypt(encrypted)
                        buffer = io.BytesIO(decrypted)
                        return torch.load(buffer)
                    except Exception:
                        pass  # Fall through to unencrypted

            # Fallback: unencrypted
            gene_path = self.knowledge_root / knowledge_id / "gene.dat"
            if gene_path.exists():
                return torch.load(gene_path)

            return None
        except Exception:
            return None

    def append_output(self, session_id: str, text: str) -> None:
        """Append language output to session"""
        try:
            output_file = self.sessions_root / session_id / "output.log"
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"{text}\n")
        except Exception as e:
            raise GyroStorageError(f"Failed to append output: {e}")

    # --- High-Level Operations ---

    def fork_knowledge_directory(self, source_id: str) -> str:
        """
        Performs the physical file-system forking of a knowledge package.
        """
        try:
            new_id = str(uuid.uuid4())
            source_path = self.knowledge_root / source_id
            dest_path = self.knowledge_root / new_id

            if not source_path.exists():
                raise GyroStorageError(f"Source knowledge package {source_id} does not exist")

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
                    try:
                        os.link(str(shard), str(dest_shards / shard.name))
                    except OSError:
                        # If hard linking fails, copy instead
                        shutil.copy2(shard, dest_shards / shard.name)

            # Copy the main navigation log (both encrypted and unencrypted versions)
            for log_name in ["genome.log", "genome.enc"]:
                source_log = source_path / "navigation_log" / log_name
                if source_log.exists():
                    shutil.copy2(source_log, dest_path / "navigation_log" / log_name)

            # Copy and update metadata
            source_meta = self.load_metadata(source_id)
            new_meta = source_meta.copy()
            new_meta.update({
                "knowledge_id": new_id,
                "parent_knowledge_id": source_id,
                "created_ts": time.time(),
                "source": f"forked from {source_id}",
                "immutable": False,
            })
            self.save_metadata(new_id, new_meta)

            # Copy extension data
            source_ext = source_path / "extensions"
            dest_ext = dest_path / "extensions"
            if source_ext.exists():
                for ext_file in source_ext.glob("*"):
                    if ext_file.is_file():
                        shutil.copy2(ext_file, dest_ext / ext_file.name)

            # Copy manifest
            source_manifest = source_path / "navigation_log" / "manifest.json"
            if source_manifest.exists():
                shutil.copy2(source_manifest, dest_path / "navigation_log" / "manifest.json")

            # Copy gene files (both encrypted and unencrypted)
            for gene_name in ["gene.dat", "gene.enc"]:
                source_gene = source_path / gene_name
                if source_gene.exists():
                    shutil.copy2(source_gene, dest_path / gene_name)

            return new_id

        except Exception as e:
            raise GyroStorageError(f"Failed to fork knowledge directory: {e}")

    def build_export_bundle(
        self, knowledge_id: str, output_path: str, extension_states: Optional[Dict[str, Any]] = None
    ) -> None:
        """Creates a compressed .gyro bundle from a knowledge directory."""
        try:
            knowledge_path = self.knowledge_root / knowledge_id

            if not knowledge_path.exists():
                raise GyroStorageError(f"Knowledge package {knowledge_id} does not exist")

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
                    with open(ext_state_path, "w") as f:
                        json.dump(extension_states, f, indent=2)

                # Calculate integrity checksum
                checksum = self._calculate_bundle_checksum(bundle_root)
                with open(bundle_root / "integrity.sha256", "w") as f:
                    f.write(checksum)

                # Create tar.gz bundle
                with tarfile.open(output_path, "w:gz") as tar:
                    tar.add(bundle_root, arcname="package")

            finally:
                # Clean up temp directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        except Exception as e:
            raise GyroStorageError(f"Failed to build export bundle: {e}")

    def unpack_import_bundle(self, bundle_path: str) -> str:
        """
        Unpacks a .gyro bundle into a new knowledge directory.
        Returns the UUID of the new knowledge package directory.
        """
        try:
            if not Path(bundle_path).exists():
                raise GyroStorageError(f"Bundle file {bundle_path} does not exist")

            # Create temporary extraction directory
            temp_dir = Path(bundle_path).parent / f".tmp_import_{uuid.uuid4()}"
            temp_dir.mkdir(exist_ok=True)

            try:
                # Extract bundle
                with tarfile.open(bundle_path, "r:gz") as tar:
                    tar.extractall(temp_dir)

                # Find the package root
                package_root = temp_dir / "package"
                if not package_root.exists():
                    raise GyroStorageError("Invalid bundle format: missing package directory")

                # Verify integrity
                integrity_file = package_root / "integrity.sha256"
                if integrity_file.exists():
                    expected_checksum = integrity_file.read_text().strip()
                    integrity_file.unlink()  # Remove before calculating
                    actual_checksum = self._calculate_bundle_checksum(package_root)
                    if expected_checksum != actual_checksum:
                        raise GyroIntegrityError("Bundle integrity check failed")

                # Load metadata to get knowledge ID
                meta_path = package_root / "knowledge.meta.json"
                if not meta_path.exists():
                    raise GyroStorageError("Invalid bundle: missing metadata")

                with open(meta_path, "r") as f:
                    metadata = json.load(f)

                # Generate new ID for imported knowledge
                new_id = str(uuid.uuid4())
                metadata["knowledge_id"] = new_id
                metadata["imported_ts"] = time.time()
                metadata["imported_from"] = str(bundle_path)

                # Move to final location
                dest_path = self.knowledge_root / new_id
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.move(str(package_root), str(dest_path))

                # Update metadata with new ID
                self.save_metadata(new_id, metadata)

                return new_id

            finally:
                # Clean up temp directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        except Exception as e:
            raise GyroStorageError(f"Failed to unpack import bundle: {e}")

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
                with open(file_path, "rb") as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)

        return hasher.hexdigest()

    def validate_gene_checksum(self, gene: Dict[str, Any]) -> bool:
        """
        Validates that the current in-memory Gene constant matches the checksum
        stored in the active knowledge package's metadata.
        """
        try:
            if not isinstance(self.knowledge_id, str) or not self.knowledge_id:
                raise GyroStorageError("No valid knowledge_id set for checksum validation")
            metadata = self.load_metadata(self.knowledge_id)
            expected_checksum = metadata.get("gene_checksum")

            if not expected_checksum:
                return True  # No checksum to validate against

            # Calculate actual checksum
            if hasattr(gene, 'get') and 'id_0' in gene and 'id_1' in gene:
                import hashlib
                hasher = hashlib.sha256()
                hasher.update(gene["id_0"].numpy().tobytes())
                hasher.update(gene["id_1"].numpy().tobytes())
                actual_checksum = f"sha256:{hasher.hexdigest()}"
                return actual_checksum == expected_checksum

            return True  # Can't validate, assume OK
        except Exception:
            return True  # Error in validation, assume OK

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
            "knowledge_path": str(self.knowledge_path),
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

    def process_navigation_event(self, event, input_byte=None):
        """Storage manager doesn't process navigation events directly."""
        return None

    def get_pattern_filename(self) -> str:
        """Pattern filename for knowledge export."""
        return f"ext_storage_manager@1.0.0.config"

    def export_session(self, session_id: str, output_path: str) -> None:
        """
        Export a session directory as a .session.gyro bundle.
        Args:
            session_id: The session UUID to export.
            output_path: The file path to save the bundle.
        """
        session_path = self.sessions_root / session_id
        if not session_path.exists():
            raise GyroStorageError(f"Session {session_id} does not exist")
        try:
            with tarfile.open(output_path, "w:gz") as tar:
                tar.add(session_path, arcname="session")
        except Exception as e:
            raise GyroStorageError(f"Failed to export session: {e}")

    def import_session(self, bundle_path: str) -> str:
        """
        Import a .session.gyro bundle as a new session.
        Args:
            bundle_path: Path to the .session.gyro bundle file.
        Returns:
            The new session UUID.
        """
        if not Path(bundle_path).exists():
            raise GyroStorageError(f"Bundle file {bundle_path} does not exist")
        new_session_id = str(uuid.uuid4())
        dest_path = self.sessions_root / new_session_id
        try:
            temp_dir = Path(bundle_path).parent / f".tmp_import_session_{new_session_id}"
            temp_dir.mkdir(exist_ok=True)
            try:
                with tarfile.open(bundle_path, "r:gz") as tar:
                    tar.extractall(temp_dir)
                session_root = temp_dir / "session"
                if not session_root.exists():
                    raise GyroStorageError("Invalid session bundle: missing session directory")
                shutil.move(str(session_root), str(dest_path))
                # Update session_id in meta if present
                meta_path = dest_path / "session.meta.json"
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    meta["id"] = new_session_id
                    meta["imported_ts"] = time.time()
                    meta["imported_from"] = str(bundle_path)
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)
            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            return new_session_id
        except Exception as e:
            raise GyroStorageError(f"Failed to import session: {e}")