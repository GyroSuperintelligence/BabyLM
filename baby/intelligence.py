"""
intelligence.py - S4 Orchestration for GyroSI Baby LM

This module implements the Intelligence Engine for orchestration, file I/O,
and thread lifecycle management, representing the Intelligence (S4) layer
of the Common Governance Model.
"""

import os
import uuid
import numpy as np
import random
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, cast
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from baby.governance import gene_stateless

from baby.inference import InferenceEngine
from baby.information import (
    InformationEngine,
    ensure_agent_uuid,
    create_thread,
    save_thread,
    load_thread,
    store_thread_key,
    store_gene_keys,
    parent,
    children,
    list_formats,
    load_format,
    store_format,
    load_pattern_distances,
    get_memory_preferences,
    shard_path,
    PatternIndex,
    load_thread_key,
    json_loads,
    json_dumps,
)
from baby.types import PatternMetadata, FormatMetadata, GeneKeysMetadata

import base64
import json

__all__ = [
    "IntelligenceEngine",
    "weighted_choice",
    "initialize_intelligence_engine",
]


class IntelligenceEngine:
    """
    Intelligence Engine for orchestration, file I/O, and thread lifecycle

    Manages the overall system state, thread lifecycles, and file operations.
    This layer contains the "conscious" decision-making processes.
    """

    def __init__(
        self,
        agent_uuid: Optional[str],
        agent_secret: Optional[str],
        inference_engine: InferenceEngine,
        information_engine: InformationEngine,
        format_uuid: Optional[str] = None,
        formats: Optional[FormatMetadata] = None,
    ):
        """
        Initialize the Intelligence Engine

        Args:
            agent_uuid: UUID of the current agent
            agent_secret: Persistent secret for encryption
            inference_engine: The InferenceEngine instance
            information_engine: The InformationEngine instance
            format_uuid: UUID of the active format
            formats: Optional format metadata dictionary
        """
        # Allow None for public mode
        self.inference_engine = inference_engine
        self.information_engine = information_engine
        self.agent_uuid = agent_uuid
        self.agent_secret = agent_secret
        self.format_uuid = format_uuid or self._get_or_create_format_uuid()
        self.thread_uuid = None
        self.thread_file_key = None
        self.current_thread_keys = []
        self.current_thread_size = 0
        self.active_thread_content = bytearray()
        self.parent_thread_uuid = None
        self.child_thread_uuids = []
        self.M: FormatMetadata = formats if formats else self._load_or_init_formats()
        self.memory_prefs = get_memory_preferences()
        self._validate_format_compatibility()
        self.pattern_index = (
            PatternIndex(self.agent_uuid, self.agent_secret)
            if (self.agent_uuid is not None and self.agent_secret is not None)
            else None
        )
        self.pattern_distances = None
        if self.M and "pattern_distances" in self.M and "path" in self.M["pattern_distances"]:
            self.pattern_distances = load_pattern_distances(self.format_uuid)
        # File handle for public thread NDJSON streaming
        self._active_public_thread_handle = None

    def _get_or_create_format_uuid(self) -> str:
        """
        Get an existing format UUID or create a new one.

        Returns:
            str: Format UUID
        """
        formats = list_formats()
        if formats:
            return formats[0]

        # Create a default format
        format_data = {
            "format_uuid": str(uuid.uuid4()),
            "format_name": "default_format",
            "format_version": "1.0.0",
            "stability": "experimental",
            "compatibility": {
                "min_format_version": "1.0.0",
                "max_format_version": "1.0.0",
                "depends_on": [],
                "conflicts_with": [],
            },
            "metadata": {
                "author": f"agent_{self.agent_uuid[:8]}" if self.agent_uuid else "public",
                "description": "Default format initialized automatically",
                "tags": ["default", "auto_generated"],
                "created_at": datetime.datetime.now().isoformat(),
                "last_updated": datetime.datetime.now().isoformat(),
                "usage_count": 0,
                "validation_status": "unverified",
            },
            "cgm_policies": {
                "governance": {"operation": "L0", "bits": [0, 7], "policy": "traceability"},
                "information": {"operation": "LI", "bits": [1, 6], "policy": "variety"},
                "inference": {"operation": "FG", "bits": [2, 5], "policy": "accountability"},
                "intelligence": {"operation": "BG", "bits": [3, 4], "policy": "integrity"},
            },
            "patterns": [],
        }

        # Initialize pattern metadata
        patterns: list[PatternMetadata] = []
        for i in range(256):
            pattern: PatternMetadata = {
                "index": i,
                "character": None,
                "description": None,
                "type": None,
                "count": 0,
                "first_cycle": None,
                "last_cycle": None,
                "gyration_feature": self.inference_engine.gyration_features[i],
                "confidence": 0.0,
            }
            patterns.append(pattern)

        format_data["patterns"] = patterns

        return store_format(cast(FormatMetadata, format_data))

    def _validate_format_compatibility(self) -> None:
        """
        Validate the loaded format for compatibility with current format version

        Raises:
            ValueError: If the format is incompatible with the current format version
        """
        # Check format exists
        if not self.M:
            return  # New format will be created

        # Get current format version from preferences (if any)
        current_format_version = self.memory_prefs["format_config"].get("default_format_version", "1.0.0")

        # Check format compatibility
        if "compatibility" in self.M:
            min_version = self.M["compatibility"].get("min_format_version")
            max_version = self.M["compatibility"].get("max_format_version")

            # Simple version comparison (this could be enhanced with proper semver comparison)
            if min_version and current_format_version < min_version:
                raise ValueError(
                    f"Format requires minimum format version {min_version}, but current is {current_format_version}"
                )

            if max_version and current_format_version > max_version:
                raise ValueError(
                    f"Format requires maximum format version {max_version}, but current is {current_format_version}"
                )

        # Validate policy mapping
        if "cgm_policies" in self.M:
            policies = self.M["cgm_policies"]
            required_policies = ["governance", "information", "inference", "intelligence"]

            for policy in required_policies:
                if policy not in policies:
                    raise ValueError(f"Required CGM policy '{policy}' missing from format")

    def _load_or_init_formats(self) -> FormatMetadata:
        """
        Load format metadata from file or initialize if not present

        Returns:
            Dict: Format metadata
        """
        format_data = load_format(self.format_uuid)

        if not format_data:
            # Initialize new format metadata
            format_data = self._initialize_format_metadata()
            store_format(format_data)

        return format_data

    def _initialize_format_metadata(self) -> FormatMetadata:
        """
        Initialize new format metadata

        Returns:
            Dict: Initialized format metadata
        """
        # Create format metadata structure
        M: FormatMetadata = {
            "format_uuid": self.format_uuid,
            "format_name": "default_format",
            "format_version": "1.0.0",
            "stability": "experimental",
            "compatibility": {
                "min_format_version": "1.0.0",
                "max_format_version": "1.0.0",
                "depends_on": [],
                "conflicts_with": [],
            },
            "metadata": {
                "author": f"agent_{self.agent_uuid[:8]}" if self.agent_uuid else "public",
                "description": "Default format initialized automatically",
                "tags": ["default", "auto_generated"],
                "created_at": datetime.datetime.now().isoformat(),
                "last_updated": datetime.datetime.now().isoformat(),
                "usage_count": 0,
                "validation_status": "unverified",
            },
            "cgm_policies": {
                "governance": {"operation": "L0", "bits": [0, 7], "policy": "traceability"},
                "information": {"operation": "LI", "bits": [1, 6], "policy": "variety"},
                "inference": {"operation": "FG", "bits": [2, 5], "policy": "accountability"},
                "intelligence": {"operation": "BG", "bits": [3, 4], "policy": "integrity"},
            },
            "patterns": [],
        }

        # Initialize pattern metadata
        patterns: list[PatternMetadata] = []
        for i in range(256):
            pattern: PatternMetadata = {
                "index": i,
                "character": None,
                "description": None,
                "type": None,
                "count": 0,
                "first_cycle": None,
                "last_cycle": None,
                "gyration_feature": self.inference_engine.gyration_features[i],
                "confidence": 0.0,
            }
            patterns.append(pattern)

        M["patterns"] = patterns

        return M

    def start_new_thread(self, privacy: str = "private") -> str:
        """
        Start a new thread, setting its parent to the current thread if one exists.

        Returns:
            str: UUID of the newly created active thread.
        """
        # This method's job is to unconditionally start a new thread.
        # The decision to do so is made by the calling code (e.g., _append_to_thread).

        # Create a new thread, linking it to the previous one.
        parent_uuid = self.thread_uuid if self.thread_uuid else None
        new_thread_uuid = create_thread(
            privacy=privacy,
            parent_uuid=parent_uuid,
            format_uuid=self.format_uuid,
        )

        # --- NEW: Update parent metadata to add child ---
        if parent_uuid is not None and self.agent_uuid is not None:
            private_dir = Path("memories/private/agents")
            agent_shard = shard_path(private_dir, self.agent_uuid)
            agent_dir = agent_shard / f"agent-{self.agent_uuid}"
            threads_dir = agent_dir / "threads"
            parent_shard = shard_path(threads_dir, parent_uuid)
            parent_meta_path = parent_shard / f"thread-{parent_uuid}.json"
            if parent_meta_path.exists():
                with open(parent_meta_path, "r") as f:
                    parent_meta = json_loads(f.read())
                if "children" not in parent_meta or not isinstance(parent_meta["children"], list):
                    parent_meta["children"] = []
                if new_thread_uuid not in [c["uuid"] for c in parent_meta["children"]]:
                    parent_meta["children"].append({"uuid": new_thread_uuid, "name": None})
                with open(parent_meta_path, "w") as f:
                    f.write(json_dumps(parent_meta))

        # Reset the engine's state for the new thread.
        self.thread_uuid = new_thread_uuid
        self.thread_file_key = self._derive_file_key(self.inference_engine.T.copy(), self.agent_uuid, new_thread_uuid)
        self.current_thread_keys = []
        self.current_thread_size = 0
        self.active_thread_content = bytearray()
        self.parent_thread_uuid = parent_uuid
        self.child_thread_uuids = []

        # Close any existing handle before starting a new thread
        if self._active_public_thread_handle:
            self._active_public_thread_handle.close()
            self._active_public_thread_handle = None

        # Store the new thread's key for future use.
        if (
            privacy == "private"
            and self.agent_uuid is not None
            and self.agent_secret is not None
            and self.thread_file_key is not None
        ):
            store_thread_key(self.agent_uuid, new_thread_uuid, self.thread_file_key, self.agent_secret)

        if privacy == "public":
            threads_root = Path("memories/public/threads")
            thread_shard = shard_path(threads_root, str(new_thread_uuid))
            thread_path = thread_shard / f"thread-{new_thread_uuid}.ndjson"
            thread_shard.mkdir(parents=True, exist_ok=True)
            self._active_public_thread_handle = open(thread_path, "a", encoding="utf-8")
            meta_path = thread_shard / f"thread-{new_thread_uuid}.json"
            now = datetime.datetime.now().isoformat()
            meta = {
                "thread_uuid": new_thread_uuid,
                "thread_name": None,
                "agent_uuid": None,
                "parent_uuid": None,
                "parent_name": None,
                "children": [],
                "format_uuid": self.format_uuid,
                "curriculum": None,
                "tags": None,
                "created_at": now,
                "last_updated": now,
                "size_bytes": 0,
                "privacy": "public",
            }
            with open(meta_path, "w") as f:
                f.write(json_dumps(meta))

        return new_thread_uuid

    def process_input_stream(self, input_stream: bytes, privacy: str = "private") -> Tuple[bytes, bytes]:
        """
        Process an external input stream and append to current thread or create new one
        """
        # 1. First, ensure the thread state is ready (creates new thread if needed).
        self._append_to_thread(
            {"type": "input", "data": base64.b64encode(input_stream).decode("utf-8")}, privacy=privacy
        )

        # 2. Now, process the stream and log gene keys to the *active* thread.
        def callback(source_byte, key_index, resonance, event_type):
            self.update_learning_state(source_byte, key_index, resonance, event_type, privacy=privacy)

        intermediate_ciphertext, dynamic_keystream = self.information_engine.process_stream(
            self.inference_engine, callback, input_stream
        )
        return input_stream, intermediate_ciphertext

    def _append_to_thread(self, event: dict, privacy: str = "private") -> None:
        # 1. Ensure a thread exists (this will call start_new_thread if needed)
        if not self.thread_uuid:
            self.start_new_thread(privacy=privacy)
        # 2. Serialize event
        json_line = json_dumps(event).encode("utf-8") + b"\n"
        # 3. Check for thread rotation BEFORE writing
        max_size_bytes = self.memory_prefs["storage_config"]["max_thread_size_mb"] * 1024 * 1024
        if self.current_thread_size > 0 and (self.current_thread_size + len(json_line) > max_size_bytes):
            self.finalize_and_save_thread(privacy=privacy)  # This will be refactored in a later step
            self.start_new_thread(privacy=privacy)
        # 4. Write or Buffer based on privacy
        if privacy == "public":
            if self.thread_uuid is None:
                raise ValueError("thread_uuid must not be None for public thread operations.")
            thread_shard = shard_path(Path("memories/public/threads"), self.thread_uuid)
            thread_path = thread_shard / f"thread-{self.thread_uuid}.ndjson"
            if self._active_public_thread_handle is None:
                thread_shard.mkdir(parents=True, exist_ok=True)
                self._active_public_thread_handle = open(thread_path, "a", encoding="utf-8")
            print(f"Writing to NDJSON: {thread_path}")
            self._active_public_thread_handle.write(json_line.decode("utf-8"))
            self._active_public_thread_handle.flush()
        else:
            # For private threads, continue using the in-memory buffer
            self.active_thread_content.extend(json_line)
        # 5. Reliably update size
        self.current_thread_size += len(json_line)
        # 6. DO NOT save the entire thread here anymore for public threads.
        # self._save_current_thread(privacy=privacy)  # <-- REMOVED for public threads

    def finalize_and_save_thread(self, privacy: str = "private") -> None:
        """
        Finalizes the active thread. For public threads, this closes the file handle.
        For private threads, this encrypts and writes the in-memory buffer to disk.
        This method should be called when a thread is full or a session ends.
        """
        if not self.thread_uuid:
            return

        # --- Finalize Public Thread ---
        if privacy == "public":
            # Always close the file handle if open
            if self._active_public_thread_handle:
                self._active_public_thread_handle.close()
                self._active_public_thread_handle = None

            # --- FIX: Calculate the shard path correctly (only once) ---
            from baby.information import shard_path, json_loads, json_dumps
            from pathlib import Path
            import os

            threads_root = Path("memories/public/threads")
            thread_shard = shard_path(threads_root, self.thread_uuid)
            meta_path = thread_shard / f"thread-{self.thread_uuid}.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)

            now = datetime.datetime.now().isoformat()
            meta = None
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json_loads(f.read())
            if not meta:
                meta = {
                    "thread_uuid": self.thread_uuid,
                    "thread_name": None,
                    "agent_uuid": None,
                    "parent_uuid": self.parent_thread_uuid,
                    "parent_name": None,
                    "children": [],
                    "format_uuid": self.format_uuid,
                    "curriculum": None,
                    "tags": None,
                    "created_at": now,
                    "last_updated": now,
                    "size_bytes": 0,
                    "privacy": "public",
                }
            meta["last_updated"] = now
            meta["privacy"] = "public"
            # Update size_bytes (accurate at finalization)
            thread_path = thread_shard / f"thread-{self.thread_uuid}.ndjson"
            if os.path.exists(thread_path):
                meta["size_bytes"] = os.path.getsize(thread_path)
            with open(meta_path, "w") as f:
                f.write(json_dumps(meta))
        # --- Finalize Private Thread ---
        else:  # privacy == "private"
            if not self.active_thread_content:
                return  # Nothing to save
            final_data_to_save = bytes(self.active_thread_content)
            if self.thread_file_key is None:
                raise RuntimeError("Thread file key is not set for a private thread.")
            # Use thread_file_key directly as the AES-256-GCM key
            aes_key = self.thread_file_key
            encrypted_blob = self._encrypt_data(final_data_to_save, aes_key)
            save_thread(self.thread_uuid, encrypted_blob, privacy)

            # Update thread metadata with actual file size
            from baby.information import shard_path, json_loads, json_dumps
            from pathlib import Path
            import os

            private_dir = Path("memories/private/agents")
            if self.agent_uuid is None:
                raise ValueError("agent_uuid must not be None when finalizing a private thread.")
            agent_shard = shard_path(private_dir, self.agent_uuid)
            agent_dir = agent_shard / f"agent-{self.agent_uuid}"
            threads_dir = agent_dir / "threads"
            thread_shard = shard_path(threads_dir, self.thread_uuid)
            meta_path = thread_shard / f"thread-{self.thread_uuid}.json"

            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json_loads(f.read())
                meta["last_updated"] = datetime.datetime.now().isoformat()
                # Update size_bytes (accurate at finalization)
                thread_path = thread_shard / f"thread-{self.thread_uuid}.enc"
                if os.path.exists(thread_path):
                    meta["size_bytes"] = os.path.getsize(thread_path)
                with open(meta_path, "w") as f:
                    f.write(json_dumps(meta))

        # --- Common Finalization Logic for BOTH public and private ---
        if self.current_thread_keys:
            store_gene_keys(self.thread_uuid, self.current_thread_keys, privacy, self.agent_secret, self.agent_uuid)
        self.M.setdefault("metadata", {})["last_updated"] = datetime.datetime.now().isoformat()
        self.M.setdefault("metadata", {})["usage_count"] = self.M.get("metadata", {}).get("usage_count", 0) + 1
        store_format(self.M)
        # Update pattern index if it exists (conditional)
        if self.pattern_index:
            self.pattern_index.update_from_thread(self.thread_uuid, self.current_thread_keys)
        # --- FIX: ALWAYS clear the state buffers after finalizing ---
        self.active_thread_content.clear()
        self.current_thread_keys.clear()

    def generate_and_save_response(self, length: int = 100, privacy: str = "private") -> bytes:
        """
        Generate a response and append to current thread or create new one

        Args:
            length: Number of bytes to generate

        Returns:
            bytes: Generated response
        """
        # Generate the response using the intelligent selection algorithm
        response_bytes = self._generate_response_bytes(length)

        # Append to current thread content
        self._append_to_thread(
            {"type": "output", "data": base64.b64encode(response_bytes).decode("utf-8")}, privacy=privacy
        )

        return response_bytes

    def _generate_response_bytes(self, length: int) -> bytes:
        response_bytes = bytearray()
        for _ in range(length):
            output_byte, key_index = self._generate_response_byte()
            response_bytes.append(output_byte)
            # S4 instructs S3 to process this self-generated byte, creating a feedback loop
            _key_index, resonance = self.inference_engine.process_byte(output_byte)
            # Update learning state for this self-generated event
            self.update_learning_state(output_byte, _key_index, resonance, "OUTPUT", privacy="private")
        return bytes(response_bytes)

    def _generate_response_byte(self) -> Tuple[int, int]:
        """
        Generates a single, intelligent response byte by selecting the most
        semantically meaningful pattern from the set of physically resonant candidates.

        This method combines S3's physical resonance with S4's learned knowledge.

        Returns:
            A tuple containing:
            - output_byte (int): The selected byte value (0-255).
            - key_index (int): The index of the winning canonical pattern (0-255).
        """
        # 1. S3 Physics: Get all physically plausible next states.
        resonances = self.inference_engine.compute_pattern_resonances()
        resonant_threshold = np.pi / 2

        candidate_indices = [i for i, dist in enumerate(resonances) if dist < resonant_threshold]

        # If no patterns are physically resonant, fall back to the single closest one.
        if not candidate_indices:
            selected_pattern = int(np.argmin(resonances))
            output_byte = self.inference_engine.G[selected_pattern]
            if hasattr(output_byte, "item"):
                output_byte = output_byte.item()
            return int(output_byte), int(selected_pattern)

        # 2. S4 Linguistics: Evaluate the candidates based on learned meaning.
        best_candidate_index = -1
        max_combined_score = -1.0

        for index in candidate_indices:
            # Physical Score: How good is the physical match? (0 to 1)
            physical_score = 1.0 - (resonances[index] / np.pi)

            # Semantic Score: How meaningful is this pattern, based on long-term learning?
            # We use the pattern's learned confidence from the format metadata.
            pattern_meta = self.M.get("patterns", [])[index]
            # A pattern is only meaningful if it has an assigned character and some confidence.
            if pattern_meta.get("character") is not None:
                semantic_score = pattern_meta.get("confidence", 0.0)
            else:
                semantic_score = 0.0  # No character mapping means no semantic value.

            # Combined Score: A pattern is a great choice if it is both
            # physically resonant AND semantically meaningful.
            combined_score = physical_score * semantic_score

            if combined_score > max_combined_score:
                max_combined_score = combined_score
                best_candidate_index = index

        # If no candidate had any semantic meaning, fall back to the most resonant one.
        if best_candidate_index == -1:
            # We must choose from the original candidate_indices list.
            min_dist = float("inf")
            for idx in candidate_indices:
                if resonances[idx] < min_dist:
                    min_dist = resonances[idx]
                    best_candidate_index = idx

        selected_pattern = best_candidate_index

        # 3. Get the final output byte for the winning pattern.
        output_byte = self.inference_engine.G[selected_pattern]
        if hasattr(output_byte, "item"):
            output_byte = output_byte.item()

        return int(output_byte), int(selected_pattern)

    def update_learning_state(
        self, source_byte: int, key_index: int, resonance: float, event_type: str, privacy: str = "private"
    ) -> None:
        """
        Update learning state based on processed byte, including pattern
        statistics and confidence.
        """
        # 1. Update pattern metadata
        pattern = self.M.get("patterns", [])[key_index]

        # Increment count
        current_count = pattern.get("count", 0)
        pattern["count"] = current_count + 1

        # Update cycle info
        pattern["last_cycle"] = self.inference_engine.cycle_counter
        if pattern.get("first_cycle") is None:
            pattern["first_cycle"] = self.inference_engine.cycle_counter

        # --- Update confidence as a moving average of resonance ---
        current_confidence = pattern.get("confidence", 0.0)
        new_event_confidence = 1.0 - (resonance / np.pi)
        alpha = 0.01
        pattern["confidence"] = (1 - alpha) * current_confidence + alpha * new_event_confidence

        # 2. Record the FULL Gene Key
        gene_key_event: GeneKeysMetadata = {
            "cycle": self.inference_engine.cycle_counter,
            "pattern_index": int(key_index),
            "thread_uuid": self.thread_uuid or "",
            "agent_uuid": self.agent_uuid,  # Deprecated
            "format_uuid": self.format_uuid,
            "event_type": event_type,
            "source_byte": int(source_byte),
            "resonance": float(resonance),
            "created_at": datetime.datetime.now().isoformat(),
            "privacy": privacy,
        }
        self.current_thread_keys.append(gene_key_event)

    def encode(self, character_label: str) -> Optional[int]:
        """
        Find pattern index for character label
        Args:
            character_label: Character label to search for
        Returns:
            int: Pattern index or None if not found
        """
        for index, pattern in enumerate(self.M.get("patterns", [])):
            character = pattern.get("character")
            if isinstance(character, str) and character == character_label:
                return index
        return None

    def intelligent_encode(self, character_label: str) -> Optional[int]:
        """
        Intelligently finds the best pattern index for a character label
        by leveraging learned statistics (count and confidence).

        Args:
            character_label: The character to encode (e.g., "A").

        Returns:
            The pattern index that is the most reliable representation, or None.
        """
        candidate_patterns = []
        for index, p_meta in enumerate(self.M.get("patterns", [])):
            if p_meta.get("character") == character_label:
                candidate_patterns.append(p_meta)

        if not candidate_patterns:
            return None  # No pattern maps to this character

        if len(candidate_patterns) == 1:
            return candidate_patterns[0].get("index")  # Only one choice

        # Disambiguate using count and confidence
        best_candidate = None
        max_score = -1.0
        for candidate in candidate_patterns:
            count = candidate.get("count", 0)
            confidence = candidate.get("confidence", 0.0)
            score = count * confidence
            if score > max_score:
                max_score = score
                best_candidate = candidate
        if best_candidate is not None:
            return best_candidate.get("index")
        return None

    def decode(self, key_index: int) -> Optional[str]:
        """
        Decode a pattern index to its character label
        Args:
            key_index: Pattern index to decode
        Returns:
            str: Character label or None if not found
        """
        patterns = self.M.get("patterns", [])
        if 0 <= key_index < len(patterns):
            char = patterns[key_index].get("character")
            if isinstance(char, str):
                return char
        return None

    def load_thread_content(self, thread_uuid: str) -> Optional[list]:
        """
        Load a thread's decrypted content as a list of NDJSON event dicts.
        """
        thread_content_raw = load_thread(self.agent_uuid, thread_uuid)
        if not thread_content_raw:
            return None
        if self.agent_uuid and self.agent_secret:
            thread_key = load_thread_key(self.agent_uuid, thread_uuid, self.agent_secret)
            if not thread_key:
                return None
            # Use thread_key directly as the AES-256-GCM key
            aes_key = thread_key
            try:
                thread_content_raw = self._decrypt_data(thread_content_raw, aes_key)
            except Exception:
                return None
        # At this point, thread_content_raw is NDJSON (utf-8 bytes)
        try:
            text = thread_content_raw.decode("utf-8")
            events = []
            for line in text.splitlines():
                if not line.strip():
                    continue
                event = json_loads(line)
                # If the event has a 'data' field, decode base64
                if "data" in event:
                    try:
                        event["data"] = base64.b64decode(event["data"])
                    except Exception:
                        pass  # Leave as-is if not valid base64
                events.append(event)
            return events
        except Exception:
            return None

    def select_stable_format(self, domain: str, stability: str = "stable") -> Optional[str]:
        """
        Select a stable format for a specific domain

        Args:
            domain: Domain to find a format for (e.g., "english", "code")
            stability: Desired stability level ("stable", "beta", "experimental")

        Returns:
            str: UUID of selected format, or None if not found
        """
        format_uuids = list_formats()
        matching_formats = []
        for format_uuid in format_uuids:
            format_data = load_format(format_uuid)
            if not format_data:
                continue
            if format_data.get("stability") == stability:
                tags = format_data.get("metadata", {}).get("tags", [])
                description = format_data.get("metadata", {}).get("description", "")
                if domain in tags or domain.lower() in description.lower():
                    matching_formats.append(
                        {
                            "uuid": format_uuid,
                            "name": format_data.get("format_name"),
                            "usage_count": format_data.get("metadata", {}).get("usage_count", 0),
                        }
                    )
        if matching_formats:
            matching_formats.sort(key=lambda x: x["usage_count"], reverse=True)
            return matching_formats[0]["uuid"]
        return None

    def discover_formats_from_agent(self, agent_uuid: str) -> List[str]:
        """
        Discover formats used by another agent

        Args:
            agent_uuid: UUID of agent to discover formats from

        Returns:
            List[str]: List of discovered format UUIDs
        """
        format_uuids = list_formats()
        discovered_formats = []
        for format_uuid in format_uuids:
            format_data = load_format(format_uuid)
            if not format_data:
                continue
            author = format_data.get("metadata", {}).get("author", "")
            if agent_uuid in author:
                discovered_formats.append(format_uuid)
        return list(set(discovered_formats))

    def compose_formats(self, primary_format: str, secondary_formats: List[str]) -> Optional[str]:
        """
        Compose multiple formats for multi-domain capability

        Args:
            primary_format: UUID of primary format
            secondary_formats: List of secondary format UUIDs

        Returns:
            str: UUID of composed format, or None if composition failed
        """
        try:
            # Load the primary format
            primary_data = load_format(primary_format)
            if not primary_data:
                return None

            # Create a new composed format
            composed_format = primary_data.copy()
            composed_format["format_uuid"] = str(uuid.uuid4())
            composed_format["format_name"] = f"composed_{primary_data.get('format_name', '')}"
            composed_format["stability"] = "experimental"
            composed_format.setdefault("metadata", {})["created_at"] = datetime.datetime.now().isoformat()
            composed_format.setdefault("metadata", {})["last_updated"] = datetime.datetime.now().isoformat()
            composed_format.setdefault("metadata", {})["usage_count"] = 0
            composed_format.setdefault("metadata", {})["author"] = (
                f"agent_{self.agent_uuid[:8]}" if self.agent_uuid else "public"
            )
            composed_format.setdefault("metadata", {})["description"] = "Composed format from multiple sources"

            # Update dependencies
            composed_format.setdefault("compatibility", {})["depends_on"] = [primary_format] + secondary_formats

            # Process each secondary format
            for secondary_uuid in secondary_formats:
                secondary_data = load_format(secondary_uuid)
                if not secondary_data:
                    continue
                # Merge pattern metadata
                primary_patterns = composed_format.get("patterns")
                secondary_patterns = secondary_data.get("patterns")
                if not isinstance(primary_patterns, list) or not isinstance(secondary_patterns, list):
                    continue
                for i in range(256):
                    if i >= len(primary_patterns) or i >= len(secondary_patterns):
                        continue
                    primary_pattern = primary_patterns[i]
                    secondary_pattern = secondary_patterns[i]

                    # If primary doesn't have character but secondary does, use secondary's
                    primary_char = primary_pattern.get("character")
                    secondary_char = secondary_pattern.get("character")
                    if primary_char is None and isinstance(secondary_char, str):
                        primary_pattern["character"] = secondary_char

                    # Merge counts and confidences (weighted average)
                    p_count = primary_pattern.get("count", 0)
                    s_count = secondary_pattern.get("count", 0)
                    total_count = p_count + s_count

                    if total_count > 0:
                        # Update confidence with weighted average
                        p_conf = primary_pattern.get("confidence", 0.0)
                        s_conf = secondary_pattern.get("confidence", 0.0)
                        composed_format.get("patterns", [])[i]["confidence"] = (
                            p_conf * p_count + s_conf * s_count
                        ) / total_count

                # Add secondary format's tags
                composed_format.setdefault("metadata", {}).setdefault("tags", []).extend(
                    secondary_data.get("metadata", {}).get("tags", [])
                )
                # Remove duplicates
                composed_format.setdefault("metadata", {})["tags"] = list(
                    set(composed_format.get("metadata", {}).get("tags", []))
                )

            # Save the composed format
            store_format(composed_format)

            return composed_format["format_uuid"]

        except Exception:
            return None

    def _derive_file_key(
        self, epigenome_snapshot: np.ndarray, agent_uuid: Optional[str], thread_uuid: str
    ) -> Optional[bytes]:
        if agent_uuid is None:
            return None
        tensor_bytes = epigenome_snapshot.tobytes()
        salt = (agent_uuid + thread_uuid).encode("utf-8")
        # Derive a 32-byte AES key directly
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=default_backend())
        stateless_bytes = bytes([gene_stateless])
        key_material = tensor_bytes + stateless_bytes
        aes_key = kdf.derive(key_material)
        return aes_key

    def _derive_agent_key(self) -> Optional[bytes]:
        if self.agent_uuid is None or self.agent_secret is None:
            return None
        salt = self.agent_uuid.encode("utf-8")
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=default_backend())
        key = kdf.derive(self.agent_secret.encode("utf-8"))
        return key

    def _encrypt_data(self, data: bytes, key: bytes) -> bytes:
        """
        Encrypt data with AES-GCM

        Args:
            data: Data to encrypt
            key: Encryption key

        Returns:
            bytes: Encrypted data with nonce and tag
        """
        # Generate random nonce
        nonce = os.urandom(12)

        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
        encryptor = cipher.encryptor()

        # Encrypt data
        ciphertext = encryptor.update(data) + encryptor.finalize()

        # Return nonce + tag + ciphertext
        return nonce + encryptor.tag + ciphertext

    def _decrypt_data(self, encrypted_data: bytes, key: bytes) -> bytes:
        """
        Decrypt data with AES-GCM

        Args:
            encrypted_data: Encrypted data with nonce and tag
            key: Decryption key

        Returns:
            bytes: Decrypted data
        """
        # Extract nonce, tag, and ciphertext
        nonce = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]

        # Create cipher
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend())
        decryptor = cipher.decryptor()

        # Decrypt data
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        return plaintext

    def get_thread_relationships(self, thread_uuid: str) -> Dict:
        if self.agent_uuid is None:
            return {"parent": None, "children": []}
        parent_uuid = parent(self.agent_uuid, thread_uuid)
        child_uuids = children(self.agent_uuid, thread_uuid)
        return {"parent": parent_uuid, "children": child_uuids}

    def get_thread_chain(self, thread_uuid: str, max_depth: int = 5) -> List[str]:
        if self.agent_uuid is None:
            return []
        chain = []
        current_uuid = thread_uuid
        depth = 0
        while current_uuid and depth < max_depth:
            chain.insert(0, current_uuid)
            parent_uuid = parent(self.agent_uuid, current_uuid)
            if parent_uuid:
                current_uuid = parent_uuid
            else:
                break
            depth += 1
        return chain

    def _add_children_to_chain(self, parent_uuid: str, chain: List[str], max_depth: int) -> None:
        if self.agent_uuid is None:
            return
        if max_depth <= 0:
            return
        child_uuids = children(self.agent_uuid, parent_uuid)
        for child_uuid in child_uuids:
            if child_uuid not in chain:
                chain.append(child_uuid)
                self._add_children_to_chain(child_uuid, chain, max_depth - 1)

    def load_thread_with_context(self, thread_uuid: str, include_related: bool = True) -> Dict:
        """
        Load a thread with its related context from parent and child threads

        Args:
            thread_uuid: UUID of the thread to load
            include_related: Whether to include related threads

        Returns:
            Dict: Thread content with context information
        """
        # Load the main thread
        main_content = self.load_thread_content(thread_uuid)
        if main_content is None:
            return {"error": f"Thread {thread_uuid} not found"}

        result = {
            "thread_uuid": thread_uuid,
            "content": main_content,
            "size_bytes": len(main_content),
            "relationships": self.get_thread_relationships(thread_uuid),
        }

        if include_related:
            # Load related threads
            chain = self.get_thread_chain(thread_uuid)
            related_threads = []

            for related_uuid in chain:
                if related_uuid != thread_uuid:
                    related_content = self.load_thread_content(related_uuid)
                    if related_content:
                        parent_uuid = parent(self.agent_uuid, related_uuid) if self.agent_uuid is not None else None
                        relationship = "parent" if related_uuid == parent_uuid else "child"

                        related_threads.append(
                            {
                                "thread_uuid": related_uuid,
                                "content": related_content,
                                "size_bytes": len(related_content),
                                "relationship": relationship,
                            }
                        )

            result["related_threads"] = related_threads

        return result

    def get_thread_statistics(self) -> Dict:
        if self.agent_uuid is None:
            return {}
        private_dir = Path("memories/private/agents")
        agent_shard = shard_path(private_dir, self.agent_uuid)
        if agent_shard is None:
            return {}
        agent_dir = agent_shard / f"agent-{self.agent_uuid}"
        threads_dir = agent_dir / "threads"
        registry_path = threads_dir / "registry.json"
        if not registry_path.exists():
            return {}
        with open(registry_path, "r") as f:
            registry = json_loads(f.read())
        thread_uuids = registry.get("uuids", [])
        # Filter out invalid UUIDs to prevent errors
        valid_thread_uuids = []
        for uuid_str in thread_uuids:
            if isinstance(uuid_str, str) and len(uuid_str) == 36 and uuid_str.count("-") == 4:
                try:
                    # Validate UUID format
                    uuid.UUID(uuid_str)
                    valid_thread_uuids.append(uuid_str)
                except ValueError:
                    continue  # Skip invalid UUIDs
        thread_uuids = valid_thread_uuids
        max_size_bytes = (
            self.memory_prefs["storage_config"]["max_thread_size_mb"] * 1024 * 1024
            if self.memory_prefs and "storage_config" in self.memory_prefs
            else 1
        )
        stats = {
            "total_threads": len(thread_uuids),
            "total_size_bytes": 0,
            "capacity_usage_percent": 0,
            "thread_details": [],
            "relationship_stats": {"threads_with_parents": 0, "threads_with_children": 0, "isolated_threads": 0},
        }
        for thread_uuid in thread_uuids:
            thread_shard = shard_path(threads_dir, thread_uuid)
            meta_path = thread_shard / f"thread-{thread_uuid}.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r") as f:
                    meta = json_loads(f.read())
                size = meta.get("size_bytes", 0)
                stats["total_size_bytes"] += size
                has_parent = meta.get("parent_uuid") is not None
                has_children = len(meta.get("children", [])) > 0
                if has_parent:
                    stats["relationship_stats"]["threads_with_parents"] += 1
                if has_children:
                    stats["relationship_stats"]["threads_with_children"] += 1
                if not has_parent and not has_children:
                    stats["relationship_stats"]["isolated_threads"] += 1
                thread_detail = {
                    "thread_uuid": thread_uuid,
                    "thread_name": meta.get("thread_name"),
                    "curriculum": meta.get("curriculum"),
                    "tags": meta.get("tags"),
                    "size_bytes": size,
                    "capacity_percent": (size / max_size_bytes) * 100 if max_size_bytes else 0,
                    "has_parent": has_parent,
                    "has_children": has_children,
                    "child_count": len(meta.get("children", [])),
                }
                stats["thread_details"].append(thread_detail)
            except FileNotFoundError:
                continue
        if stats["total_threads"] > 0 and max_size_bytes:
            stats["capacity_usage_percent"] = (
                stats["total_size_bytes"] / (max_size_bytes * stats["total_threads"])
            ) * 100
        return stats

    def get_pattern_statistics(self, pattern_index: int) -> Dict:
        """
        Get comprehensive statistics about a specific pattern

        Args:
            pattern_index: The pattern index to analyze

        Returns:
            Dict: Pattern statistics including usage, relationships, and contexts
        """
        # Get pattern metadata from format
        pattern_meta = (
            self.M.get("patterns", [])[pattern_index] if pattern_index < len(self.M.get("patterns", [])) else {}
        )

        # Get historical contexts from pattern index
        contexts = {}
        if self.pattern_index and pattern_index in getattr(self.pattern_index, "pattern_contexts", {}):
            pattern_contexts = self.pattern_index.pattern_contexts[pattern_index]

            # Get most common preceding patterns
            before_patterns = sorted(pattern_contexts.get("before", {}).items(), key=lambda x: x[1], reverse=True)[:10]

            # Get most common following patterns
            after_patterns = sorted(pattern_contexts.get("after", {}).items(), key=lambda x: x[1], reverse=True)[:10]

            contexts = {"before": before_patterns, "after": after_patterns}

        # Get resonance with current state
        current_resonance = None
        if self.inference_engine:
            resonances = self.inference_engine.compute_pattern_resonances()
            if pattern_index < len(resonances):
                current_resonance = resonances[pattern_index]

        return {
            "pattern_index": pattern_index,
            "character": pattern_meta.get("character"),
            "count": pattern_meta.get("count", 0),
            "first_cycle": pattern_meta.get("first_cycle"),
            "last_cycle": pattern_meta.get("last_cycle"),
            "gyration_feature": pattern_meta.get("gyration_feature"),
            "confidence": pattern_meta.get("confidence", 0.0),
            "current_resonance": current_resonance,
            "contexts": contexts,
        }

    def load_thread_metadata(self, thread_uuid: str, privacy: str = "private") -> dict:
        """
        Load thread metadata from disk.
        """
        if privacy == "public":
            threads_dir = shard_path(Path("memories/public/threads"), thread_uuid)
        else:
            if self.agent_uuid is None:
                raise ValueError("agent_uuid must not be None for private thread metadata loading")
            private_dir = Path("memories/private/agents")
            agent_shard = shard_path(private_dir, self.agent_uuid)
            agent_dir = agent_shard / f"agent-{self.agent_uuid}"
            threads_dir = agent_dir / "threads"
            threads_dir = shard_path(threads_dir, thread_uuid)
        meta_path = threads_dir / f"thread-{thread_uuid}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Thread metadata not found for {thread_uuid}")
        with open(meta_path, "r") as f:
            return json_loads(f.read())

    def resume_thread(self, thread_uuid: str, privacy: str = "private") -> None:
        """
        Resume appending to an existing thread. Sets current_thread_size and opens file handle if public.
        """
        meta = self.load_thread_metadata(thread_uuid, privacy=privacy)
        self.current_thread_size = meta.get("size_bytes", 0)
        self.thread_uuid = thread_uuid
        if privacy == "public":
            if self._active_public_thread_handle:
                self._active_public_thread_handle.close()
                self._active_public_thread_handle = None
            threads_root = Path("memories/public/threads")
            thread_shard = shard_path(threads_root, thread_uuid)
            thread_path = thread_shard / f"thread-{thread_uuid}.ndjson"
            self._active_public_thread_handle = open(thread_path, "a", encoding="utf-8")
        # For private: buffer is not loaded for appending, as private threads are not typically resumed for append


def weighted_choice(items: List[Any], weights: List[float]) -> Any:
    """
    Select an item from a list based on weights

    Args:
        items: List of items to choose from
        weights: List of corresponding weights

    Returns:
        Selected item
    """
    # Convert weights to cumulative distribution
    cumulative_weights = []
    total = 0
    for w in weights:
        total += w
        cumulative_weights.append(total)

    # Select a random point
    r = random.random() * total

    # Find the corresponding item
    for i, cw in enumerate(cumulative_weights):
        if r <= cw:
            return items[i]

    # Fallback (should never happen)
    return items[-1]


def initialize_intelligence_engine(
    agent_uuid: Optional[str] = None,
    agent_secret: Optional[str] = None,
    format_uuid: Optional[str] = None,
    formats: Optional[FormatMetadata] = None,
) -> IntelligenceEngine:
    """
    Initialize the complete intelligence engine.
    - No args: Auto-detects mode. Private if 'baby_preferences.json' has a secret, else Public.
    - Explicit args: Initializes based on provided uuid/secret.
    """
    # Default to public mode
    _agent_uuid = None
    _agent_secret = None

    # If no explicit args were passed, try to auto-detect private mode
    if agent_uuid is None and agent_secret is None:
        baby_prefs_path = Path("baby/baby_preferences.json")
        if baby_prefs_path.exists():
            try:
                with open(baby_prefs_path, "r") as f:
                    baby_prefs = json_loads(f.read())
                loaded_secret = baby_prefs.get("agent_secret")
                if loaded_secret:
                    # Auto-private mode detected
                    _agent_uuid = ensure_agent_uuid()
                    _agent_secret = loaded_secret
            except (json.JSONDecodeError, IOError):
                pass  # Failed to read prefs, remain in public mode
    else:
        # Use explicit args for private mode
        _agent_uuid = agent_uuid or ensure_agent_uuid()
        _agent_secret = agent_secret
        # Optional: attempt to load secret if only UUID was provided
        if _agent_secret is None:
            baby_prefs_path = Path("baby/baby_preferences.json")
            if baby_prefs_path.exists():
                try:
                    with open(baby_prefs_path, "r") as f:
                        baby_prefs = json_loads(f.read())
                    loaded_secret = baby_prefs.get("agent_secret")
                    if loaded_secret:
                        _agent_secret = loaded_secret
                except (json.JSONDecodeError, IOError):
                    pass  # Remain with None if cannot load

    # Now, initialize the engines with the determined state
    inference_engine = InferenceEngine()
    information_engine = InformationEngine()
    return IntelligenceEngine(
        agent_uuid=_agent_uuid,
        agent_secret=_agent_secret,
        inference_engine=inference_engine,
        information_engine=information_engine,
        format_uuid=format_uuid,
        formats=formats,
    )
