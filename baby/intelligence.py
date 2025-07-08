"""
intelligence.py - S4 Orchestration for GyroSI Baby LM

This module implements the Intelligence Engine for orchestration, file I/O,
and thread lifecycle management, representing the Intelligence (S4) layer
of the Common Governance Model.
"""

import os
import json
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
    load_thread_key,
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
)
from baby.types import PatternMetadata, FormatMetadata

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
        agent_uuid: str,
        agent_secret: str,
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
        if agent_uuid is None:
            raise ValueError("agent_uuid must not be None")
        if agent_secret is None:
            raise ValueError("agent_secret must not be None")
        if inference_engine is None:
            raise ValueError("inference_engine must not be None")
        if information_engine is None:
            raise ValueError("information_engine must not be None")
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
        self.pattern_index = PatternIndex(self.agent_uuid, self.agent_secret) if self.agent_secret else None
        self.pattern_distances = None
        if self.M and "pattern_distances" in self.M and "path" in self.M["pattern_distances"]:
            self.pattern_distances = load_pattern_distances(self.format_uuid)

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
                "author": f"agent_{self.agent_uuid[:8]}",
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
                "gyration_feature": self.inference_engine.gyration_featurees[i],
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
                "author": f"agent_{self.agent_uuid[:8]}",
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
                "gyration_feature": self.inference_engine.gyration_featurees[i],
                "confidence": 0.0,
            }
            patterns.append(pattern)

        M["patterns"] = patterns

        return M

    def start_new_thread(self) -> str:
        """
        Start a new thread, setting its parent to the current thread if one exists.

        Returns:
            str: UUID of the newly created active thread.
        """
        # This method's job is to unconditionally start a new thread.
        # The decision to do so is made by the calling code (e.g., _append_to_thread).

        # Create a new thread, linking it to the previous one.
        parent_uuid = self.thread_uuid if self.thread_uuid else None
        new_thread_uuid = create_thread(self.agent_uuid, parent_uuid, self.format_uuid)

        # Reset the engine's state for the new thread.
        self.thread_uuid = new_thread_uuid
        self.thread_file_key = self._derive_file_key(self.inference_engine.T.copy(), self.agent_uuid, new_thread_uuid)
        self.current_thread_keys = []
        self.current_thread_size = 0
        self.active_thread_content = bytearray()
        self.parent_thread_uuid = parent_uuid
        self.child_thread_uuids = []

        # Store the new thread's key for future use.
        store_thread_key(self.agent_uuid, new_thread_uuid, self.thread_file_key, self.agent_secret)

        return new_thread_uuid

    def process_input_stream(self, input_stream: bytes) -> Tuple[bytes, bytes]:
        """
        Process an external input stream and append to current thread or create new one

        Args:
            input_stream: Bytes to process

        Returns:
            Tuple containing:
            - plaintext: Original input
            - intermediate_ciphertext: Encrypted form
        """
        # 1. Process the stream
        intermediate_ciphertext, dynamic_keystream = self.information_engine.process_stream(
            self.inference_engine, self.update_learning_state, input_stream
        )

        # 2. Append to current thread content
        self._append_to_thread(input_stream)

        # 3. Return processed data for convenience
        return input_stream, intermediate_ciphertext

    def _append_to_thread(self, new_content: bytes) -> None:
        """
        Append new content to the current thread, creating a new one if capacity is exceeded.

        Args:
            new_content: Content to append to the thread
        """
        # 1. Ensure a thread exists.
        if not self.thread_uuid:
            self.start_new_thread()
        # else: do nothing; assume state is already loaded

        # 2. Decide if a new thread is needed BEFORE modifying the current one.
        max_size_bytes = self.memory_prefs["storage_config"]["max_thread_size_mb"] * 1024 * 1024

        # A new thread is needed if the current thread has content AND adding the new
        # content would exceed the capacity.
        if self.current_thread_size > 0 and (self.current_thread_size + len(new_content) > max_size_bytes):
            # The current thread is full. Save it and start a new one.
            self._save_current_thread()
            self.start_new_thread()
            # The engine state (thread_uuid, current_thread_size, etc.) is now reset.

        # 3. Append content to the active thread (which might be the original or a new one).
        # Add a separator only if the thread we are about to write to is not empty.
        if self.active_thread_content:
            separator = b"\n---\n"
            self.active_thread_content.extend(separator)

        self.active_thread_content.extend(new_content)

        # 4. Reliably update the size and save the thread's current state.
        self.current_thread_size = len(self.active_thread_content)
        self._save_current_thread()

    def _save_current_thread(self) -> None:
        """Save the current thread content to disk"""
        if not self.active_thread_content or not self.thread_uuid:
            return

        # Use the existing end_current_thread logic but with accumulated content
        self.end_current_thread(plaintext_to_save=bytes(self.active_thread_content))

    def end_current_thread(self, plaintext_to_save: bytes) -> None:
        """
        End the current thread, encrypting the provided plaintext and saving all data

        This method is generalized to handle both input processing and self-generation.

        Args:
            plaintext_to_save: Content to encrypt and save
        """
        if not self.thread_uuid:
            raise RuntimeError("Thread UUID is not set.")

        if self.thread_file_key is None:
            raise RuntimeError("Thread file key is not set. Call start_new_thread() first.")

        # 1. Re-encrypt with thread key
        thread_file_key: bytes = self.thread_file_key
        final_encrypted_data = bytearray(len(plaintext_to_save))
        for i in range(len(plaintext_to_save)):
            final_encrypted_data[i] = plaintext_to_save[i] ^ thread_file_key[i % 256]

        # 2. Save thread file using information.py helper
        save_thread(self.agent_uuid, self.thread_uuid, bytes(final_encrypted_data), len(plaintext_to_save))

        # 3. Store gene keys in encrypted form
        store_gene_keys(self.agent_uuid, self.thread_uuid, self.current_thread_keys, self.agent_secret)

        # 4. Update format metadata
        self.M.setdefault("metadata", {})["last_updated"] = datetime.datetime.now().isoformat()
        self.M.setdefault("metadata", {})["usage_count"] = self.M.get("metadata", {}).get("usage_count", 0) + 1
        store_format(self.M)

        # 5. Update pattern index for faster inference
        if self.pattern_index:
            self.pattern_index.update_from_thread(self.thread_uuid, self.current_thread_keys)

    def generate_and_save_response(self, length: int = 100) -> bytes:
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
        self._append_to_thread(response_bytes)

        return response_bytes

    def _generate_response_bytes(self, length: int) -> bytes:
        """
        Generate a sequence of bytes using intelligent selection

        Args:
            length: Number of bytes to generate

        Returns:
            bytes: Generated response
        """
        response_bytes = bytearray()

        for _ in range(length):
            # 1. S4 makes an intelligent choice for the next byte
            output_byte, key_index = self._generate_response_byte()

            # 2. Append the byte to our response
            response_bytes.append(output_byte)

            # 3. S4 instructs S3 to process this self-generated byte, creating a feedback loop
            # This mutates the Epigenome tensor based on the system's own output
            self.inference_engine.process_byte(output_byte)

            # 4. S4 updates its own learning state based on this "conscious" action
            self.update_learning_state(key_index, self.inference_engine)

        return bytes(response_bytes)

    def _generate_response_byte(self) -> Tuple[int, int]:
        """
        Generate a single, intelligently-selected response byte using contextual awareness.

        This is the S4 "conscious choice" mechanism, implementing section 4.5.3
        of the specification with added historical context awareness.

        Returns:
            Tuple containing:
            - output_byte: Selected byte value (0-255)
            - key_index: Index of the selected pattern (0-255)
        """
        # Get contextual resonances if pattern index is available
        if self.pattern_index and self.pattern_index.pattern_contexts:
            resonances = self.inference_engine.compute_contextual_resonances(self.pattern_index.pattern_contexts)
        else:
            resonances = self.inference_engine.compute_pattern_resonances()

        # Apply π/2 threshold for resonant patterns
        resonant_threshold = np.pi / 2
        resonant_patterns = [j for j in range(256) if resonances[j] < resonant_threshold]

        # Handle no resonant patterns
        if len(resonant_patterns) == 0:
            closest_pattern = int(np.argmin(resonances))
            resonant_patterns = [closest_pattern]

        # Apply contextual weighting
        pattern_weights = []
        for pattern_idx in resonant_patterns:
            # Base weight from usage frequency
            usage_count = self.M.get("patterns", [])[pattern_idx].get("count", 0) + 1  # Add 1 to avoid zeros

            # Recency bias
            last_cycle = self.M.get("patterns", [])[pattern_idx].get("last_cycle")
            recency_factor = 1.0 if last_cycle is None else 1.0 / (self.inference_engine.cycle_counter - last_cycle + 1)

            # Resonance strength (closer = stronger)
            resonance_strength = 1.0 / (resonances[pattern_idx] + 0.1)

            # Historical context bias from pattern index
            historical_bias = 1.0
            if self.pattern_index and self.inference_engine.recent_patterns:
                last_pattern = self.inference_engine.recent_patterns[-1]
                likely_next = self.pattern_index.get_likely_next_patterns(last_pattern)
                for likely_pattern, probability in likely_next:
                    if likely_pattern == pattern_idx:
                        historical_bias = 1.0 + probability * 3.0  # Boost by up to 4x
                        break

            # Combined weight
            weight = usage_count * recency_factor * resonance_strength * historical_bias
            pattern_weights.append(weight)

        # Select pattern
        total_weight = sum(pattern_weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in pattern_weights]
            selected_pattern = weighted_choice(resonant_patterns, normalized_weights)
        else:
            selected_pattern = random.choice(resonant_patterns)

        # Get output byte
        output_byte = self.inference_engine.G[selected_pattern]

        # Ensure integers
        if hasattr(output_byte, "item"):
            output_byte = output_byte.item()

        return int(output_byte), int(selected_pattern)

    def update_learning_state(self, key_index: int, inference_engine: InferenceEngine) -> None:
        """
        Update learning state based on processed byte

        Args:
            key_index: Index of matched pattern
            inference_engine: The InferenceEngine instance
        """
        # 1. Update pattern metadata
        pattern = self.M.get("patterns", [])[key_index]
        pattern["count"] = pattern.get("count", 0) + 1
        pattern["last_cycle"] = inference_engine.cycle_counter
        if pattern.get("first_cycle") is None:
            pattern["first_cycle"] = inference_engine.cycle_counter

        # 2. Record Gene Key
        gene_key_entry = {"cycle": inference_engine.cycle_counter, "pattern_index": key_index}
        self.current_thread_keys.append(gene_key_entry)

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

    def decode(self, key_index: int) -> Optional[str]:
        """
        Find character label for pattern index
        Args:
            key_index: Pattern index to decode
        Returns:
            str: Character label or None if not set
        """
        pattern = self.M.get("patterns", [])[key_index]
        character = pattern.get("character")
        if isinstance(character, str):
            return character
        return None

    def load_thread_content(self, thread_uuid: str) -> Optional[bytes]:
        """
        Load a thread's decrypted content.

        Args:
            thread_uuid: UUID of thread to load

        Returns:
            bytes: Decrypted thread content, or None if not found
        """
        encrypted_data = load_thread(self.agent_uuid, thread_uuid)
        if not encrypted_data:
            return None
        thread_key = load_thread_key(self.agent_uuid, thread_uuid, self.agent_secret)
        if not thread_key:
            return None
        plaintext = bytearray(len(encrypted_data))
        for i in range(len(encrypted_data)):
            plaintext[i] = encrypted_data[i] ^ thread_key[i % 256]
        return bytes(plaintext)

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
            composed_format.setdefault("metadata", {})["author"] = f"agent_{self.agent_uuid[:8]}"
            composed_format.setdefault("metadata", {})["description"] = "Composed format from multiple sources"

            # Update dependencies
            composed_format.setdefault("compatibility", {})["depends_on"] = [primary_format] + secondary_formats

            # Process each secondary format
            for secondary_uuid in secondary_formats:
                secondary_data = load_format(secondary_uuid)
                if not secondary_data:
                    continue

                    # Merge pattern metadata
                    for i in range(256):
                        primary_pattern = composed_format["patterns"][i]
                        secondary_pattern = secondary_data["patterns"][i]

                        # If primary doesn't have character but secondary does, use secondary's
                        if primary_pattern.get("character") is None and secondary_pattern.get("character") is not None:
                            primary_pattern["character"] = secondary_pattern["character"]

                        # Merge counts and confidences (weighted average)
                        p_count = primary_pattern.get("count", 0)
                        s_count = secondary_pattern.get("count", 0)
                        total_count = p_count + s_count

                        if total_count > 0:
                            # Update confidence with weighted average
                            p_conf = primary_pattern.get("confidence", 0.0)
                            s_conf = secondary_pattern.get("confidence", 0.0)
                            composed_format["patterns"][i]["confidence"] = (
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

    def _derive_file_key(self, epigenome_snapshot: np.ndarray, agent_uuid: str, thread_uuid: str) -> bytes:
        """
        Derive a file encryption key from Epigenome state

        Args:
            epigenome_snapshot: Current Epigenome tensor
            agent_uuid: Agent UUID
            thread_uuid: Thread UUID

        Returns:
            bytes: 256-byte key for thread file encryption
        """
        # Convert tensor to bytes
        tensor_bytes = epigenome_snapshot.tobytes()

        # Create a salt from UUID combination
        salt = (agent_uuid + thread_uuid).encode("utf-8")

        # Use PBKDF2 to derive key
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=256, salt=salt, iterations=100000, backend=default_backend())

        # Derive key from tensor bytes and gene_stateless
        stateless_bytes = bytes([gene_stateless])
        key_material = tensor_bytes + stateless_bytes

        # Generate the key
        key = kdf.derive(key_material)

        return key

    def _derive_agent_key(self) -> bytes:
        """
        Derive encryption key for agent's private data

        Returns:
            bytes: 32-byte key for agent file encryption
        """
        # Create a salt from agent UUID
        salt = self.agent_uuid.encode("utf-8")

        # Use PBKDF2 to derive key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=default_backend()  # 256 bits
        )

        # Derive key from agent secret
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
        """
        Get the relationships for a specific thread

        Args:
            thread_uuid: UUID of the thread

        Returns:
            Dict: Thread relationships including parent and children
        """
        parent_uuid = parent(self.agent_uuid, thread_uuid)
        child_uuids = children(self.agent_uuid, thread_uuid)

        return {"parent": parent_uuid, "children": child_uuids}

    def get_thread_chain(self, thread_uuid: str, max_depth: int = 5) -> List[str]:
        """
        Get a chain of related threads (parent -> current -> children)

        Args:
            thread_uuid: UUID of the starting thread
            max_depth: Maximum depth to traverse

        Returns:
            List[str]: Chain of thread UUIDs
        """
        chain = []
        current_uuid = thread_uuid
        depth = 0

        # Go up the chain to find the root
        while current_uuid and depth < max_depth:
            chain.insert(0, current_uuid)  # Add to front of chain
            parent_uuid = parent(self.agent_uuid, current_uuid)
            if parent_uuid:
                current_uuid = parent_uuid
                depth += 1
            else:
                break

        # Now go down the chain from original thread (add children)
        self._add_children_to_chain(thread_uuid, chain, max_depth - depth)

        return chain

    def _add_children_to_chain(self, parent_uuid: str, chain: List[str], max_depth: int) -> None:
        """Recursively add children to the chain"""
        if max_depth <= 0:
            return

        child_uuids = children(self.agent_uuid, parent_uuid)
        for child_uuid in child_uuids:
            if child_uuid not in chain:  # Avoid cycles
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
                        parent_uuid = parent(self.agent_uuid, related_uuid)
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
        """
        Get comprehensive statistics about all threads

        Returns:
            Dict: Thread statistics including sizes, relationships, and capacity usage
        """
        # List all threads for this agent
        thread_uuids = []

        # Get agent directory
        private_dir = Path("memories/private/agents")
        agent_shard = shard_path(private_dir, self.agent_uuid)
        agent_dir = agent_shard / f"agent-{self.agent_uuid}"
        threads_dir = agent_dir / "threads"

        # Read registry
        registry_path = threads_dir / "registry.json"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                registry = json.load(f)
            thread_uuids = registry.get("uuids", [])

        max_size_bytes = self.memory_prefs["storage_config"]["max_thread_size_mb"] * 1024 * 1024

        stats = {
            "total_threads": len(thread_uuids),
            "total_size_bytes": 0,
            "capacity_usage_percent": 0,
            "thread_details": [],
            "relationship_stats": {"threads_with_parents": 0, "threads_with_children": 0, "isolated_threads": 0},
        }

        for thread_uuid in thread_uuids:
            # Get thread metadata
            thread_shard = shard_path(threads_dir, thread_uuid)
            meta_path = thread_shard / f"thread-{thread_uuid}.json"

            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                size = meta.get("size_bytes", 0)
                stats["total_size_bytes"] += size

                # Get relationships
                has_parent = meta.get("parent_uuid") is not None
                has_children = len(meta.get("child_uuids", [])) > 0

                if has_parent:
                    stats["relationship_stats"]["threads_with_parents"] += 1
                if has_children:
                    stats["relationship_stats"]["threads_with_children"] += 1
                if not has_parent and not has_children:
                    stats["relationship_stats"]["isolated_threads"] += 1

                thread_detail = {
                    "thread_uuid": thread_uuid,
                    "size_bytes": size,
                    "capacity_percent": (size / max_size_bytes) * 100,
                    "has_parent": has_parent,
                    "has_children": has_children,
                    "child_count": len(meta.get("child_uuids", [])),
                }
                stats["thread_details"].append(thread_detail)

            except FileNotFoundError:
                continue

        # Calculate overall capacity usage
        if stats["total_threads"] > 0:
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
        pattern_meta = self.M.get("patterns", [])[pattern_index]

        # Get historical contexts from pattern index
        contexts = {}
        if self.pattern_index and pattern_index in self.pattern_index.pattern_contexts:
            pattern_contexts = self.pattern_index.pattern_contexts[pattern_index]

            # Get most common preceding patterns
            before_patterns = sorted(pattern_contexts["before"].items(), key=lambda x: x[1], reverse=True)[:10]

            # Get most common following patterns
            after_patterns = sorted(pattern_contexts["after"].items(), key=lambda x: x[1], reverse=True)[:10]

            contexts = {"before": before_patterns, "after": after_patterns}

        # Get resonance with current state
        current_resonance = None
        if self.inference_engine:
            resonances = self.inference_engine.compute_pattern_resonances()
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


def initialize_intelligence_engine() -> IntelligenceEngine:
    """
    Initialize the complete intelligence engine

    Returns:
        IntelligenceEngine: Initialized intelligence engine
    """
    # 1. Get agent UUID
    agent_uuid = ensure_agent_uuid()

    # 2. Load agent preferences
    baby_prefs_path = Path("baby/baby_preferences.json")
    baby_prefs_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(baby_prefs_path, "r") as f:
            baby_prefs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Create default preferences
        baby_prefs = {
            "agent_secret": str(uuid.uuid4()),
            "log_level": "info",
            "response_length": 100,
            "learning_rate": 1.0,
        }
        with open(baby_prefs_path, "w") as f:
            json.dump(baby_prefs, f, indent=2)

    agent_secret = baby_prefs["agent_secret"]

    # 3. Initialize engines
    inference_engine = InferenceEngine()
    information_engine = InformationEngine()

    # 4. Create Intelligence Engine
    intelligence_engine = IntelligenceEngine(
        agent_uuid=agent_uuid,
        agent_secret=agent_secret,
        inference_engine=inference_engine,
        information_engine=information_engine,
    )

    return intelligence_engine
