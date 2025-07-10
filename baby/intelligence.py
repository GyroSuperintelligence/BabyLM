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
)
from baby.types import PatternMetadata, FormatMetadata, GeneKeysMetadata

import base64

# pyright: reportMissingModuleSource=false
try:
    import orjson as json

    def json_loads(s):
        if isinstance(s, str):
            s = s.encode("utf-8")
        return json.loads(s)

    def json_dumps(obj):
        return json.dumps(obj).decode("utf-8")

except ImportError:
    try:
        import ujson as json

        json_loads = json.loads
        json_dumps = json.dumps
    except ImportError:
        import json

        json_loads = json.loads
        json_dumps = json.dumps

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
                "gyration_feature": self.inference_engine.gyration_featurees[i],
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

        # Reset the engine's state for the new thread.
        self.thread_uuid = new_thread_uuid
        self.thread_file_key = self._derive_file_key(self.inference_engine.T.copy(), self.agent_uuid, new_thread_uuid)
        self.current_thread_keys = []
        self.current_thread_size = 0
        self.active_thread_content = bytearray()
        self.parent_thread_uuid = parent_uuid
        self.child_thread_uuids = []

        # Store the new thread's key for future use.
        if (
            privacy == "private"
            and self.agent_uuid is not None
            and self.agent_secret is not None
            and self.thread_file_key is not None
        ):
            store_thread_key(self.agent_uuid, new_thread_uuid, self.thread_file_key, self.agent_secret)

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
        """
        Append a structured event to the current thread as NDJSON, creating a new one if capacity is exceeded.

        Args:
            event: Structured event dict to append (e.g., {"type": "input", "data": <base64 str>})
        """
        # 1. Ensure a thread exists.
        if not self.thread_uuid:
            self.start_new_thread(privacy=privacy)
        # else: do nothing; assume state is already loaded

        # 2. Serialize event as NDJSON line
        json_line = json_dumps(event)
        if not json_line.endswith("\n"):
            json_line += "\n"
        json_bytes = json_line.encode("utf-8")

        # 3. Decide if a new thread is needed BEFORE modifying the current one.
        max_size_bytes = self.memory_prefs["storage_config"]["max_thread_size_mb"] * 1024 * 1024
        if self.current_thread_size > 0 and (self.current_thread_size + len(json_bytes) > max_size_bytes):
            self._save_current_thread(privacy=privacy)
            self.start_new_thread(privacy=privacy)

        # 4. Append NDJSON line to the active thread
        self.active_thread_content.extend(json_bytes)

        # 5. Reliably update the size and save the thread's current state.
        self.current_thread_size = len(self.active_thread_content)
        self._save_current_thread(privacy=privacy)

    def _save_current_thread(self, privacy: str = "private") -> None:
        """Save the current thread content to disk (NDJSON or encrypted NDJSON)"""
        if not self.active_thread_content or not self.thread_uuid:
            return

        # Use the existing end_current_thread logic but with accumulated content
        # Note: self.active_thread_content is now NDJSON (or encrypted NDJSON), not a binary blob.
        self.end_current_thread(plaintext_to_save=bytes(self.active_thread_content), privacy=privacy)

    def end_current_thread(self, plaintext_to_save: bytes, privacy: str = "private") -> None:
        """
        End the current thread, encrypting the provided NDJSON (if in private mode)
        and saving all data.
        """
        if not self.thread_uuid:
            raise RuntimeError("Thread UUID is not set.")

        final_data_to_save = plaintext_to_save
        if privacy == "private":
            if self.thread_file_key is None:
                raise RuntimeError("Thread file key is not set for a private thread. Call start_new_thread() first.")
            # Derive a 32-byte AES key from the 256-byte thread_file_key using PBKDF2HMAC with thread_uuid as salt
            salt = (self.thread_uuid).encode("utf-8")
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # AES-256 requires a 32-byte key
                salt=salt,
                iterations=100000,
                backend=default_backend(),
            )
            aes_key = kdf.derive(self.thread_file_key)
            final_data_to_save = self._encrypt_data(plaintext_to_save, aes_key)

        save_thread(privacy, self.thread_uuid, final_data_to_save, len(plaintext_to_save))
        store_gene_keys(self.thread_uuid, self.current_thread_keys, privacy, self.agent_secret)
        self.M.setdefault("metadata", {})["last_updated"] = datetime.datetime.now().isoformat()
        self.M.setdefault("metadata", {})["usage_count"] = self.M.get("metadata", {}).get("usage_count", 0) + 1
        store_format(self.M)
        if self.pattern_index:
            self.pattern_index.update_from_thread(self.thread_uuid, self.current_thread_keys)

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

        return best_candidate.get("index") if best_candidate else None

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
            # Derive the same 32-byte AES key as in end_current_thread
            salt = (thread_uuid).encode("utf-8")
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000, backend=default_backend()
            )
            aes_key = kdf.derive(thread_key)
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

    def _derive_file_key(
        self, epigenome_snapshot: np.ndarray, agent_uuid: Optional[str], thread_uuid: str
    ) -> Optional[bytes]:
        if agent_uuid is None:
            return None
        tensor_bytes = epigenome_snapshot.tobytes()
        salt = (agent_uuid + thread_uuid).encode("utf-8")
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=256, salt=salt, iterations=100000, backend=default_backend())
        stateless_bytes = bytes([gene_stateless])
        key_material = tensor_bytes + stateless_bytes
        key = kdf.derive(key_material)
        return key

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
            # A full UUID is 36 characters. Shard names are 2. This filters them out.
            if len(thread_uuid) < 36:
                continue
            thread_shard = shard_path(threads_dir, thread_uuid)
            meta_path = thread_shard / f"thread-{thread_uuid}.json"
            try:
                with open(meta_path, "r") as f:
                    meta = json_loads(f.read())
                size = meta.get("size_bytes", 0)
                stats["total_size_bytes"] += size
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
                    "thread_name": meta.get("thread_name"),
                    "curriculum": meta.get("curriculum"),
                    "tags": meta.get("tags"),
                    "size_bytes": size,
                    "capacity_percent": (size / max_size_bytes) * 100 if max_size_bytes else 0,
                    "has_parent": has_parent,
                    "has_children": has_children,
                    "child_count": len(meta.get("child_uuids", [])),
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
    # This is the ambiguous case: called with no arguments, or with explicit Nones.
    if agent_uuid is None and agent_secret is None:
        baby_prefs_path = Path("baby/baby_preferences.json")
        # Try to enter auto-private mode by checking preferences file.
        if baby_prefs_path.exists():
            try:
                with open(baby_prefs_path, "r") as f:
                    baby_prefs = json.loads(f.read())
                loaded_secret = baby_prefs.get("agent_secret")
                if loaded_secret:
                    # Success: A secret exists. We are in auto-private mode.
                    # Set the parameters and let the private-mode logic below handle it.
                    agent_uuid = ensure_agent_uuid()
                    agent_secret = loaded_secret
                # If no secret in file, fall through to public mode init.
            except (json.JSONDecodeError, IOError):
                # Corrupt or unreadable file, fall through to public mode init.
                pass
        # If after the above, uuid is still None, it must be public mode.
        if agent_uuid is None:
            inference_engine = InferenceEngine()
            information_engine = InformationEngine()
            return IntelligenceEngine(
                agent_uuid=None,
                agent_secret=None,
                inference_engine=inference_engine,
                information_engine=information_engine,
                format_uuid=format_uuid,
                formats=formats,
            )
    # --- If we reach here, we are in a private mode (explicitly or auto-detected) ---
    # Ensure we have a UUID (for cases where only a secret was passed, or for auto-mode)
    if agent_uuid is None:
        agent_uuid = ensure_agent_uuid()
    # Try to load secret if not provided (read-only private mode)
    if agent_secret is None:
        try:
            with open(Path("baby/baby_preferences.json"), "r") as f:
                agent_secret = json.loads(f.read()).get("agent_secret")
        except (FileNotFoundError, json.JSONDecodeError, IOError):
            agent_secret = None  # Remain read-only if file not found/invalid
    # Initialize the private engine
    inference_engine = InferenceEngine()
    information_engine = InformationEngine()
    return IntelligenceEngine(
        agent_uuid=agent_uuid,
        agent_secret=agent_secret,
        inference_engine=inference_engine,
        information_engine=information_engine,
        format_uuid=format_uuid,
        formats=formats,
    )
