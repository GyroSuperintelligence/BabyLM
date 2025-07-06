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
import glob
from typing import List, Dict, Any, Optional, Tuple
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

from baby.governance import apply_operation, gene_stateless, gene_add
from baby.inference import InferenceEngine
from baby.information import InformationEngine


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
        format_uuid: str,
        inference_engine: InferenceEngine,
        information_engine: InformationEngine,
        formats: Optional[Dict] = None,
    ):
        """
        Initialize the Intelligence Engine

        Args:
            agent_uuid: UUID of the current agent
            agent_secret: Persistent secret for encryption
            format_uuid: UUID of the active format
            inference_engine: The InferenceEngine instance
            information_engine: The InformationEngine instance
            formats: Optional format metadata dictionary
        """
        # Assign engines
        self.inference_engine = inference_engine
        self.information_engine = information_engine

        # Store agent identity
        self.agent_uuid = agent_uuid
        self.agent_secret = agent_secret

        # Format information
        self.format_uuid = format_uuid

        # Thread state
        self.thread_uuid = None
        self.thread_file_key = None
        self.current_thread_keys = []

        # Load or initialize format metadata
        self.M = formats if formats else self._load_or_init_formats()

        # Load memory preferences for configuration
        self.memory_prefs = self._load_memory_preferences()

        # Validate format compatibility
        self._validate_format_compatibility()

    def _validate_format_compatibility(self) -> None:
        """
        Validate the loaded format for compatibility with current CGM version

        Raises:
            ValueError: If the format is incompatible with the current CGM version
        """
        # Check format exists
        if not self.M:
            return  # New format will be created

        # Get current CGM version from preferences
        current_cgm_version = self.memory_prefs["format_config"]["default_cgm_version"]

        # Check format compatibility
        if "compatibility" in self.M:
            min_version = self.M["compatibility"].get("min_cgm_version")
            max_version = self.M["compatibility"].get("max_cgm_version")

            # Simple version comparison (this could be enhanced with proper semver comparison)
            if min_version and current_cgm_version < min_version:
                raise ValueError(
                    f"Format requires minimum CGM version {min_version}, but current is {current_cgm_version}"
                )

            if max_version and current_cgm_version > max_version:
                raise ValueError(
                    f"Format requires maximum CGM version {max_version}, but current is {current_cgm_version}"
                )

        # Validate policy mapping
        if "cgm_policies" in self.M:
            policies = self.M["cgm_policies"]
            required_policies = ["governance", "information", "inference", "intelligence"]

            for policy in required_policies:
                if policy not in policies:
                    raise ValueError(f"Required CGM policy '{policy}' missing from format")

    def _load_or_init_formats(self) -> Dict:
        """
        Load format metadata from file or initialize if not present

        Returns:
            Dict: Format metadata
        """
        formats_path = f"memories/public/formats/formats-{self.format_uuid}.json"

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(formats_path), exist_ok=True)

        try:
            # Try to load formats from file
            with open(formats_path, "r") as f:
                M = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Initialize new format metadata
            M = self._initialize_format_metadata()

            # Save to file
            with open(formats_path, "w") as f:
                json.dump(M, f, indent=2)

        return M

    def _load_memory_preferences(self) -> Dict:
        """
        Load memory preferences or initialize with defaults

        Returns:
            Dict: Memory preferences
        """
        registry_path = "memories/memory_preferences.json"

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)

        try:
            # Try to load preferences
            with open(registry_path, "r") as f:
                prefs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Initialize with defaults
            prefs = {
                "uuid_registry": {"agent_uuid": self.agent_uuid, "format_uuid": self.format_uuid, "thread_uuids": []},
                "storage_config": {
                    "max_thread_size_mb": 64,
                    "shard_prefix_length": 2,
                    "encryption_algorithm": "AES-256-GCM",
                },
                "format_config": {
                    "default_cgm_version": "1.0.0",
                    "resonance_threshold": float(np.pi / 2),
                    "max_semantic_label_length": 128,
                },
            }

            # Save defaults
            with open(registry_path, "w") as f:
                json.dump(prefs, f, indent=2)

        return prefs

    def _initialize_format_metadata(self) -> Dict:
        """
        Initialize new format metadata

        Returns:
            Dict: Initialized format metadata
        """
        # Create format metadata structure
        M = {
            "format_uuid": self.format_uuid,
            "format_name": "default_format",
            "cgm_version": "1.0.0",
            "format_version": "1.0.0",
            "stability": "experimental",
            "compatibility": {
                "min_cgm_version": "0.9.0",
                "max_cgm_version": "1.0.0",
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
        patterns = []
        for i in range(256):
            pattern = {
                "index": i,
                "semantic": None,
                "count": 0,
                "first_cycle": None,
                "last_cycle": None,
                "resonance_class": self.inference_engine.resonance_classes[i],
                "confidence": 0.0,
            }
            patterns.append(pattern)

        M["patterns"] = patterns

        return M

    def start_new_thread(self) -> str:
        """
        Start a new thread

        Returns:
            str: UUID of the new thread
        """
        # 1. Capture Epigenome state
        epigenome_snapshot = self.inference_engine.T.copy()

        # 2. Generate thread UUID
        self.thread_uuid = str(uuid.uuid4())

        # 3. Derive thread file key
        self.thread_file_key = self._derive_file_key(epigenome_snapshot, self.agent_uuid, self.thread_uuid)

        # 4. Reset observation log
        self.current_thread_keys = []

        # 5. Update UUID registry
        self._update_uuid_registry()

        return self.thread_uuid

    def _update_uuid_registry(self) -> None:
        """Update the UUID registry with the current thread"""
        # Add thread UUID if not already present
        if self.thread_uuid not in self.memory_prefs["uuid_registry"]["thread_uuids"]:
            self.memory_prefs["uuid_registry"]["thread_uuids"].append(self.thread_uuid)

        # Update last_updated timestamp
        self.M["metadata"]["last_updated"] = datetime.datetime.now().isoformat()
        self.M["metadata"]["usage_count"] += 1

        # Save updated registry
        with open("memories/memory_preferences.json", "w") as f:
            json.dump(self.memory_prefs, f, indent=2)

    def process_input_stream(self, input_stream: bytes) -> Tuple[bytes, bytes]:
        """
        Process an external input stream and save as a thread

        Args:
            input_stream: Bytes to process

        Returns:
            Tuple containing:
            - plaintext: Original input
            - intermediate_ciphertext: Encrypted form
        """
        # 1. Start a new thread
        self.start_new_thread()

        # 2. Process the stream
        intermediate_ciphertext, dynamic_keystream = self.information_engine.process_stream(
            self.inference_engine, self.update_learning_state, input_stream
        )

        # 3. End the thread, saving the original input as content
        self.end_current_thread(plaintext_to_save=input_stream)

        # 4. Return processed data for convenience
        return input_stream, intermediate_ciphertext

    def end_current_thread(self, plaintext_to_save: bytes) -> None:
        """
        End the current thread, encrypting the provided plaintext and saving all data

        This method is generalized to handle both input processing and self-generation.

        Args:
            plaintext_to_save: Content to encrypt and save
        """
        # Defensive: ensure memory_prefs and storage_config are present
        if not self.memory_prefs or not self.memory_prefs.get("storage_config"):
            raise RuntimeError("Memory preferences or storage config not initialized.")
        if not self.thread_uuid:
            raise RuntimeError("Thread UUID is not set.")
        shard = str(self.thread_uuid)[: self.memory_prefs["storage_config"]["shard_prefix_length"]]

        # 1. Re-encrypt with thread key
        if self.thread_file_key is None:
            raise RuntimeError("Thread file key is not set. Call start_new_thread() first.")
        # At this point, thread_file_key is guaranteed to be not None
        thread_file_key: bytes = self.thread_file_key
        final_encrypted_data = bytearray(len(plaintext_to_save))
        for i in range(len(plaintext_to_save)):
            final_encrypted_data[i] = plaintext_to_save[i] ^ thread_file_key[i % 256]

        # 2. Save thread file
        thread_dir = f"memories/private/{self.agent_uuid}/threads/{shard}"
        os.makedirs(thread_dir, exist_ok=True)

        thread_path = f"{thread_dir}/thread-{self.thread_uuid}.enc"
        with open(thread_path, "wb") as f:
            f.write(final_encrypted_data)

        # 3. Save Gene Keys
        keys_dir = f"memories/private/{self.agent_uuid}/keys"
        os.makedirs(keys_dir, exist_ok=True)

        keys_path = f"{keys_dir}/keys-{self.agent_uuid}.json.enc"
        agent_key = self._derive_agent_key()

        try:
            with open(keys_path, "rb") as f:
                encrypted_keys = f.read()
            decrypted_json_str = self._decrypt_data(encrypted_keys, agent_key)
            all_keys_data = json.loads(decrypted_json_str)
        except (FileNotFoundError, json.JSONDecodeError):
            all_keys_data = {}

        all_keys_data[str(self.thread_uuid)] = self.current_thread_keys

        updated_json_str = json.dumps(all_keys_data)
        encrypted_updated_keys = self._encrypt_data(updated_json_str.encode("utf-8"), agent_key)
        with open(keys_path, "wb") as f:
            f.write(encrypted_updated_keys)

        # 4. Save formats metadata
        formats_path = f"memories/public/formats/formats-{self.format_uuid}.json"
        with open(formats_path, "w") as f:
            json.dump(self.M, f, indent=2)

    def generate_and_save_response(self, length: int = 100) -> bytes:
        """
        Generate a response and save the generation process as a thread

        Args:
            length: Number of bytes to generate

        Returns:
            bytes: Generated response
        """
        # Start a new thread for this generation
        self.start_new_thread()

        # Generate the response using the intelligent selection algorithm
        response_bytes = self._generate_response_bytes(length)

        # End the thread, saving the generated response as content
        self.end_current_thread(plaintext_to_save=response_bytes)

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
        Generate a single, intelligently-selected response byte

        This is the S4 "conscious choice" mechanism, implementing section 4.5.3
        of the specification.

        Returns:
            Tuple containing:
            - output_byte: Selected byte value (0-255)
            - key_index: Index of the selected pattern (0-255)
        """
        # Always use the canonical pattern-matching route
        G = self.inference_engine.G

        # FIX: Call the correct method to get the index.
        # self.information_engine.tensor_to_output_byte returns the final byte, not the index.
        # We need the index directly from the inference engine.
        key_index = self.inference_engine.find_closest_pattern_index()

        output_byte = G[key_index]
        # Use .item() for numpy scalars (output_byte may be a numpy type)
        if hasattr(output_byte, "item"):
            output_byte = output_byte.item()
        # key_index is already an int from find_closest_pattern_index()
        return int(output_byte), int(key_index)

    def update_learning_state(self, key_index: int, inference_engine: InferenceEngine) -> None:
        """
        Update learning state based on processed byte

        Args:
            key_index: Index of matched pattern
            inference_engine: The InferenceEngine instance
        """
        # 1. Update pattern metadata
        self.M["patterns"][key_index]["count"] += 1
        self.M["patterns"][key_index]["last_cycle"] = inference_engine.cycle_counter
        if self.M["patterns"][key_index]["first_cycle"] is None:
            self.M["patterns"][key_index]["first_cycle"] = inference_engine.cycle_counter

        # 2. Record Gene Key
        gene_key_entry = {"cycle": inference_engine.cycle_counter, "pattern_index": key_index}
        self.current_thread_keys.append(gene_key_entry)

    def encode(self, semantic_label: str) -> Optional[int]:
        """
        Find pattern index for semantic label

        Args:
            semantic_label: Semantic label to search for

        Returns:
            int: Pattern index or None if not found
        """
        for index, pattern in enumerate(self.M["patterns"]):
            if pattern.get("semantic") == semantic_label:
                return index
        return None

    def decode(self, key_index: int) -> Optional[str]:
        """
        Find semantic label for pattern index

        Args:
            key_index: Pattern index to decode

        Returns:
            str: Semantic label or None if not set
        """
        return self.M["patterns"][key_index].get("semantic")

    def load_thread(self, thread_uuid: str) -> Optional[bytes]:
        """
        Load a thread's content with full state reconstruction

        Args:
            thread_uuid: UUID of thread to load

        Returns:
            bytes: Decrypted thread content, or None if not found
        """
        # 1. Validate thread UUID
        if thread_uuid not in self.memory_prefs["uuid_registry"]["thread_uuids"]:
            return None

        # 2. Determine shard and path
        shard = str(thread_uuid)[: self.memory_prefs["storage_config"]["shard_prefix_length"]]
        thread_path = f"memories/private/{self.agent_uuid}/threads/{shard}/thread-{thread_uuid}.enc"

        try:
            # 3. Load encrypted content
            with open(thread_path, "rb") as f:
                encrypted_data = f.read()

            # 4. Load thread keys to determine the initial state
            keys_path = f"memories/private/{self.agent_uuid}/keys/keys-{self.agent_uuid}.json.enc"
            agent_key = self._derive_agent_key()

            with open(keys_path, "rb") as f:
                encrypted_keys = f.read()

            decrypted_keys = self._decrypt_data(encrypted_keys, agent_key)
            all_keys = json.loads(decrypted_keys.decode("utf-8"))

            if thread_uuid not in all_keys:
                return None

            # 5. Replay operations to recreate the Epigenome state at thread creation
            # (thread_keys is not used, so do not assign it)

            # 6. Derive thread file key from reconstructed state
            initial_tensor = gene_add.copy().astype(np.float32)

            # Apply the stateless gene cycle (required initialization)
            # This should match exactly what happens in _initialize_epigenome
            gene_mutated = gene_stateless
            for i in range(8):
                if gene_mutated & (1 << i):
                    apply_operation(initial_tensor, i)

            # Note: We don't replay the thread operations because we only have pattern_index,
            # not the input bytes that were processed. The thread file key is derived from
            # the initial state, not the final state after processing.

            thread_file_key = self._derive_file_key(initial_tensor, self.agent_uuid, thread_uuid)

            # 7. Decrypt the thread content
            plaintext = bytearray(len(encrypted_data))
            for i in range(len(encrypted_data)):
                plaintext[i] = encrypted_data[i] ^ thread_file_key[i % 256]

            return bytes(plaintext)

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
        # Search formats directory for matching formats
        formats_dir = "memories/public/formats/"

        try:
            # Get all format files
            format_files = glob.glob(f"{formats_dir}formats-*.json")

            matching_formats = []

            for format_file in format_files:
                try:
                    with open(format_file, "r") as f:
                        format_data = json.load(f)

                    # Check if format matches criteria
                    if format_data.get("stability") == stability:
                        # Check if domain is in tags or description
                        tags = format_data.get("metadata", {}).get("tags", [])
                        description = format_data.get("metadata", {}).get("description", "")

                        if domain in tags or domain.lower() in description.lower():
                            matching_formats.append(
                                {
                                    "uuid": format_data.get("format_uuid"),
                                    "name": format_data.get("format_name"),
                                    "usage_count": format_data.get("metadata", {}).get("usage_count", 0),
                                }
                            )
                except (json.JSONDecodeError, IOError):
                    continue

            # Return the most used matching format
            if matching_formats:
                matching_formats.sort(key=lambda x: x["usage_count"], reverse=True)
                return matching_formats[0]["uuid"]

            return None

        except Exception:
            return None

    def discover_formats_from_agent(self, agent_uuid: str) -> List[str]:
        """
        Discover formats used by another agent

        Args:
            agent_uuid: UUID of agent to discover formats from

        Returns:
            List[str]: List of discovered format UUIDs
        """
        try:
            # Check if agent exists in memory
            memory_path = "memories/memory_preferences.json"

            with open(memory_path, "r") as f:
                all_prefs = json.load(f)

            # Search for agents that have registry entries
            discovered_formats = []

            # If the specific agent is in the registry
            if agent_uuid in str(all_prefs):
                # Look for the agent's format UUID
                if "format_uuid" in all_prefs.get("uuid_registry", {}):
                    discovered_formats.append(all_prefs["uuid_registry"]["format_uuid"])

                # Try to find public formats used by this agent
                formats_dir = "memories/public/formats/"
                format_files = glob.glob(f"{formats_dir}formats-*.json")

                for format_file in format_files:
                    try:
                        with open(format_file, "r") as f:
                            format_data = json.load(f)

                        # Check if format is associated with this agent
                        author = format_data.get("metadata", {}).get("author", "")
                        if agent_uuid in author:
                            discovered_formats.append(format_data.get("format_uuid"))
                    except:
                        continue

            # Remove duplicates and return
            return list(set(discovered_formats))

        except Exception:
            return []

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
            primary_path = f"memories/public/formats/formats-{primary_format}.json"
            with open(primary_path, "r") as f:
                primary_data = json.load(f)

            # Create a new composed format
            composed_format = primary_data.copy()
            composed_format["format_uuid"] = str(uuid.uuid4())
            composed_format["format_name"] = f"composed_{primary_data['format_name']}"
            composed_format["stability"] = "experimental"
            composed_format["metadata"]["created_at"] = datetime.datetime.now().isoformat()
            composed_format["metadata"]["last_updated"] = datetime.datetime.now().isoformat()
            composed_format["metadata"]["usage_count"] = 0
            composed_format["metadata"]["author"] = f"agent_{self.agent_uuid[:8]}"
            composed_format["metadata"]["description"] = "Composed format from multiple sources"

            # Update dependencies
            composed_format["compatibility"]["depends_on"] = [primary_format] + secondary_formats

            # Process each secondary format
            for secondary_uuid in secondary_formats:
                secondary_path = f"memories/public/formats/formats-{secondary_uuid}.json"

                try:
                    with open(secondary_path, "r") as f:
                        secondary_data = json.load(f)

                    # Merge pattern metadata
                    for i in range(256):
                        primary_pattern = composed_format["patterns"][i]
                        secondary_pattern = secondary_data["patterns"][i]

                        # If primary doesn't have semantic but secondary does, use secondary's
                        if primary_pattern.get("semantic") is None and secondary_pattern.get("semantic") is not None:
                            primary_pattern["semantic"] = secondary_pattern["semantic"]

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
                    composed_format["metadata"]["tags"].extend(secondary_data.get("metadata", {}).get("tags", []))
                    # Remove duplicates
                    composed_format["metadata"]["tags"] = list(set(composed_format["metadata"]["tags"]))

                except (FileNotFoundError, json.JSONDecodeError):
                    continue

            # Save the composed format
            composed_path = f"memories/public/formats/formats-{composed_format['format_uuid']}.json"
            with open(composed_path, "w") as f:
                json.dump(composed_format, f, indent=2)

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


def ensure_uuid_registry() -> Dict:
    """
    Ensure the UUID registry exists and contains necessary UUIDs

    Returns:
        Dict: UUID registry
    """
    registry_path = "memories/memory_preferences.json"

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)

    try:
        with open(registry_path, "r") as f:
            prefs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        prefs = {
            "uuid_registry": {},
            "storage_config": {
                "max_thread_size_mb": 64,
                "shard_prefix_length": 2,
                "encryption_algorithm": "AES-256-GCM",
            },
            "format_config": {
                "default_cgm_version": "1.0.0",
                "resonance_threshold": float(np.pi / 2),
                "max_semantic_label_length": 128,
            },
        }

    # Ensure uuid_registry exists
    if "uuid_registry" not in prefs:
        prefs["uuid_registry"] = {}

    # Ensure agent UUID exists
    if "agent_uuid" not in prefs["uuid_registry"]:
        prefs["uuid_registry"]["agent_uuid"] = str(uuid.uuid4())

    # Ensure format UUID exists
    if "format_uuid" not in prefs["uuid_registry"]:
        prefs["uuid_registry"]["format_uuid"] = str(uuid.uuid4())

    # Ensure thread_uuids list exists
    if "thread_uuids" not in prefs["uuid_registry"]:
        prefs["uuid_registry"]["thread_uuids"] = []

    # Save updated registry
    with open(registry_path, "w") as f:
        json.dump(prefs, f, indent=2)

    return prefs["uuid_registry"]


def initialize_intelligence_engine() -> IntelligenceEngine:
    """
    Initialize the complete intelligence engine

    Returns:
        IntelligenceEngine: Initialized intelligence engine
    """
    # 1. Ensure UUID registry exists and load it
    uuid_registry = ensure_uuid_registry()
    agent_uuid = uuid_registry["agent_uuid"]
    format_uuid = uuid_registry["format_uuid"]

    # 2. Load agent preferences
    baby_prefs_path = "baby/baby_preferences.json"
    os.makedirs(os.path.dirname(baby_prefs_path), exist_ok=True)

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
            "default_resonance_threshold": float(np.pi / 2),
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
        format_uuid=format_uuid,
        inference_engine=inference_engine,
        information_engine=information_engine,
    )

    return intelligence_engine
