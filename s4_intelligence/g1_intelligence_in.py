"""
g1_intelligence_in.py - Intelligence Engine API

The sole controller of S3 and the only module permitted to write into S2.
Exposes exactly one high-level API: process_stream.
"""

import os
import json
import uuid
import hashlib
import struct
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np

from s3_inference.g1_inference import GovernanceEngine, AcceptedOpPair, CycleComplete
from s3_inference.g2_inference import InformationEngine
from s3_inference.g3_inference import (
    InferenceEngine,
    CompressedBlock,
    PatternPromotion,
    GeneSnapshot,
)
from s4_intelligence.g2_intelligence_eg import byte_to_gyrations, gyrations_to_byte


def get_shard_from_uuid(uuid_str: str) -> str:
    """Extract shard prefix from UUID string."""
    clean_uuid = uuid_str.replace("-", "")
    return clean_uuid[:2].lower()


class IntelligenceEngine:
    """
    Main orchestration engine that coordinates S3 processing and S2 persistence.
    This is the only class that writes to S2 storage.
    """

    def __init__(
        self,
        agent_uuid: Optional[str] = None,
        base_path: str = "s2_information",
        encryption_enabled: bool = True,
    ):
        """
        Initialize the Intelligence Engine.

        Args:
            agent_uuid: Optional agent UUID. If None, generates a new one.
            base_path: Base path for S2 information storage
            encryption_enabled: Whether to enable GyroCrypt encryption
        """
        self.base_path = base_path
        self.agent_uuid = agent_uuid or str(uuid.uuid4())
        self.shard = get_shard_from_uuid(self.agent_uuid)

        # GyroCrypt encryption settings
        self._encryption_enabled = encryption_enabled  # Enable/disable encryption
        self._gyrocrypt_key: Optional[bytes] = None  # Will be loaded from session
        self._cycle_index = 0  # Current cycle counter for encryption

        # Load pack size from manifest
        manifest_path = os.path.join(base_path, "s2_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
                self.pack_size = manifest.get("pack_size", 4096)
        else:
            self.pack_size = 4096  # Default fallback

        # Make sure base directories exist
        self._ensure_directories()

        # Initialize S3 engines
        self.governance_engine = GovernanceEngine()
        self.information_engine = InformationEngine(
            os.path.join(base_path, "agency", "g2_information", "g2_information.dat")
        )
        self.inference_engine = InferenceEngine()

        # Load agent state and curriculum
        self._load_state()
        self._load_curriculum()

        # Initialize current pack file
        self.current_pack_uuid = None  # Will be set when first pack is opened
        self.current_pack_bytes = 0  # Count of payload bytes in current pack (excluding header)
        self.current_pack_file = None
        self._header_update_counter = 0  # Track when to update header

        # Cycle buffering for compression
        self._pending_cycle_buffer = []  # Buffer for current 48 op-pairs
        self._cycle_compression_enabled = True  # Enable/disable compression

        self._open_current_pack()

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        # Base paths
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "agency"), exist_ok=True)

        # Agency directories
        for subdir in ["g1_information", "g2_information", "g4_information", "g5_information"]:
            os.makedirs(os.path.join(self.base_path, "agency", subdir), exist_ok=True)

        # Shard directories for agency
        for subdir in ["g1_information", "g4_information", "g5_information"]:
            shard_path = os.path.join(self.base_path, "agency", subdir, self.shard)
            os.makedirs(shard_path, exist_ok=True)

        # Agent directories
        agent_path = os.path.join(self.base_path, "agents", self.shard, self.agent_uuid)
        os.makedirs(os.path.join(agent_path, "g4_information"), exist_ok=True)
        os.makedirs(os.path.join(agent_path, "g5_information"), exist_ok=True)

    def _load_state(self):
        """Load agent state and global navigation logs."""
        # Load agent session if exists
        session_path = os.path.join(
            self.base_path, "agents", self.shard, self.agent_uuid, "g5_information", "session.json"
        )

        if os.path.exists(session_path):
            with open(session_path, "r") as f:
                self.session = json.load(f)
                # Load GyroCrypt key if present
                if "gyrocrypt_key" in self.session:
                    import base64

                    self._gyrocrypt_key = base64.b64decode(self.session["gyrocrypt_key"])
                self._cycle_index = self.session.get("cycle_index", 0)
        else:
            self.session = {
                "agent_uuid": self.agent_uuid,
                "created": datetime.utcnow().isoformat(),
                "last_checkpoint": None,
                "phase": 0,
                "cycle_count": 0,
                "active_curriculum": None,
                "compression_metadata": [],  # Store compression details here
            }

            # Add GyroCrypt fields only if encryption is enabled
            if self._encryption_enabled:
                self.session["gyrocrypt_key"] = None  # Will be generated
                self.session["cycle_index"] = 0  # Encryption cycle counter

                # Generate GyroCrypt key if encryption is enabled
                import secrets
                import base64

                # Generate 32-byte key
                key_bytes = secrets.token_bytes(32)
                self._gyrocrypt_key = key_bytes
                self.session["gyrocrypt_key"] = base64.b64encode(key_bytes).decode("utf-8")
                self.session["cycle_index"] = 0

        # Load active curriculum
        self._load_curriculum()

    def _load_curriculum(self):
        """Load the active curriculum/dictionary."""
        # First check agent-specific curriculum
        agent_curriculum_path = os.path.join(
            self.base_path,
            "agents",
            self.shard,
            self.agent_uuid,
            "g4_information",
            "curriculum.json",
        )

        if os.path.exists(agent_curriculum_path):
            with open(agent_curriculum_path, "r") as f:
                self.curriculum = json.load(f)
                return

        # Otherwise load latest global curriculum
        global_curriculum_dir = os.path.join(self.base_path, "agency", "g4_information", self.shard)

        if os.path.exists(global_curriculum_dir):
            # Get most recent curriculum
            files = sorted(
                [f for f in os.listdir(global_curriculum_dir) if f.endswith("-curriculum.json")]
            )
            if files:
                with open(os.path.join(global_curriculum_dir, files[-1]), "r") as f:
                    self.curriculum = json.load(f)
                    return

        # Default empty curriculum
        self.curriculum = {
            "version": "1.0",
            "patterns": {},
            "byte_to_token": {},  # Map bytes to tokens
            "token_to_byte": {},  # Reverse mapping
            "metadata": {},
        }

    def process_stream(self, data_bytes: bytes) -> Dict[str, List[Any]]:
        """
        Process a stream of bytes through the full GyroSI pipeline.

        This is the main API endpoint that:
        1. Splits each byte → two op-pairs via byte_to_gyrations()
        2. Feeds op-pairs through all S3 engines
        3. Persists outputs into S2
        4. Returns all emitted artifacts

        Args:
            data_bytes: Raw byte stream to process

        Returns:
            Dictionary containing all emitted artifacts:
            - accepted_ops: List of accepted operation pairs
            - resonances: List of resonance classifications
            - compressed_blocks: List of compressed cycle data
            - pattern_promotions: List of newly discovered patterns
            - gene_snapshots: List of gene state snapshots
        """
        artifacts = {
            "accepted_ops": [],
            "resonances": [],
            "compressed_blocks": [],
            "pattern_promotions": [],
            "gene_snapshots": [],
        }

        # Process each byte through the pipeline
        for byte_val in data_bytes:
            # Convert byte to two op-pairs
            op_pair1, op_pair2 = byte_to_gyrations(byte_val)

            # Process first op-pair
            # Get resonance classification first
            info_event1 = self.information_engine.process_accepted_op_pair(
                self.governance_engine.phase, op_pair1, byte_val
            )
            artifacts["resonances"].append(info_event1)

            # Process with resonance flag
            gov_events1 = self.governance_engine.process_op_pair(
                op_pair1, info_event1.resonance_flag
            )

            # Process governance events for first op-pair
            for event in gov_events1:
                if isinstance(event, AcceptedOpPair):
                    # Buffer op-pair for cycle compression
                    self._write_op_pair(event.op_pair)
                    artifacts["accepted_ops"].append(event)

            # Process second op-pair
            # Get resonance classification first
            info_event2 = self.information_engine.process_accepted_op_pair(
                self.governance_engine.phase, op_pair2, byte_val
            )
            artifacts["resonances"].append(info_event2)

            # Process with resonance flag
            gov_events2 = self.governance_engine.process_op_pair(
                op_pair2, info_event2.resonance_flag
            )

            # Process governance events for second op-pair
            for event in gov_events2:
                if isinstance(event, AcceptedOpPair):
                    # Buffer op-pair for cycle compression
                    self._write_op_pair(event.op_pair)
                    artifacts["accepted_ops"].append(event)

            # Process cycle completion events
            for event in gov_events1 + gov_events2:
                if isinstance(event, CycleComplete):
                    # Process inference events first to detect compression
                    inference_events = self.inference_engine.process_cycle_complete(event.op_pairs)

                    # Analyze cycle for pruning
                    analysis = self.inference_engine.analyse_cycle(
                        event.op_pairs, event.resonance_flags
                    )

                    # Check if cycle should be pruned
                    if analysis["prune"]:
                        print(f"Pruning cycle {event.cycle_number}: {analysis['prune_reason']}")
                        # Skip writing this cycle entirely
                        self._pending_cycle_buffer = []  # Clear buffer
                        continue

                    # Write the buffered cycle (compressed or raw)
                    if len(self._pending_cycle_buffer) == 48:
                        # Check if this cycle is a repeat by looking at compressed blocks
                        is_repeat = False
                        repeat_count = 1
                        cycle_hash = None
                        compressed_data = None

                        # Look for cycle repeat in the inference events
                        for inf_event in inference_events:
                            if (
                                isinstance(inf_event, CompressedBlock)
                                and inf_event.block_type == "cycle_repeat"
                            ):
                                is_repeat = True
                                # Extract from the data dictionary structure
                                repeat_count = inf_event.data.get("count", 1)
                                cycle_hash_str = inf_event.data.get("hash", "")
                                # Convert hex string to integer for storage
                                cycle_hash = (
                                    int(cycle_hash_str, 16)
                                    if cycle_hash_str
                                    else hash(tuple(self._pending_cycle_buffer))
                                )
                                compressed_data = inf_event.data

                                # Store compression metadata in session
                                if "compression_metadata" not in self.session:
                                    self.session["compression_metadata"] = []
                                self.session["compression_metadata"].append(
                                    {
                                        "cycle_hash": cycle_hash_str,
                                        "repeat_count": repeat_count,
                                        "cycle_index": self.session.get("cycle_count", 0),
                                        "timestamp": datetime.utcnow().isoformat(),
                                    }
                                )
                                break

                        # Write the cycle (compressed or raw)
                        self._write_cycle(self._pending_cycle_buffer, analysis, compressed_data)

                        # Clear the buffer
                        self._pending_cycle_buffer = []

                        # Process all inference events
                        for inf_event in inference_events:
                            if isinstance(inf_event, CompressedBlock):
                                artifacts["compressed_blocks"].append(inf_event)
                            elif isinstance(inf_event, PatternPromotion):
                                artifacts["pattern_promotions"].append(inf_event)
                                self._persist_pattern_promotion(inf_event)
                            elif isinstance(inf_event, GeneSnapshot):
                                artifacts["gene_snapshots"].append(inf_event)

        # Update session
        self._update_session()

        return artifacts

    def generate(self, prompt: bytes = b"", max_length: int = 100) -> bytes:
        """
        Generate bytes based on learned patterns.

        Args:
            prompt: Initial bytes to seed the generation
            max_length: Maximum number of bytes to generate

        Returns:
            Generated byte sequence
        """
        # Process the prompt first to set the initial state
        if prompt:
            self.process_stream(prompt)

        generated = bytearray()

        for _ in range(max_length):
            # Use the InferenceEngine to predict the next operation
            op_pair1 = self.inference_engine.predict_next_operation(self.curriculum)
            op_pair2 = self.inference_engine.predict_next_operation(self.curriculum)

            # Convert the predicted op_pairs to a byte
            next_byte = gyrations_to_byte(op_pair1, op_pair2)
            generated.append(next_byte)

            # Process the generated byte to update state
            self.process_stream(bytes([next_byte]))

        return bytes(generated)

    def _open_current_pack(self):
        """Open a new pack file for writing."""
        if self.current_pack_file:
            # Update header count before closing
            self._update_header_count()
            self.current_pack_file.flush()  # Guarantee header hits disk
            self.current_pack_file.close()

        # Generate new UUID for this pack
        self.current_pack_uuid = str(uuid.uuid4())

        pack_dir = os.path.join(self.base_path, "agency", "g1_information", self.shard)
        os.makedirs(pack_dir, exist_ok=True)

        pack_path = os.path.join(pack_dir, f"{self.current_pack_uuid}-genome.dat")
        self.current_pack_file = open(pack_path, "wb")

        # Write GyroCrypt header: 32B anchor + 4B cycle_index + 4B salt
        if self._encryption_enabled and self._gyrocrypt_key:
            # Use constant gene anchor for pack header (never changes)
            from s1_governance import get_gene_anchor

            gene_anchor = get_gene_anchor()

            # Get current gene snapshot for salt (changes with state)
            gene_snapshot = self.get_gene_snapshot()
            if len(gene_snapshot) == 32:  # SHA-256 hash
                self.current_pack_file.write(gene_anchor)  # 32B constant anchor
                self.current_pack_file.write(struct.pack("<I", self._cycle_index))  # 4B cycle_index
                self.current_pack_file.write(
                    gene_snapshot[:4]
                )  # 4B salt (first 4 bytes of current snapshot)
            else:
                # Fallback to old format if no valid snapshot
                self.current_pack_file.write(struct.pack("B", 1))  # version
                self.current_pack_file.write(struct.pack(">Q", int(datetime.utcnow().timestamp())))
                self.current_pack_file.write(struct.pack(">I", 0))  # op count
        else:
            # Old format for non-encrypted packs
            self.current_pack_file.write(struct.pack("B", 1))  # version
            self.current_pack_file.write(struct.pack(">Q", int(datetime.utcnow().timestamp())))
            self.current_pack_file.write(struct.pack(">I", 0))  # op count

        self.current_pack_bytes = 0
        self._header_update_counter = 0

    def _write_op_pair(self, op_pair):
        """Buffer a single op-pair for cycle compression."""
        # Add to pending cycle buffer
        self._pending_cycle_buffer.append(op_pair)

        # If we have a full cycle (48 op-pairs), we'll write it when CycleComplete arrives
        if len(self._pending_cycle_buffer) >= 48:
            # This should be handled by the cycle completion logic
            pass

    def _write_cycle(self, op_pairs, analysis, compressed_data=None):
        """
        Write a complete cycle to the pack file (compressed or raw).

        Args:
            op_pairs: List of 48 op-pairs
            analysis: Dict with pruning analysis results
            compressed_data: Optional compressed data from InferenceEngine
        """
        # Check if we have compressed data
        is_repeat = compressed_data is not None and "count" in compressed_data
        repeat_count = compressed_data.get("count", 1) if compressed_data else 1

        if not self._cycle_compression_enabled or not is_repeat:
            # Write raw 48 op-pairs
            cycle_data = bytearray()
            for op_pair in op_pairs:
                op_code, tensor_id = op_pair
                byte_val = (op_code << 4) | tensor_id
                cycle_data.append(byte_val)

            # Apply encryption if enabled
            if self._encryption_enabled and self._gyrocrypt_key:
                # Get gene snapshot for keystream generation
                gene_snapshot = self.get_full_gene_snapshot()
                if len(gene_snapshot) == 96 and self._gyrocrypt_key is not None:  # Full snapshot
                    keystream = self._make_keystream(gene_snapshot, self._gyrocrypt_key)
                    # XOR encrypt the cycle data
                    for i in range(len(cycle_data)):
                        cycle_data[i] ^= keystream[i]
                    # Increment cycle index
                    self._cycle_index += 1
                else:
                    # Fallback: write unencrypted if no valid snapshot
                    pass

            # Write the cycle data
            self._write_encrypted_cycle_data(bytes(cycle_data))
        else:
            # Write compressed repeat token
            cycle_hash = None
            if compressed_data and "hash" in compressed_data:
                cycle_hash_str = compressed_data["hash"]
                cycle_hash = int(cycle_hash_str, 16) if cycle_hash_str else hash(tuple(op_pairs))
            self._write_compressed_cycle(repeat_count, cycle_hash)

    def _write_raw_op_pair(self, op_pair):
        """Write a single op-pair directly to the pack file."""
        # Check if current pack would exceed pack_size (excluding header)
        header_size = (
            40 if (self._encryption_enabled and self._gyrocrypt_key) else 13
        )  # GyroCrypt vs old format
        payload_size = self.current_pack_bytes + 1  # +1 for the new op-pair

        if payload_size >= (self.pack_size - header_size):
            # Current pack would exceed size limit, start new one
            self._open_current_pack()

        # Ensure file is open
        if self.current_pack_file is None:
            self._open_current_pack()

        # At this point, file is definitely open
        assert self.current_pack_file is not None

        # Pack op-pair as 4-bit values into one byte
        op_code, tensor_id = op_pair
        byte_val = (op_code << 4) | tensor_id
        self.current_pack_file.write(struct.pack("B", byte_val))

        self.current_pack_bytes += 1
        self._header_update_counter += 1

        # Update op count in header periodically (every 256 ops) to reduce I/O
        if self._header_update_counter >= 256:
            self._update_header_count()
            self._header_update_counter = 0

    def _write_compressed_cycle(self, repeat_count, cycle_hash):
        """Write a compressed cycle repeat token."""
        # Check if current pack would exceed pack_size (excluding header)
        header_size = (
            40 if (self._encryption_enabled and self._gyrocrypt_key) else 13
        )  # GyroCrypt vs old format
        # Simplified compressed token: 1 byte type + 1 byte repeat count = 2 bytes
        payload_size = self.current_pack_bytes + 2

        if payload_size >= (self.pack_size - header_size):
            # Current pack would exceed size limit, start new one
            self._open_current_pack()

        # Ensure file is open
        if self.current_pack_file is None:
            self._open_current_pack()

        # At this point, file is definitely open
        assert self.current_pack_file is not None

        # Write simplified compressed token
        self.current_pack_file.write(struct.pack("B", 0xFF))  # Compression marker
        self.current_pack_file.write(
            struct.pack("B", min(repeat_count, 255))
        )  # Repeat count (capped at 255)

        self.current_pack_bytes += 2
        self._header_update_counter += 2

        # Update op count in header periodically
        if self._header_update_counter >= 256:
            self._update_header_count()
            self._header_update_counter = 0

    def _update_header_count(self):
        """Update the op count in the pack header."""
        if self.current_pack_file:
            if self._encryption_enabled and self._gyrocrypt_key:
                # GyroCrypt format: update cycle_index at offset 32
                self.current_pack_file.seek(32)
                self.current_pack_file.write(struct.pack("<I", self._cycle_index))
                self.current_pack_file.seek(0, 2)  # Go back to end
            else:
                # Old format: update op count at offset 9
                self.current_pack_file.seek(9)  # Skip version and timestamp
                self.current_pack_file.write(struct.pack(">I", self.current_pack_bytes))
                self.current_pack_file.seek(0, 2)  # Go back to end

    def _persist_pattern_promotion(self, promotion: PatternPromotion):
        """Persist a newly promoted pattern to curriculum."""
        # Update curriculum
        self.curriculum["patterns"][promotion.pattern_id] = {
            "sequence": [{"op": op, "tensor": tid} for op, tid in promotion.op_sequence],
            "frequency": promotion.frequency,
            "discovered": datetime.utcnow().isoformat(),
        }

        # Write new curriculum version
        curriculum_uuid = str(uuid.uuid4())
        curriculum_dir = os.path.join(self.base_path, "agency", "g4_information", self.shard)
        os.makedirs(curriculum_dir, exist_ok=True)

        curriculum_path = os.path.join(curriculum_dir, f"{curriculum_uuid}-curriculum.json")
        with open(curriculum_path, "w") as f:
            json.dump(self.curriculum, f, indent=2)

        # Also update the agent's local copy
        agent_curriculum_path = os.path.join(
            self.base_path,
            "agents",
            self.shard,
            self.agent_uuid,
            "g4_information",
            "curriculum.json",
        )
        with open(agent_curriculum_path, "w") as f:
            json.dump(self.curriculum, f, indent=2)

    def _update_session(self):
        """Update and persist session state."""
        # Update session data
        gov_state = self.governance_engine.get_state()
        self.session.update(
            {
                "phase": gov_state["phase"],
                "cycle_count": gov_state["cycle_count"],
                "last_checkpoint": datetime.utcnow().isoformat(),
            }
        )

        # Update GyroCrypt cycle index if encryption is enabled
        if self._encryption_enabled and self._gyrocrypt_key:
            self.session["cycle_index"] = self._cycle_index

        # Ensure agent directories exist
        agent_dir = os.path.join(self.base_path, "agents", self.shard, self.agent_uuid)
        os.makedirs(os.path.join(agent_dir, "g4_information"), exist_ok=True)
        os.makedirs(os.path.join(agent_dir, "g5_information"), exist_ok=True)

        # Write session with backup
        session_path = os.path.join(agent_dir, "g5_information", "session.json")
        backup_path = session_path + ".bak"

        # Backup existing if present
        if os.path.exists(session_path):
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.rename(session_path, backup_path)

        # Write new session
        with open(session_path, "w") as f:
            json.dump(self.session, f, indent=2)

    def learn_token_mapping(self, tokens_bytes: Dict[str, bytes]):
        """
        Learn mappings between tokens and byte sequences.

        Args:
            tokens_bytes: Dictionary mapping token strings to byte sequences
        """
        for token, byte_seq in tokens_bytes.items():
            # Store byte sequence as list of integers
            byte_list = list(byte_seq)

            # Update curriculum dictionaries
            byte_key = str(byte_list)  # Use string representation as key
            self.curriculum["byte_to_token"][byte_key] = token
            self.curriculum["token_to_byte"][token] = byte_list

        # Persist updated curriculum
        self._persist_curriculum()

    def _persist_curriculum(self):
        """Save the current curriculum."""
        # Write to agent's local copy
        agent_curriculum_path = os.path.join(
            self.base_path,
            "agents",
            self.shard,
            self.agent_uuid,
            "g4_information",
            "curriculum.json",
        )
        with open(agent_curriculum_path, "w") as f:
            json.dump(self.curriculum, f, indent=2)

        # Write new global version
        curriculum_uuid = str(uuid.uuid4())
        curriculum_dir = os.path.join(self.base_path, "agency", "g4_information", self.shard)
        curriculum_path = os.path.join(curriculum_dir, f"{curriculum_uuid}-curriculum.json")
        with open(curriculum_path, "w") as f:
            json.dump(self.curriculum, f, indent=2)

    def get_gene_snapshot(self) -> bytes:
        """
        Get current gene snapshot for key derivation.

        Returns:
            96-byte gene configuration state
        """
        snapshot = self.inference_engine.get_gene_snapshot()
        return snapshot.snapshot_hash

    def get_full_gene_snapshot(self) -> bytes:
        """
        Get full 96-byte gene snapshot for GyroCrypt keystream generation.

        Returns:
            96-byte gene state derived from tensor data
        """
        snapshot = self.inference_engine.get_gene_snapshot()

        # Convert tensor data to 96-byte representation
        gene_state = snapshot.gene_state
        tensor_0 = gene_state["id_0"]  # 4x2x3x2 tensor
        tensor_1 = gene_state["id_1"]  # 4x2x3x2 tensor

        # Flatten tensors to bytes
        # Each tensor is 4x2x3x2 = 48 elements, each element is 1 byte
        # Convert from int8 (-1, 1) to uint8 (0, 255)
        def tensor_to_bytes(tensor):
            # Flatten to 1D array
            flat = tensor.flatten()
            # Convert from int8 to uint8: -1 -> 0, 1 -> 255
            bytes_data = bytearray(48)
            for i, val in enumerate(flat):
                bytes_data[i] = 0 if val == -1 else 255
            return bytes(bytes_data)

        # Combine both tensors: 48 + 48 = 96 bytes
        snapshot_bytes = tensor_to_bytes(tensor_0) + tensor_to_bytes(tensor_1)
        return snapshot_bytes

    def get_state(self) -> Dict[str, Any]:
        """Get complete engine state."""
        return {
            "agent_uuid": self.agent_uuid,
            "session": self.session,
            "governance": self.governance_engine.get_state(),
            "information": self.information_engine.get_state(),
            "inference": self.inference_engine.get_state(),
            "pack_buffer_size": self.current_pack_bytes,
        }

    def close(self):
        """Explicitly close the engine and flush all data to disk."""
        # Write any remaining buffered op-pairs
        if self._pending_cycle_buffer:
            # Write remaining op-pairs as raw data
            for op_pair in self._pending_cycle_buffer:
                self._write_raw_op_pair(op_pair)
            self._pending_cycle_buffer = []

        if hasattr(self, "current_pack_file") and self.current_pack_file:
            # Update header count before closing
            self._update_header_count()
            self.current_pack_file.flush()  # Guarantee header hits disk
            self.current_pack_file.close()
            self.current_pack_file = None

    def __del__(self):
        """Cleanup when the engine is destroyed."""
        # Write any remaining buffered op-pairs
        if hasattr(self, "_pending_cycle_buffer") and self._pending_cycle_buffer:
            for op_pair in self._pending_cycle_buffer:
                self._write_raw_op_pair(op_pair)

        if hasattr(self, "current_pack_file") and self.current_pack_file:
            # Update header count before closing
            self._update_header_count()
            self.current_pack_file.close()

    def _permute_quarter(self, block: bytes, code: int) -> bytes:
        # block is 24 bytes, each either 0 or 255
        b = bytearray(block)
        if code == 1:  # inverse – flip every sign
            for i in range(24):
                b[i] ^= 0xFF  # 0 ↔ 255
        elif code == 2:  # forward – rows 0 and 2
            for row in (0, 2):
                for col in range(6):
                    idx = row * 6 + col
                    b[idx] ^= 0xFF
        elif code == 3:  # backward – rows 1 and 3
            for row in (1, 3):
                for col in range(6):
                    idx = row * 6 + col
                    b[idx] ^= 0xFF
        # code 0: identity – nothing to do
        return bytes(b)

    def _make_keystream(self, snapshot: bytes, key: bytes) -> bytes:
        """
        Generate 48-byte keystream from gene snapshot and agent key.
        Args:
            snapshot: 96-byte gene snapshot
            key: Agent's encryption key (16-32 bytes)
        Returns:
            48-byte keystream for XOR encryption
        """
        if len(snapshot) != 96:
            raise ValueError(f"Snapshot must be 96 bytes, got {len(snapshot)}")
        # Split snapshot into four 24-byte quarters
        Q0 = snapshot[0:24]
        Q1 = snapshot[24:48]
        Q2 = snapshot[48:72]
        Q3 = snapshot[72:96]
        # Pad key to 32 bytes if needed
        padded_key = key + b"\x00" * (32 - len(key))
        # Extract gyration codes from key (4 chunks of 8 bytes each)
        gyration_codes = []
        for i in range(4):
            chunk = padded_key[i * 8 : (i + 1) * 8]
            # Use low 2 bits of first byte for gyration code
            gyration_codes.append(chunk[0] & 0x03)
        # Apply direct permutation to each quarter
        Q0_bytes = self._permute_quarter(Q0, gyration_codes[0])
        Q1_bytes = self._permute_quarter(Q1, gyration_codes[1])
        Q2_bytes = self._permute_quarter(Q2, gyration_codes[2])
        Q3_bytes = self._permute_quarter(Q3, gyration_codes[3])
        # XOR pairs: (Q0⊕Q1) || (Q2⊕Q3) → 48 bytes
        keystream = bytearray(48)
        for i in range(24):
            keystream[i] = Q0_bytes[i] ^ Q1_bytes[i]
            keystream[i + 24] = Q2_bytes[i] ^ Q3_bytes[i]
        return bytes(keystream)

    def _write_encrypted_cycle_data(self, cycle_data: bytes):
        """Write encrypted cycle data to the pack file."""
        # Check if current pack would exceed pack_size (excluding header)
        header_size = (
            40 if (self._encryption_enabled and self._gyrocrypt_key) else 13
        )  # GyroCrypt vs old format
        payload_size = self.current_pack_bytes + len(cycle_data)

        if payload_size >= (self.pack_size - header_size):
            # Current pack would exceed size limit, start new one
            self._open_current_pack()

        # Ensure file is open
        if self.current_pack_file is None:
            self._open_current_pack()

        # At this point, file is definitely open
        assert self.current_pack_file is not None

        # Write the encrypted cycle data
        self.current_pack_file.write(cycle_data)

        self.current_pack_bytes += len(cycle_data)
        self._header_update_counter += len(cycle_data)

        # Update header periodically
        if self._header_update_counter >= 256:
            self._update_header_count()
            self._header_update_counter = 0


# System-level functions
def initialize_system():
    """
    Create the S2 directory structure if missing and verify seed files.
    Returns paths and version info.
    """
    base_path = "s2_information"

    # Create base directories
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, "agency"), exist_ok=True)

    # Create agency subdirectories
    for subdir in ["g1_information", "g2_information", "g4_information", "g5_information"]:
        os.makedirs(os.path.join(base_path, "agency", subdir), exist_ok=True)

    # Create shard directories
    for subdir in ["g1_information", "g4_information", "g5_information"]:
        for i in range(256):
            shard = f"{i:02x}"
            os.makedirs(os.path.join(base_path, "agency", subdir, shard), exist_ok=True)

    # Create agents directory
    os.makedirs(os.path.join(base_path, "agents"), exist_ok=True)
    for i in range(256):
        shard = f"{i:02x}"
        os.makedirs(os.path.join(base_path, "agents", shard), exist_ok=True)

    # Write manifest
    manifest = {"version": "1.0", "pack_size": 65536, "shard_prefix_length": 2}

    manifest_path = os.path.join(base_path, "s2_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Generate epigenome if not exists
    epigenome_path = os.path.join(base_path, "agency", "g2_information", "g2_information.dat")
    if not os.path.exists(epigenome_path):
        import s1_governance

        s1_governance.build_epigenome_projection(epigenome_path)

    return {
        "base_path": base_path,
        "manifest_path": manifest_path,
        "epigenome_path": epigenome_path,
        "version": manifest["version"],
    }


def create_agent(agent_uuid: Optional[str] = None) -> str:
    """
    Create a new agent with empty curriculum and session files.
    Returns the agent UUID.
    """
    agent_id = agent_uuid or str(uuid.uuid4())
    shard = get_shard_from_uuid(agent_id)

    # Create agent directories
    base_path = "s2_information"
    agent_dir = os.path.join(base_path, "agents", shard, agent_id)
    os.makedirs(os.path.join(agent_dir, "g4_information"), exist_ok=True)
    os.makedirs(os.path.join(agent_dir, "g5_information"), exist_ok=True)

    # Create empty curriculum
    curriculum = {
        "version": "1.0",
        "patterns": {},
        "byte_to_token": {},
        "token_to_byte": {},
        "metadata": {"created": datetime.utcnow().isoformat()},
    }

    with open(os.path.join(agent_dir, "g4_information", "curriculum.json"), "w") as f:
        json.dump(curriculum, f, indent=2)

    # Create empty session
    session = {
        "agent_uuid": agent_id,
        "created": datetime.utcnow().isoformat(),
        "last_checkpoint": None,
        "phase": 0,
        "cycle_count": 0,
        "active_curriculum": None,
        "compression_metadata": [],  # Store compression details here
    }

    with open(os.path.join(agent_dir, "g5_information", "session.json"), "w") as f:
        json.dump(session, f, indent=2)

    return agent_id


def load_epigenome_tensor(
    path: str = "s2_information/agency/g2_information/g2_information.dat",
) -> np.ndarray:
    """
    Load the epigenome projection table into memory.
    """
    with open(path, "rb") as f:
        # Skip 32-byte SHA-256 header
        f.read(32)
        # Read 48x256 table
        data = f.read(48 * 256)
        return np.frombuffer(data, dtype=np.uint8).reshape(48, 256)


def set_active_agent(agent_uuid: str) -> IntelligenceEngine:
    """
    Create and return an IntelligenceEngine for the specified agent.
    """
    return IntelligenceEngine(agent_uuid=agent_uuid)


def process_stream(data_bytes: bytes, agent_uuid: Optional[str] = None) -> Dict[str, List[Any]]:
    """
    Process a byte stream through GyroSI.

    Args:
        data_bytes: Raw bytes to process
        agent_uuid: Optional agent UUID

    Returns:
        Dictionary of all emitted artifacts
    """
    engine = IntelligenceEngine(agent_uuid)
    try:
        return engine.process_stream(data_bytes)
    finally:
        engine.close()


def generate_text(prompt: str = "", max_length: int = 100, agent_uuid: Optional[str] = None) -> str:
    """
    Generate text using the GyroSI Baby LM.

    Args:
        prompt: Initial text to seed the generation
        max_length: Maximum number of characters to generate
        agent_uuid: Optional agent UUID

    Returns:
        Generated text
    """
    engine = IntelligenceEngine(agent_uuid)
    try:
        # Process the prompt
        if prompt:
            engine.process_stream(prompt.encode("utf-8"))

        # Generate bytes
        generated_bytes = engine.generate(b"", max_length)

        # Try to decode as UTF-8, falling back to latin-1 if needed
        try:
            return generated_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return generated_bytes.decode("latin-1")
    finally:
        engine.close()
