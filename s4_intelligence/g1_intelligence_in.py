"""
g1_intelligence_in.py - Intelligence Engine API

The sole controller of S3 and the only module permitted to write into S2.
Exposes exactly one high-level API: process_stream.

Device logic: All tensors are created on the selected device (GPU if available, else CPU).
"""

import os
import json
import uuid
import hashlib
import struct
import fcntl
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, BinaryIO, Union, Set
from pathlib import Path
import torch
# Select device for all tensors and models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np

# Import from S3 inference modules
from s3_inference.g1_inference import GovernanceEngine, AcceptedOpPair, CycleComplete
from s3_inference.g2_inference import InformationEngine, ResonanceEvent
from s3_inference.g3_inference import (
    InferenceEngine,
    CompressedBlock,
    PatternPromotion,
)

# Import from S1 governance
from s1_governance import (
    get_gene_constant,
    get_gene_anchor,
    build_epigenome_projection,
    get_gene_tensors,
)

# Import from S4 gyration primitives
from s4_intelligence.g2_intelligence_eg import (
    byte_to_gyrations,
    gyrations_to_byte,
    gyration_op,
)


# Utility for JSON serialization of state
def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return str(obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


def get_shard_from_uuid(uuid_str: str) -> str:
    """
    Extract shard prefix from UUID string.

    Args:
        uuid_str: UUID string

    Returns:
        First two characters of UUID (shard identifier)
    """
    clean_uuid = uuid_str.replace("-", "")
    return clean_uuid[:2].lower()


class IntelligenceEngine:
    """
    Main orchestration engine that coordinates S3 processing and S2 persistence.
    This is the only class that writes to S2 storage and the sole controller of
    the three S3 inference engines.

    This class manages:
    1. Agent lifecycle and state persistence
    2. Stream processing through S3 engines
    3. File I/O for genome packs and curriculum
    4. GyroCrypt encryption/decryption
    5. Pattern learning and curriculum management
    """

    # Class constants
    PACK_HEADER_SIZE = 40  # 32B gene anchor + 4B cycle index + 4B salt
    DEFAULT_PACK_SIZE = 65536  # 64KB
    MAX_CYCLE_INDEX = 0xFFFFFFFF  # 32-bit max

    def __init__(
        self,
        agent_uuid: Optional[str] = None,
        base_path: Union[str, Path] = "s2_information",
        encryption_enabled: bool = True,
    ):
        """
        Initialize the Intelligence Engine.

        Args:
            agent_uuid: Agent UUID (generated if None)
            base_path: Base path for S2 information storage
            encryption_enabled: Whether to enable GyroCrypt encryption
        """
        # Core identity
        self.agent_uuid = agent_uuid or str(uuid.uuid4())
        self.shard = get_shard_from_uuid(self.agent_uuid)
        self.base_path = Path(base_path)

        # Encryption settings
        self._encryption_enabled = encryption_enabled
        self._gyrocrypt_key: Optional[bytes] = None
        self._cycle_index = 0

        # Initialize storage structure
        self._ensure_directories()

        # Read manifest for configuration
        self.pack_size = self._load_manifest_config()

        # Load state and curriculum
        self._load_agent_state()
        self._load_curriculum()

        # Initialize S3 engines
        self._initialize_engines()

        # File handling
        self.current_pack_file: Optional[BinaryIO] = None
        self.current_pack_uuid: Optional[str] = None
        self.current_pack_bytes = 0
        self._pending_cycle_buffer: List[Tuple[int, int]] = []
        self._current_resonance_flags: List[bool] = []

        # Initialize the first pack file
        self._open_current_pack()

    def _ensure_directories(self) -> None:
        """
        Create all required directories for the agent.
        """
        # Global directories
        (self.base_path / "agency" / "g1_information" / self.shard).mkdir(
            parents=True, exist_ok=True
        )
        (self.base_path / "agency" / "g2_information").mkdir(parents=True, exist_ok=True)
        (self.base_path / "agency" / "g4_information" / self.shard).mkdir(
            parents=True, exist_ok=True
        )
        (self.base_path / "agency" / "g5_information" / self.shard).mkdir(
            parents=True, exist_ok=True
        )

        # Agent-specific directories
        (self.base_path / "agents" / self.shard / self.agent_uuid / "g4_information").mkdir(
            parents=True, exist_ok=True
        )
        (self.base_path / "agents" / self.shard / self.agent_uuid / "g5_information").mkdir(
            parents=True, exist_ok=True
        )

    def _load_manifest_config(self) -> int:
        """
        Load configuration from manifest.json.

        Returns:
            Pack size from manifest or default (65536)
        """
        manifest_path = self.base_path / "s2_manifest.json"

        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                return manifest.get("pack_size", self.DEFAULT_PACK_SIZE)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading manifest: {e}. Using default pack size.")

        return self.DEFAULT_PACK_SIZE

    def _load_agent_state(self) -> None:
        """
        Load the agent's session state from disk.
        Initializes with defaults if not found.
        """
        session_path = (
            self.base_path
            / "agents"
            / self.shard
            / self.agent_uuid
            / "g5_information"
            / "session.json"
        )

        # Default initial state
        self.session = {
            "agent_uuid": self.agent_uuid,
            "created": datetime.utcnow().isoformat(),
            "last_checkpoint": None,
            "phase": 0,
            "cycle_count": 0,
            "active_curriculum": None,
        }

        # Load existing state if available
        if session_path.exists():
            try:
                with open(session_path, "r") as f:
                    loaded_session = json.load(f)
                    self.session.update(loaded_session)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading session state: {e}. Using defaults.")

        # Initialize GyroCrypt
        if self._encryption_enabled:
            if "gyrocrypt_key" in self.session:
                # Use existing key
                self._gyrocrypt_key = bytes.fromhex(self.session["gyrocrypt_key"])
            else:
                # Generate new key
                import secrets

                self._gyrocrypt_key = secrets.token_bytes(32)
                self.session["gyrocrypt_key"] = self._gyrocrypt_key.hex()

            # Load cycle index
            self._cycle_index = self.session.get("cycle_index", 0)

    def _load_curriculum(self) -> None:
        """
        Load the agent's curriculum (learned patterns and token mappings).
        """
        curriculum_path = (
            self.base_path
            / "agents"
            / self.shard
            / self.agent_uuid
            / "g4_information"
            / "curriculum.json"
        )

        # Default empty curriculum
        self.curriculum = {
            "version": "1.0",
            "patterns": {},
            "byte_to_token": {},
            "token_to_byte": {},
            "metadata": {"created": datetime.utcnow().isoformat()},
        }

        # Load existing curriculum if available
        if curriculum_path.exists():
            try:
                with open(curriculum_path, "r") as f:
                    loaded_curriculum = json.load(f)
                    self.curriculum.update(loaded_curriculum)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading curriculum: {e}. Using defaults.")

    def _initialize_engines(self) -> None:
        """
        Initialize the three S3 inference engines.
        """
        # Ensure epigenome projection exists
        epigenome_path = self.base_path / "agency" / "g2_information" / "g2_information.dat"
        if not epigenome_path.exists():
            build_epigenome_projection(str(epigenome_path))

        # Create inference engines
        self.governance_engine = GovernanceEngine()
        self.information_engine = InformationEngine(str(epigenome_path))
        self.inference_engine = InferenceEngine(agent_uuid=self.agent_uuid)

        # Set initial phase from session if available
        if "phase" in self.session:
            self.governance_engine.phase = self.session["phase"]

        # Set initial cycle count from session if available
        if "cycle_count" in self.session:
            self.governance_engine.cycle_count = self.session["cycle_count"]

    def process_stream(self, data_bytes: bytes) -> Dict[str, List[Any]]:
        """
        Process a stream of bytes through the full GyroSI pipeline.

        This is the main entry point for the Intelligence Engine and the
        only public API that processes data.

        Args:
            data_bytes: Raw bytes to process

        Returns:
            Dictionary containing all artifacts from processing:
            - accepted_ops: List of AcceptedOpPair events
            - resonances: List of ResonanceEvent events
            - compressed_blocks: List of CompressedBlock events
            - pattern_promotions: List of PatternPromotion events
        """
        if not isinstance(data_bytes, bytes):
            raise TypeError(f"Expected bytes, got {type(data_bytes)}")

        # Initialize collection of artifacts
        artifacts = {
            "accepted_ops": [],
            "resonances": [],
            "compressed_blocks": [],
            "pattern_promotions": [],
        }
        try:
            # Process each byte in the stream
            for byte_val in data_bytes:
                # Convert byte to two op-pairs
                op_pair1, op_pair2 = byte_to_gyrations(byte_val)

                # Process each op-pair through the pipeline
                for op_pair in [op_pair1, op_pair2]:
                    # 1. Information Engine: determine resonance
                    info_event = self.information_engine.process_accepted_op_pair(
                        phase=self.governance_engine.phase, op_pair=op_pair, byte_val=byte_val
                    )
                    artifacts["resonances"].append(info_event)

                    # 2. Governance Engine: accept and advance phase
                    gov_events = self.governance_engine.process_op_pair(
                        op_pair, info_event.resonance_flag
                    )

                    # Store resonance flag for cycle analysis
                    self._current_resonance_flags.append(info_event.resonance_flag)

                    # Process governance events
                    for event in gov_events:
                        if isinstance(event, AcceptedOpPair):
                            # Record accepted op-pair
                            artifacts["accepted_ops"].append(event)

                            # Buffer for cycle writing
                            self._pending_cycle_buffer.append(event.op_pair)

                        elif isinstance(event, CycleComplete):
                            # 3. Inference Engine: analyze completed cycle
                            inference_events = self.inference_engine.process_cycle_complete(
                                event.op_pairs, event.resonance_flags
                            )

                            cycle_was_written = False
                            # Process inference events
                            for inf_event in inference_events:
                                if isinstance(inf_event, CompressedBlock):
                                    artifacts["compressed_blocks"].append(inf_event)
                                    if self._handle_compressed_block(inf_event):
                                        cycle_was_written = True

                                elif isinstance(inf_event, PatternPromotion):
                                    artifacts["pattern_promotions"].append(inf_event)
                                    self._persist_pattern_promotion(inf_event)

                            # If the inference engine didn't explicitly handle the cycle by writing it, write it now.
                            if not cycle_was_written:
                                self._write_cycle(event.op_pairs)
                                cycle_was_written = True

                            # Reset for next cycle only if it was persisted.
                            if cycle_was_written:
                                self._current_resonance_flags = []
                                self._pending_cycle_buffer = []

            # Update session with latest state
            self._update_session()

            return artifacts

        except Exception as e:
            print(f"Error in process_stream: {e}")
            # Try to save state even on error
            try:
                self._update_session()
            except:
                pass
            raise

    def _handle_compressed_block(self, block: CompressedBlock) -> bool:
        """
        Process a compressed block from the inference engine.

        Args:
            block: CompressedBlock event from InferenceEngine

        Returns:
            True if the cycle was written to disk, False otherwise.
        """
        if block.block_type == "full_cycle":
            # Write the full cycle to disk
            self._write_cycle(block.data["ops"])
            return True

        elif block.block_type == "cycle_repeat":
            # Log the repeated cycle (actual writing handled by InferenceEngine)
            hash_val = block.data.get("hash", "unknown")
            count = block.data.get("count", 0)

            # Record compressed blocks in session for analytics
            if "compression_stats" not in self.session:
                self.session["compression_stats"] = {}

            self.session["compression_stats"][hash_val] = {
                "count": count,
                "last_seen": datetime.utcnow().isoformat(),
            }

            # Increment cycle index for encryption
            self._cycle_index += 1

            # Flush to persist the updated cycle index in the header.
            self._flush_pack_file()

        elif block.block_type == "pruned_cycle":
            # Log pruned cycles for analytics
            if "pruned_cycles" not in self.session:
                self.session["pruned_cycles"] = 0

            self.session["pruned_cycles"] += 1

        return False

    def _persist_pattern_promotion(self, promotion: PatternPromotion) -> None:
        """
        Add a promoted pattern to the curriculum and persist it.

        Args:
            promotion: PatternPromotion event from InferenceEngine
        """
        # Convert pattern to serializable format
        serialized_pattern = []
        for op_code, tensor_id in promotion.pattern:
            serialized_pattern.append({"op": op_code, "tensor": tensor_id})

        # Add to curriculum
        self.curriculum["patterns"][promotion.pattern_hash] = {
            "sequence": serialized_pattern,
            "frequency": promotion.frequency,
            "created": datetime.utcnow().isoformat(),
            "length": len(promotion.pattern),
        }

        # Save curriculum
        self._persist_curriculum()

        # Also save to global curriculum for sharing
        self._persist_global_curriculum(promotion)

    def _persist_curriculum(self) -> None:
        """
        Save the agent's curriculum to disk.
        """
        curriculum_path = (
            self.base_path
            / "agents"
            / self.shard
            / self.agent_uuid
            / "g4_information"
            / "curriculum.json"
        )

        # Write with atomic replacement
        temp_path = curriculum_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(self.curriculum, f, indent=2)

            # Use os.replace for atomic operation
            os.replace(temp_path, curriculum_path)
        except Exception as e:
            print(f"Error saving curriculum: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def _persist_global_curriculum(self, promotion: PatternPromotion) -> None:
        """
        Add promoted pattern to global curriculum for sharing across agents.

        Args:
            promotion: PatternPromotion event from InferenceEngine
        """
        # Create a new curriculum snapshot with UUID
        curriculum_uuid = str(uuid.uuid4())
        curriculum_path = (
            self.base_path
            / "agency"
            / "g4_information"
            / self.shard
            / f"{curriculum_uuid}-curriculum.json"
        )

        # Create a minimal shared curriculum with just this pattern
        shared_curriculum = {
            "version": "1.0",
            "source_agent": self.agent_uuid,
            "created": datetime.utcnow().isoformat(),
            "patterns": {
                promotion.pattern_hash: {
                    "sequence": [{"op": op, "tensor": tid} for op, tid in promotion.pattern],
                    "frequency": promotion.frequency,
                    "created": datetime.utcnow().isoformat(),
                    "length": len(promotion.pattern),
                }
            },
        }

        # Write the global curriculum
        try:
            with open(curriculum_path, "w") as f:
                json.dump(shared_curriculum, f, indent=2)
        except Exception as e:
            print(f"Error saving global curriculum: {e}")

    def _update_session(self) -> None:
        """
        Update the session file with current state.
        """
        # Update session with engine states
        gov_state = self.governance_engine.get_state()
        info_state = self.information_engine.get_state()
        inf_state = self.inference_engine.get_state()

        # Extract before sanitization
        phase = gov_state["phase"]
        cycle_count = gov_state["cycle_count"]

        gov_state = sanitize_for_json(gov_state)
        info_state = sanitize_for_json(info_state)
        inf_state = sanitize_for_json(inf_state)

        self.session.update(
            {
                "phase": phase,
                "cycle_count": cycle_count,
                "last_checkpoint": datetime.utcnow().isoformat(),
                "governance": gov_state,
                "information": info_state,
                "inference": inf_state,
            }
        )

        # Add encryption data if enabled
        if self._encryption_enabled and self._gyrocrypt_key:
            self.session["gyrocrypt_key"] = self._gyrocrypt_key.hex()
            self.session["cycle_index"] = self._cycle_index

        # Write the session file atomically
        session_path = Path(
            self.base_path, "agents", self.shard, self.agent_uuid, "g5_information", "session.json"
        )

        # Ensure directory exists
        session_path.parent.mkdir(parents=True, exist_ok=True)

        # Use a temporary file and rename approach
        temp_path = session_path.with_suffix(".tmp")
        backup_path = session_path.with_suffix(".bak")

        try:
            # Write to temporary file first
            with open(temp_path, "w") as f:
                json.dump(self.session, f, indent=2)

            # Backup existing if present
            if session_path.exists():
                try:
                    if backup_path.exists():
                        backup_path.unlink()
                    os.replace(session_path, backup_path)
                except:
                    pass  # Backup is optional

            # Move temporary to main
            os.replace(temp_path, session_path)

        except Exception as e:
            print(f"Error updating session: {e}")
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except:
                    pass

    def _open_current_pack(self) -> None:
        """
        Open a new genome pack file for writing.
        Closes the current pack if one is open.
        """
        # Close current pack if open
        if self.current_pack_file and not self.current_pack_file.closed:
            self._flush_pack_file()
            self.current_pack_file.close()

        # Generate new UUID for this pack
        self.current_pack_uuid = str(uuid.uuid4())

        # Create pack path
        pack_dir = self.base_path / "agency" / "g1_information" / self.shard
        pack_path = pack_dir / f"{self.current_pack_uuid}-genome.dat"

        try:
            # Open new pack file
            self.current_pack_file = open(pack_path, "wb")

            # Write header
            if self._encryption_enabled and self._gyrocrypt_key:
                # GyroCrypt header (32B anchor + 4B cycle_index + 4B salt)
                gene_anchor = get_gene_anchor()
                gene_snapshot = self.get_current_cycle_decoded()

                self.current_pack_file.write(gene_anchor)  # 32B
                self.current_pack_file.write(struct.pack("<I", self._cycle_index))  # 4B
                self.current_pack_file.write(gene_snapshot[:4])  # 4B salt
            else:
                # Simple header for unencrypted packs
                version = 1
                timestamp = int(datetime.utcnow().timestamp())
                op_count = 0

                self.current_pack_file.write(struct.pack("B", version))  # 1B
                self.current_pack_file.write(struct.pack(">Q", timestamp))  # 8B
                self.current_pack_file.write(struct.pack(">I", op_count))  # 4B

            # Acquire exclusive lock
            fcntl.flock(self.current_pack_file.fileno(), fcntl.LOCK_EX)

            # Reset byte counter
            self.current_pack_bytes = 0

        except IOError as e:
            print(f"Error opening pack file: {e}")
            self.current_pack_file = None
            raise

    def _flush_pack_file(self) -> None:
        """
        Flush current pack file to disk and update header.
        """
        if not self.current_pack_file or self.current_pack_file.closed:
            return

        try:
            # Update header based on format
            if self._encryption_enabled and self._gyrocrypt_key:
                # Update cycle index in header
                self.current_pack_file.seek(32)  # Skip anchor
                self.current_pack_file.write(struct.pack("<I", self._cycle_index))
            else:
                # Update op count in header
                self.current_pack_file.seek(9)  # Skip version and timestamp
                self.current_pack_file.write(struct.pack(">I", self.current_pack_bytes))

            # Force flush to disk
            self.current_pack_file.flush()
            os.fsync(self.current_pack_file.fileno())

            # Return to end of file for further writing
            self.current_pack_file.seek(0, 2)  # Seek to end

        except IOError as e:
            print(f"Error flushing pack file: {e}")

    def _write_cycle(self, op_pairs: List[Tuple[int, int]]) -> None:
        """
        Write a complete cycle to the pack file.

        Args:
            op_pairs: List of 48 op-pairs
        """
        if len(op_pairs) != 48:
            raise ValueError(f"Cycle must have exactly 48 op-pairs, got {len(op_pairs)}")

        # Check if we need a new pack file
        header_size = (
            self.PACK_HEADER_SIZE if (self._encryption_enabled and self._gyrocrypt_key) else 13
        )
        cycle_size = 48  # 48 op-pairs, 1 byte each

        if (self.current_pack_bytes + cycle_size) > (self.pack_size - header_size):
            self._open_current_pack()

        # Ensure pack file is open
        if not self.current_pack_file or self.current_pack_file.closed:
            self._open_current_pack()
        if not self.current_pack_file:
            raise RuntimeError("Failed to open pack file for writing.")

        # Format the cycle data
        cycle_data = bytearray(48)
        for i, (op_code, tensor_id) in enumerate(op_pairs):
            # Pack each op-pair as a single byte
            nibble = ((op_code & 0x7) << 1) | (tensor_id & 0x1)
            cycle_data[i] = nibble

        # Apply encryption if enabled
        if self._encryption_enabled and self._gyrocrypt_key:
            # Generate keystream
            gene_snapshot = self.get_current_cycle_decoded()
            keystream = self._make_keystream(gene_snapshot, self._gyrocrypt_key)

            # XOR encrypt the cycle data
            for i in range(len(cycle_data)):
                cycle_data[i] ^= keystream[i % len(keystream)]

        # Write the cycle data
        try:
            self.current_pack_file.write(cycle_data)
            self.current_pack_bytes += len(cycle_data)

            # Increment cycle index for next encryption
            self._cycle_index += 1
            if self._cycle_index >= self.MAX_CYCLE_INDEX:
                # Handle cycle index wrap
                print("WARNING: Cycle index wrapped around")
                self._cycle_index = 0

            # Flush after every cycle to ensure data durability.
            self._flush_pack_file()

        except IOError as e:
            print(f"Error writing cycle: {e}")
            self._open_current_pack()
            if not self.current_pack_file:
                raise RuntimeError("Failed to open pack file for writing after IOError.")
            # Retry write
            self.current_pack_file.write(cycle_data)
            self.current_pack_bytes += len(cycle_data)

    def _make_keystream(self, snapshot: bytes, key: bytes) -> bytes:
        """
        Generate encryption keystream from gene snapshot and key.

        Args:
            snapshot: 96-byte gene snapshot
            key: Encryption key

        Returns:
            48-byte keystream for cycle encryption
        """
        if len(snapshot) != 96:
            # Pad or truncate to 96 bytes
            if len(snapshot) < 96:
                snapshot = snapshot + b"\x00" * (96 - len(snapshot))
            else:
                snapshot = snapshot[:96]

        # Ensure key is at least 32 bytes
        padded_key = key + b"\x00" * (32 - len(key))

        # Split snapshot into four 24-byte quarters
        quarters = []
        for i in range(0, 96, 24):
            quarters.append(snapshot[i : i + 24])

        # Get gyration codes from key (first byte of each 8-byte chunk)
        gyration_codes = [padded_key[i * 8] & 0x3 for i in range(4)]

        # Apply permutations to each quarter based on gyration code
        permuted_quarters = []
        for i, quarter in enumerate(quarters):
            code = gyration_codes[i]
            q_array = bytearray(quarter)

            if code == 1:  # Left Inverse - global sign flip
                for j in range(len(q_array)):
                    q_array[j] ^= 0xFF
            elif code == 2:  # Forward Gyration - flip rows 0,2
                for row in [0, 2]:
                    for col in range(6):
                        idx = row * 6 + col
                        if idx < len(q_array):
                            q_array[idx] ^= 0xFF
            elif code == 3:  # Backward Gyration - flip rows 1,3
                for row in [1, 3]:
                    for col in range(6):
                        idx = row * 6 + col
                        if idx < len(q_array):
                            q_array[idx] ^= 0xFF

            permuted_quarters.append(bytes(q_array))

        # XOR pairs to create keystream
        keystream = bytearray(48)
        for i in range(24):
            keystream[i] = permuted_quarters[0][i] ^ permuted_quarters[1][i]
            keystream[i + 24] = permuted_quarters[2][i] ^ permuted_quarters[3][i]

        return bytes(keystream)

    def get_current_cycle_decoded(self) -> bytes:
        """
        Get a full 96-byte snapshot for encryption by flattening the S1 gene tensors.

        This is the definitive, deterministic snapshot required for reproducible
        encryption. It is based *only* on the immutable gene constants.

        Returns:
            96-byte snapshot suitable for keystream generation.
        """
        # Get the immutable gene tensors from S1
        gene_tensors = get_gene_tensors()

        # Ensure the tensors are in the expected format
        id_0 = gene_tensors["id_0"].numpy().tobytes()
        id_1 = gene_tensors["id_1"].numpy().tobytes()

        # Each tensor is 4x2x3x2 int8 = 48 bytes. Concatenated, they are 96 bytes.
        if len(id_0) != 48 or len(id_1) != 48:
            raise ValueError(f"Unexpected tensor size. id_0: {len(id_0)}, id_1: {len(id_1)}")

        return id_0 + id_1

    def generate(self, prompt: bytes = b"", max_length: int = 100) -> bytes:
        """
        Generate new bytes based on learned patterns.

        Args:
            prompt: Initial bytes to process before generating
            max_length: Maximum number of bytes to generate

        Returns:
            Generated byte sequence
        """
        # Process prompt if provided
        if prompt:
            self.process_stream(prompt)

        # Generate new bytes
        generated = bytearray()
        for _ in range(max_length):
            # Predict two op-pairs
            op_pair1 = self.inference_engine.predict_next_operation(self.curriculum)
            op_pair2 = self.inference_engine.predict_next_operation(self.curriculum)

            # Convert to byte
            next_byte = gyrations_to_byte(op_pair1, op_pair2)
            generated.append(next_byte)

            # Process the generated byte to update state
            self.process_stream(bytes([next_byte]))

        return bytes(generated)

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the engine.

        Returns:
            Dictionary with engine state
        """
        return {
            "agent_uuid": self.agent_uuid,
            "governance": self.governance_engine.get_state(),
            "information": self.information_engine.get_state(),
            "inference": self.inference_engine.get_state(),
            "encryption_enabled": self._encryption_enabled,
            "cycle_index": self._cycle_index,
            "curriculum_size": len(self.curriculum["patterns"]),
            "pack_bytes": self.current_pack_bytes,
        }

    def learn_token_mapping(self, tokens_bytes: Dict[str, bytes]) -> None:
        """
        Learn mappings between tokens and byte sequences.

        Args:
            tokens_bytes: Dictionary mapping token strings to byte sequences
        """
        for token, byte_seq in tokens_bytes.items():
            # Store byte sequence as list of integers
            byte_list = list(byte_seq)

            # Update curriculum dictionaries
            byte_key = str(byte_list)
            self.curriculum["byte_to_token"][byte_key] = token
            self.curriculum["token_to_byte"][token] = byte_list

        # Save updated curriculum
        self._persist_curriculum()

    def close(self) -> None:
        """
        Close the engine and flush all data to disk.
        """
        try:
            # Update session state
            self._update_session()

            # Flush and close pack file
            if self.current_pack_file and not self.current_pack_file.closed:
                self._flush_pack_file()
                fcntl.flock(self.current_pack_file.fileno(), fcntl.LOCK_UN)
                self.current_pack_file.close()
                self.current_pack_file = None
        except Exception as e:
            print(f"Error closing engine: {e}")

    def __del__(self) -> None:
        """
        Ensure cleanup on garbage collection.
        """
        self.close()


# System-level functions
def initialize_system() -> Dict[str, Any]:
    """
    Create the S2 directory structure and initialize the system.

    Returns:
        Dictionary with system information
    """
    base_path = Path("s2_information")

    # Create base directories
    base_path.mkdir(exist_ok=True)
    (base_path / "agency").mkdir(exist_ok=True)

    # Create agency subdirectories
    for subdir in ["g1_information", "g2_information", "g4_information", "g5_information"]:
        (base_path / "agency" / subdir).mkdir(exist_ok=True)

    # Create shard directories
    for subdir in ["g1_information", "g4_information", "g5_information"]:
        for i in range(16):
            for j in range(16):
                shard = f"{i:x}{j:x}"
                (base_path / "agency" / subdir / shard).mkdir(exist_ok=True)

    # Create agents directory
    (base_path / "agents").mkdir(exist_ok=True)
    for i in range(16):
        for j in range(16):
            shard = f"{i:x}{j:x}"
            (base_path / "agents" / shard).mkdir(exist_ok=True)

    # Create manifest
    manifest = {
        "version": "1.0",
        "pack_size": 65536,
        "shard_prefix_length": 2,
        "initialized": datetime.utcnow().isoformat(),
    }

    manifest_path = base_path / "s2_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Generate epigenome if not exists
    epigenome_path = base_path / "agency" / "g2_information" / "g2_information.dat"
    if not epigenome_path.exists():
        build_epigenome_projection(str(epigenome_path))

    return {
        "base_path": str(base_path),
        "manifest_path": str(manifest_path),
        "epigenome_path": str(epigenome_path),
        "version": manifest["version"],
        "status": "initialized",
    }


def create_agent(agent_uuid: Optional[str] = None) -> str:
    """
    Create a new agent with empty curriculum and session files.

    Args:
        agent_uuid: Optional UUID for the agent

    Returns:
        Agent UUID
    """
    # Generate UUID if not provided
    agent_id = agent_uuid or str(uuid.uuid4())
    shard = get_shard_from_uuid(agent_id)

    # Base path
    base_path = Path("s2_information")

    # Create agent directories
    agent_dir = base_path / "agents" / shard / agent_id
    (agent_dir / "g4_information").mkdir(parents=True, exist_ok=True)
    (agent_dir / "g5_information").mkdir(parents=True, exist_ok=True)

    # Create empty curriculum
    curriculum = {
        "version": "1.0",
        "patterns": {},
        "byte_to_token": {},
        "token_to_byte": {},
        "metadata": {"created": datetime.utcnow().isoformat()},
    }

    with open(agent_dir / "g4_information" / "curriculum.json", "w") as f:
        json.dump(curriculum, f, indent=2)

    # Create empty session
    session = {
        "agent_uuid": agent_id,
        "created": datetime.utcnow().isoformat(),
        "last_checkpoint": None,
        "phase": 0,
        "cycle_count": 0,
        "active_curriculum": None,
    }

    with open(agent_dir / "g5_information" / "session.json", "w") as f:
        json.dump(session, f, indent=2)

    return agent_id


def load_epigenome_tensor(
    path: str = "s2_information/agency/g2_information/g2_information.dat",
) -> np.ndarray:
    """
    Load the epigenome projection table into memory.

    Args:
        path: Path to the epigenome projection file

    Returns:
        48x256 numpy array of epigenome values
    """
    try:
        with open(path, "rb") as f:
            # Skip 32-byte SHA-256 header
            header = f.read(32)
            if len(header) != 32:
                raise ValueError(f"Invalid epigenome header size: {len(header)}")

            # Read 48x256 table
            data = f.read(48 * 256)
            if len(data) != 48 * 256:
                raise ValueError(f"Invalid epigenome data size: {len(data)}")

            return np.frombuffer(data, dtype=np.uint8).reshape(48, 256)
    except Exception as e:
        print(f"Error loading epigenome: {e}")
        # Return empty epigenome as fallback
        return np.zeros((48, 256), dtype=np.uint8)


def set_active_agent(agent_uuid: str) -> IntelligenceEngine:
    """
    Create and return an IntelligenceEngine for the specified agent.

    Args:
        agent_uuid: UUID of the agent to activate

    Returns:
        Initialized IntelligenceEngine
    """
    return IntelligenceEngine(agent_uuid=agent_uuid)


def process_stream(data_bytes: bytes, agent_uuid: Optional[str] = None) -> Dict[str, List[Any]]:
    """
    Process a byte stream through the GyroSI pipeline.

    This is a convenience function that creates a temporary engine,
    processes the data, and then closes the engine.

    Args:
        data_bytes: Raw bytes to process
        agent_uuid: Optional agent UUID

    Returns:
        Dictionary of all emitted artifacts
    """
    engine = IntelligenceEngine(agent_uuid=agent_uuid)
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
    engine = IntelligenceEngine(agent_uuid=agent_uuid)
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
