"""
g1_intelligence_in.py - Intelligence Engine API

The sole controller of S3 and the only module permitted to write into S2.
Exposes exactly one high-level API: process_stream.

Device logic: All tensors are created on the selected device (GPU if available, else CPU).
"""

import os
import json
import uuid
import struct
import fcntl
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, BinaryIO, Union, cast
from pathlib import Path

# Import from S3 inference modules
from s3_inference.g1_inference import GovernanceEngine, AcceptedOpPair, CycleComplete
from s3_inference.g2_inference import InformationEngine
from s3_inference.g3_inference import (
    InferenceEngine,
    CompressedBlock,
    PatternPromotion,
)

# Import from S1 governance
from s1_governance import (
    get_gene_tensors,
    byte_to_gyrations,
    gyrations_to_byte,
    build_epigenome_projection,
    VOID_OP_PAIR,
)

# === Canonical GenomePack format constants ===
GENOME_PACK_MAGIC = b"GYRO"
GENOME_PACK_VERSION = 2
GENOME_PACK_HEADER_SIZE = 128
GENOME_PACK_FLAGS = 0  # always 0 for cleartext genome
GENOME_PACK_SNAPSHOT_SIZE = 96
GENOME_PACK_SALT_SIZE = 12
GENOME_PACK_RESERVED_SIZE = 8
GENOME_CYCLE_SIZE = 24  # 24 bytes per cycle

# Default pack size set to 64MB
DEFAULT_PACK_SIZE = 64 * 1024 * 1024  # 64MB


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


class GenomePack:
    """
    Canonical abstraction for genome pack files (cleartext, self-indexing, 128B header).
    Handles all header logic, file naming, and cycle read/write.
    """

    def __init__(self, path: Path, start_cycle_index: int, mode: str = "rb+"):
        self.path = Path(path)
        self.start_cycle_index = start_cycle_index
        self.file: Optional[BinaryIO] = None
        self.mode = mode
        self.header = None
        self.cycles_written = 0
        self._open()

    @property
    def closed(self) -> bool:
        return self.file is None

    @classmethod
    def make_path(cls, base_path: Path, shard: str, start_cycle_index: int) -> Path:
        # Deterministic, sortable, agent-agnostic file name
        return (
            base_path / "agency" / "g1_information" / shard / f"genome_{start_cycle_index:012x}.dat"
        )

    @classmethod
    def open_for_append(
        cls,
        base_path: Path,
        shard: str,
        start_cycle_index: int,
        gene_stateless_snapshot: bytes,
        salt: bytes,
    ) -> "GenomePack":
        path = cls.make_path(base_path, shard, start_cycle_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Create and write header if file does not exist
        if not path.exists():
            with open(path, "wb") as f:
                header = cls._build_header(start_cycle_index, gene_stateless_snapshot, salt)
                f.write(header)
        return cls(path, start_cycle_index, mode="rb+")

    @classmethod
    def open_for_read(cls, path: Path) -> "GenomePack":
        # Only supports new format (magic = b'GYRO')
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != GENOME_PACK_MAGIC:
                raise ValueError(f"Not a canonical genome pack: {path}")
            f.seek(0)
            header = f.read(GENOME_PACK_HEADER_SIZE)
            start_cycle_index = struct.unpack_from("<I", header, 8)[0]
        return cls(path, start_cycle_index, mode="rb")

    @staticmethod
    def _build_header(start_cycle_index: int, gene_stateless_snapshot: bytes, salt: bytes) -> bytes:
        if len(gene_stateless_snapshot) != GENOME_PACK_SNAPSHOT_SIZE:
            raise ValueError(f"gene_stateless_snapshot must be {GENOME_PACK_SNAPSHOT_SIZE} bytes")
        if len(salt) != GENOME_PACK_SALT_SIZE:
            raise ValueError(f"salt must be {GENOME_PACK_SALT_SIZE} bytes")
        header = bytearray(GENOME_PACK_HEADER_SIZE)
        struct.pack_into(
            "<4sBBHI",
            header,
            0,
            GENOME_PACK_MAGIC,
            GENOME_PACK_VERSION,
            GENOME_PACK_FLAGS,
            GENOME_PACK_HEADER_SIZE,
            start_cycle_index,
        )
        header[12 : 12 + GENOME_PACK_SNAPSHOT_SIZE] = gene_stateless_snapshot
        header[108 : 108 + GENOME_PACK_SALT_SIZE] = salt
        # Reserved 8 bytes at 120-127 (already zero)
        return bytes(header)

    def _open(self):
        f = open(self.path, self.mode)
        self.file = cast(BinaryIO, f)
        if self.file is None:
            raise RuntimeError("GenomePack file is not open")
        self.header = self.file.read(GENOME_PACK_HEADER_SIZE)
        self.file.seek(0, 2)  # Seek to end for append
        file_size = self.file.tell()
        self.cycles_written = max(0, (file_size - GENOME_PACK_HEADER_SIZE) // GENOME_CYCLE_SIZE)

    def append_cycle(self, cycle_bytes: bytes):
        if len(cycle_bytes) != GENOME_CYCLE_SIZE:
            raise ValueError(f"Cycle must be {GENOME_CYCLE_SIZE} bytes")
        if self.file is None:
            raise RuntimeError("GenomePack file is not open")
        self.file.seek(0, 2)  # Always append
        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
        try:
            self.file.write(cycle_bytes)
            self.file.flush()
            os.fsync(self.file.fileno())
            self.cycles_written += 1
        finally:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)

    def read_cycles(self, start_global_idx: int, count: int) -> List[bytes]:
        if self.file is None:
            raise RuntimeError("GenomePack file is not open")
        # Compute relative offset
        rel = start_global_idx - self.start_cycle_index
        if rel < 0:
            raise ValueError(
                f"Requested cycle {start_global_idx} before pack start {self.start_cycle_index}"
            )
        self.file.seek(GENOME_PACK_HEADER_SIZE + rel * GENOME_CYCLE_SIZE)
        out = []
        for _ in range(count):
            data = self.file.read(GENOME_CYCLE_SIZE)
            if len(data) < GENOME_CYCLE_SIZE:
                break
            out.append(data)
        return out

    def close(self):
        if self.file:
            self.file.close()
            self.file = None


# === GyroCrypt helpers and EncryptedFile abstraction ===
def make_keystream(snapshot: bytes, salt: bytes, key: bytes, block_size: int = 24) -> bytes:
    """
    Deterministic keystream generator for agent-private file encryption.
    - snapshot: 96B gene stateless snapshot
    - salt: 12B random/session
    - key: agent's secret key (16-32B)
    - block_size: size of keystream block (default 24)
    Returns: keystream of length block_size
    """
    if len(snapshot) != GENOME_PACK_SNAPSHOT_SIZE:
        raise ValueError(f"snapshot must be {GENOME_PACK_SNAPSHOT_SIZE} bytes")
    if len(salt) != GENOME_PACK_SALT_SIZE:
        raise ValueError(f"salt must be {GENOME_PACK_SALT_SIZE} bytes")
    if len(key) < 16:
        raise ValueError("key must be at least 16 bytes")
    # Pad key to 32 bytes
    key = key + b"\x00" * (32 - len(key))
    # Mix salt into snapshot (XOR first bytes)
    mixed = bytearray(snapshot)
    for i in range(GENOME_PACK_SALT_SIZE):
        mixed[i] ^= salt[i]
    # Split into four 24-byte quarters
    quarters = [mixed[i * 24 : (i + 1) * 24] for i in range(4)]
    # Gyration codes from key (first byte of each 8B chunk)
    gyration_codes = [key[i * 8] & 0x3 for i in range(4)]
    # Permute quarters
    for q, code in enumerate(gyration_codes):
        qarr = quarters[q]
        if code == 1:
            for j in range(24):
                qarr[j] ^= 0xFF
        elif code == 2:
            for row in [0, 2]:
                for col in range(6):
                    idx = row * 6 + col
                    if idx < 24:
                        qarr[idx] ^= 0xFF
        elif code == 3:
            for row in [1, 3]:
                for col in range(6):
                    idx = row * 6 + col
                    if idx < 24:
                        qarr[idx] ^= 0xFF
        quarters[q] = qarr
    # XOR pairs to create keystream
    keystream = bytearray(block_size)
    for i in range(block_size):
        keystream[i] = quarters[0][i] ^ quarters[1][i] ^ quarters[2][i] ^ quarters[3][i]
    return bytes(keystream)


def xor_bytes(data: bytes, keystream: bytes) -> bytes:
    return bytes(b ^ keystream[i % len(keystream)] for i, b in enumerate(data))


class EncryptedFile:
    """
    Helper for reading/writing encrypted agent-private JSON files with 128B header.
    """

    @staticmethod
    def write_json(
        path: Path, obj: dict, key: bytes, snapshot: bytes, salt: bytes, magic: bytes
    ) -> None:
        # Encode JSON and pad to 24B boundary
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        pad = (-len(raw)) % 24
        if pad:
            raw += b"\x00" * pad
        # Build header
        header = bytearray(128)
        struct.pack_into("<4sBBHI", header, 0, magic, 2, 1, 128, len(raw))
        header[12 : 12 + GENOME_PACK_SNAPSHOT_SIZE] = snapshot
        header[108 : 108 + GENOME_PACK_SALT_SIZE] = salt
        # Reserved 8B at 120-127
        if os.getenv("GYRO_DEBUG"):
            print(
                f"[DEBUG] write_json: payload_size={len(raw)}, snapshot={snapshot[:8].hex()}..., salt={salt.hex()}, raw[:64]={raw[:64].hex()} (len={len(raw)})"
            )
        # Encrypt
        keystream = make_keystream(snapshot, salt, key)
        ciphertext = xor_bytes(raw, keystream)
        # Write
        with open(path, "wb") as f:
            f.write(header)
            f.write(ciphertext)

    @staticmethod
    def read_json(path: Path, key: bytes, magic: bytes) -> dict:
        with open(path, "rb") as f:
            header = f.read(128)
            if header[:4] != magic:
                raise ValueError(f"Not an encrypted file with magic {magic!r}")
            payload_size = struct.unpack_from("<I", header, 8)[0]
            snapshot = header[12 : 12 + GENOME_PACK_SNAPSHOT_SIZE]
            salt = header[108 : 108 + GENOME_PACK_SALT_SIZE]
            ciphertext = f.read(payload_size)
        
        if os.getenv("GYRO_DEBUG"):
            print(
                f"[DEBUG] read_json: payload_size={payload_size}, snapshot={snapshot[:8].hex()}..., salt={salt.hex()}, ciphertext_len={len(ciphertext)}"
            )
        
        keystream = make_keystream(snapshot, salt, key)
        raw = xor_bytes(ciphertext, keystream)
        
        if os.getenv("GYRO_DEBUG"):
            print(f"[DEBUG] read_json: raw[:64]={raw[:64].hex()} (len={len(raw)})")
        
        # Remove trailing zeros
        raw = raw.rstrip(b"\x00")
        return json.loads(raw.decode("utf-8"))


class IntelligenceEngine:
    """
    Main orchestration engine that coordinates S3 processing and S2 persistence.
    This is the only class that writes to S2 storage and the sole controller of
    the three S3 inference engines.

    This class manages:
    1. Agent lifecycle and state persistence
    2. Stream processing through S3 engines
    3. File I/O for genome packs and dictionaries
    4. GyroCrypt encryption/decryption
    5. Pattern learning and format management
    """

    # Class constants for the MAX_CYCLE_INDEX only, since we've consolidated other constants
    MAX_CYCLE_INDEX = 0xFFFFFFFF  # 32-bit max
    
    @property
    def cycle_index(self) -> int:
        """Return the engine's current cycle index (source of truth in GovernanceEngine)."""
        return self.governance_engine.cycle_index

    @cycle_index.setter
    def cycle_index(self, value: int) -> None:
        self.governance_engine.cycle_index = value

    def __init__(
        self,
        agent_uuid: Optional[str] = None,
        base_path: Union[str, Path, None] = None,
        encryption_enabled: bool = True,
        gyrocrypt_key: Optional[bytes] = None,
    ):
        if base_path is None:
            raise ValueError(
                "IntelligenceEngine requires explicit base_path to avoid path mismatches."
            )
        if agent_uuid is None:
            raise ValueError(
                "IntelligenceEngine requires an explicit agent_uuid. Refusing to generate a new agent automatically."
            )
        self.base_path = Path(base_path)
        self.agent_uuid = agent_uuid
        self.shard = get_shard_from_uuid(self.agent_uuid)
        self._encryption_enabled = encryption_enabled
        self._ensure_directories()

        # Track the last pack path even after the pack object is closed so that
        # callers such as MessageStore.flush_thread can obtain a valid
        # reference *after* IntelligenceEngine.finalize() has run. This
        # mirrors how callers in the test‑suite expect `get_last_pack_info()`
        # to behave.
        self._last_pack_path: Optional[str] = None  # persisted after close

        # Canonical GyroCrypt key management
        if self._encryption_enabled:
            self._gyrocrypt_key = self._load_or_create_key(gyrocrypt_key)
        else:
            self._gyrocrypt_key = None
            
        self.pack_size = self._load_manifest_config()
        self._load_agent_state()
        self._load_format()
        self._initialize_engines()
        
        # Transfer saved cycle count if it exists
        if hasattr(self, "_pending_cycle_index"):
            self.cycle_index = self._pending_cycle_index
            del self._pending_cycle_index
            
        self.pack: Optional[GenomePack] = None
        self._pending_cycle_buffer: List[Tuple[int, int]] = []
        self._current_resonance_flags: List[bool] = []
        self._last_pack_start_index: Optional[int] = None
        self._open_current_pack()
        
    def _agency_path(self, *parts) -> Path:
        """Helper for building agency paths to reduce duplication"""
        return self.base_path / "agency" / Path(*parts)

    def _session_path(self, enc: bool) -> Path:
        """Helper for building session paths to reduce duplication"""
        session_dir = self.base_path / "agents" / self.shard / self.agent_uuid / "g5_information"
        return session_dir / ("session.json.enc" if enc else "session.json")

    def _ensure_directories(self) -> None:
        """
        Create all required directories for the agent.
        """
        # Global directories
        self._agency_path("g1_information", self.shard).mkdir(parents=True, exist_ok=True)
        self._agency_path("g2_information").mkdir(parents=True, exist_ok=True)
        self._agency_path("g4_information", self.shard).mkdir(parents=True, exist_ok=True)
        self._agency_path("g5_information", self.shard).mkdir(parents=True, exist_ok=True)

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
            Pack size from manifest or default (64MB)
        """
        manifest_path = self.base_path / "s2_manifest.json"

        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                return manifest.get("pack_size", DEFAULT_PACK_SIZE)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading manifest: {e}. Using default pack size.")

        return DEFAULT_PACK_SIZE

    def _read_session(self) -> dict:
        """Helper to read session with encryption handling"""
        session_path = self._session_path(self._encryption_enabled)
        if not session_path.exists():
            return {}
            
        if self._encryption_enabled:
            if self._gyrocrypt_key is None:
                raise ValueError("GyroCrypt key is not set but encryption is enabled.")
            return EncryptedFile.read_json(session_path, self._gyrocrypt_key, b"GYR5")
        else:
            with open(session_path, "r") as f:
                return json.load(f)

    def _write_session(self, session_data: dict) -> None:
        """Helper to write session with encryption handling"""
        session_path = self._session_path(self._encryption_enabled)
        session_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self._encryption_enabled:
            if self._gyrocrypt_key is None:
                raise ValueError("GyroCrypt key is not set but encryption is enabled.")
            snapshot = self.get_gene_stateless_snapshot()
            salt = os.urandom(GENOME_PACK_SALT_SIZE)
            EncryptedFile.write_json(
                session_path, session_data, self._gyrocrypt_key, snapshot, salt, b"GYR5"
            )
        else:
            with open(session_path, "w") as f:
                json.dump(session_data, f, indent=2)

    def _load_agent_state(self) -> None:
        """
        Load the agent's session state from disk (encrypted if enabled).
        No key is ever read from or written to the session file.
        """
        self.session = self._read_session()
        # Store cycle index temporarily if present
        if "cycle_index" in self.session:
            self._pending_cycle_index = self.session["cycle_index"]

    def _load_format(self) -> None:
        """
        Load the agent's format (byte_to_token, token_to_byte) from g3_information.
        """
        dict_dir = self.base_path / "agents" / self.shard / self.agent_uuid / "g3_information"
        dict_path = dict_dir / "format.json"
        if dict_path.exists():
            with open(dict_path, "r") as f:
                d = json.load(f)
            self.byte_to_token = d.get("byte_to_token", {})
            self.token_to_byte = d.get("token_to_byte", {})
        else:
            self.byte_to_token = {}
            self.token_to_byte = {}

    def _initialize_engines(self) -> None:
        """
        Initialize the three S3 inference engines.
        """
        # Use existing epigenome projection file created by initialize_system
        epigenome_path = self._agency_path("g2_information", "g2_information.dat")

        # Create inference engines
        self.governance_engine = GovernanceEngine()
        self.information_engine = InformationEngine(str(epigenome_path))
        self.inference_engine = InferenceEngine(agent_uuid=self.agent_uuid)

        # Set initial phase from session if available
        if "phase" in self.session:
            self.governance_engine.phase = self.session["phase"]

    def process_stream(self, data_bytes: bytes) -> Dict[str, List[Any]]:
        """
        Process a stream of bytes through the full GyroSI pipeline.

        This is the main entry point for the Intelligence Engine and the
        only public API that processes data.

        Args:
            data_bytes: Raw bytes to process

        Returns:
            Format containing all artifacts from processing:
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
            # Track the starting cycle index for this stream
            self._last_cycle_start = self.cycle_index
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
            try:
                self._update_session()
            except:
                pass
            raise

    def finalize(self):
        """
        Force any in-flight (partial) cycle to disk as a full cycle (padding with no-ops if needed), and flush the pack file.
        """
        if self._pending_cycle_buffer:
            needed = 48 - len(self._pending_cycle_buffer)
            if needed > 0:
                self._pending_cycle_buffer.extend([VOID_OP_PAIR] * needed)
            self._write_cycle(self._pending_cycle_buffer)
            self._pending_cycle_buffer.clear()
            self._current_resonance_flags.clear()
        if self.pack:
            self.pack.close()
            self.pack = None

    def get_last_pack_info(self) -> dict:
        """
        Return metadata about the last pack/cycle range written – even after
        the pack object has been closed by `finalize()`.
        """
        pack_path = str(self.pack.path) if self.pack is not None else self._last_pack_path
        pack_uuid = Path(pack_path).name if pack_path else None
        return {
            "pack_path": pack_path,
            "pack_uuid": pack_uuid,
            "cycle_index_start": self._last_cycle_start,
            "cycle_index_end": self.cycle_index,
        }

    def _handle_compressed_block(self, block: CompressedBlock) -> bool:
        """
        Process a compressed block from the inference engine.

        Args:
            block: CompressedBlock event from InferenceEngine

        Returns:
            True if the cycle was written to disk, False otherwise.
        """
        if block.block_type == "full_cycle":
            # Write the full cycle to disk (pad if needed)
            ops = block.data["ops"]
            if len(ops) < 48:
                ops = ops + [VOID_OP_PAIR] * (48 - len(ops))
            self._write_cycle(ops)
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
            self.cycle_index += 1

        elif block.block_type == "pruned_cycle":
            # Log pruned cycles for analytics
            if "pruned_cycles" not in self.session:
                self.session["pruned_cycles"] = 0

            self.session["pruned_cycles"] += 1

        return False

    def _persist_session(self) -> None:
        """
        Save the agent's session state to disk (encrypted if enabled).
        """
        self._write_session(self.session)

    def _update_session(self) -> None:
        """
        Update the session file with current state (encrypted if enabled).
        """
        self.session["phase"] = self.governance_engine.phase
        self.session["last_checkpoint"] = datetime.utcnow().isoformat()
        self.session["agent_uuid"] = self.agent_uuid
        self.session["cycle_index"] = self.cycle_index
        self._persist_session()

    def _open_current_pack(self) -> None:
        """
        Open a new genome pack file for writing using GenomePack.
        Closes the current pack if one is open.
        """
        if self.pack:
            self.pack.close()
        # Use current cycle index as start index
        start_cycle_index = self.cycle_index
        # Use canonical snapshot and random salt
        gene_stateless_snapshot = self.get_gene_stateless_snapshot()
        salt = os.urandom(GENOME_PACK_SALT_SIZE)
        self.pack = GenomePack.open_for_append(
            self.base_path, self.shard, start_cycle_index, gene_stateless_snapshot, salt
        )
        # keep a durable reference
        self._last_pack_path = str(self.pack.path)
        self._last_pack_start_index = start_cycle_index

    def _write_cycle(self, op_pairs: List[Tuple[int, int]]) -> None:
        if len(op_pairs) != 48:
            raise ValueError(f"Cycle must have exactly 48 op-pairs, got {len(op_pairs)}")
        cycle_data = bytearray(24)
        for i in range(24):
            op1 = op_pairs[2 * i]
            op2 = op_pairs[2 * i + 1]
            hi = ((op1[0] & 0x7) << 1) | (op1[1] & 0x1)
            lo = ((op2[0] & 0x7) << 1) | (op2[1] & 0x1)
            cycle_data[i] = (hi << 4) | lo
        # Check if we need a new pack file
        if (
            self.pack is not None
            and self.pack.cycles_written * GENOME_CYCLE_SIZE
            + GENOME_PACK_HEADER_SIZE
            + GENOME_CYCLE_SIZE
            > self.pack_size
        ):
            self._open_current_pack()
        if self.pack is not None:
            self.pack.append_cycle(bytes(cycle_data))
            self.cycle_index += 1
            if self.cycle_index >= self.MAX_CYCLE_INDEX:
                print("WARNING: Cycle index wrapped around")
                self.cycle_index = 0
                self._open_current_pack()  # Open new pack after wrapping

    def get_gene_stateless_snapshot(self) -> bytes:
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

    def _sanitize_op_pair(self, op_pair) -> Tuple[int, int]:
        if (
            isinstance(op_pair, tuple)
            and len(op_pair) == 2
            and all(isinstance(x, int) for x in op_pair)
        ):
            return cast(Tuple[int, int], op_pair)
        return (0, 0)

    def generate(self, prompt: bytes = b"", max_length: int = 100) -> bytes:
        """
        Generate bytes using learned patterns and format.
        """
        if prompt:
            self.process_stream(prompt)
        generated = bytearray()
        for _ in range(max_length):
            op_pair1 = self._sanitize_op_pair(self.inference_engine.predict_next_operation())
            op_pair2 = self._sanitize_op_pair(self.inference_engine.predict_next_operation())
            if op_pair1 == (7, 0):
                op_pair1 = (0, 0)
            if op_pair2 == (7, 0):
                op_pair2 = (0, 0)
            a1, a2 = op_pair1
            b1, b2 = op_pair2
            next_byte = gyrations_to_byte((a1, a2), (b1, b2))
            generated.append(next_byte)
            self.process_stream(bytes([next_byte]))
        return bytes(generated)

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the engine.

        Returns:
            Format with engine state
        """
        return {
            "agent_uuid": self.agent_uuid,
            "governance": self.governance_engine.get_state(),
            "information": self.information_engine.get_state(),
            "inference": self.inference_engine.get_state(),
            "encryption_enabled": self._encryption_enabled,
            "cycle_index": self.cycle_index,
            "pack_bytes": self.pack.cycles_written * GENOME_CYCLE_SIZE if self.pack else 0,
        }

    def learn_token_mapping(self, tokens_bytes: Dict[str, bytes]) -> None:
        """
        Learn mappings between tokens and byte sequences.
        """
        for token, byte_seq in tokens_bytes.items():
            byte_list = list(byte_seq)
            byte_key = str(byte_list)
            self.byte_to_token[byte_key] = token
            self.token_to_byte[token] = byte_list
        
        # Inline _persist_format logic
        dict_dir = self.base_path / "agents" / self.shard / self.agent_uuid / "g3_information"
        dict_path = dict_dir / "format.json"
        d = {
            "byte_to_token": self.byte_to_token,
            "token_to_byte": self.token_to_byte,
        }
        with open(dict_path, "w") as f:
            json.dump(d, f, indent=2)

    # === Canonical token<->op-pair encode/decode ===
    MAX_TOKEN_BYTES = 4  # Default max token length in bytes (adjust as needed)

    def encode_tokens_to_bytes(self, tokens: list) -> bytes:
        """
        Encode tokens to bytes using agent format only.
        """
        out_bytes = bytearray()
        for token in tokens:
            byte_seq = self.token_to_byte.get(token)
            if byte_seq is not None:
                out_bytes.extend(byte_seq)
            # else: skip or handle unknown token
        return bytes(out_bytes)

    def decode_bytes_to_tokens(self, bstream: bytes) -> list:
        """
        Decode bytes to tokens using agent format only.
        """
        tokens = []
        i = 0
        while i < len(bstream):
            found = False
            # Try all possible token lengths (up to MAX_TOKEN_BYTES)
            for l in range(self.MAX_TOKEN_BYTES, 0, -1):
                chunk = list(bstream[i : i + l])
                token = self.byte_to_token.get(str(chunk))
                if token is not None:
                    tokens.append(token)
                    i += l
                    found = True
                    break
            if not found:
                i += 1  # skip unknown byte
        return tokens

    def close(self) -> None:
        """
        Close the engine and flush all data to disk.
        """
        self.finalize()

    def __del__(self) -> None:
        """
        Ensure cleanup on garbage collection.
        """
        self.close()

    def read_genome_segment(self, pack_path: str, first_cycle: int, num_cycles: int) -> bytes:
        """
        Read and decode a segment from a genome pack using canonical GenomePack logic.
        Args:
            pack_path: Path to the genome pack file
            first_cycle: Index of the first cycle to read
            num_cycles: Number of cycles to read
        Returns:
            Decoded byte stream for the requested cycles
        """
        pack = GenomePack.open_for_read(Path(pack_path))
        cycles = pack.read_cycles(first_cycle, num_cycles)
        pack.close()
        # ▶▶ return **all** bytes in order; don't discard every second byte
        out = bytearray()
        for cycle in cycles:
            out.extend(cycle)
        return bytes(out)

    def _load_or_create_key(self, supplied: Optional[bytes]) -> bytes:
        """
        Canonical GyroCrypt key loader/creator. Only this method may read/write the key file.
        - If supplied is not None: must match file if exists, else writes it.
        - If supplied is None: loads file if exists, else generates and writes new key.
        """
        key_path = (
            self.base_path
            / "agents"
            / self.shard
            / self.agent_uuid
            / "g5_information"
            / "gyrocrypt.key"
        )
        if supplied is not None:
            if key_path.exists():
                on_disk = key_path.read_bytes()
                if on_disk != supplied:
                    raise ValueError("Supplied key differs from existing key‑file")
            else:
                key_path.write_bytes(supplied)
            return supplied
        if key_path.exists():
            return key_path.read_bytes()
        key = os.urandom(32)
        key_path.write_bytes(key)
        return key


# System-level functions
def initialize_system(base_path: Union[str, Path] = "s2_information") -> Dict[str, Any]:
    """
    Create the S2 directory structure and initialize the system.

    Args:
        base_path: Base path for S2 storage (default: "s2_information")

    Returns:
        Format with system information
    """
    base_path = Path(base_path)

    # Create base directories
    base_path.mkdir(exist_ok=True)
    (base_path / "agency").mkdir(exist_ok=True)
    (base_path / "agents").mkdir(exist_ok=True)

    # Create agency subdirectories (but NOT all shard folders)
    for subdir in ["g1_information", "g2_information", "g4_information", "g5_information"]:
        (base_path / "agency" / subdir).mkdir(exist_ok=True)

    # Only create epigenome file if not exists
    epigenome_path = base_path / "agency" / "g2_information" / "g2_information.dat"
    if not epigenome_path.exists():
        build_epigenome_projection(str(epigenome_path))

    # Create manifest
    manifest = {
        "version": "1.0",
        "pack_size": DEFAULT_PACK_SIZE,
        "shard_prefix_length": 2,
        "initialized": datetime.utcnow().isoformat(),
    }

    manifest_path = base_path / "s2_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "base_path": str(base_path),
        "manifest_path": str(manifest_path),
        "epigenome_path": str(epigenome_path),
        "version": manifest["version"],
        "status": "initialized",
    }


def create_agent(
    agent_uuid: Optional[str] = None, base_path: Union[str, Path] = "s2_information"
) -> str:
    """
    Create a new agent with session file only.

    Args:
        agent_uuid: Optional UUID for the agent
        base_path: Base path for S2 storage (default: "s2_information")

    Returns:
        Agent UUID
    """
    agent_id = agent_uuid or str(uuid.uuid4())
    shard = get_shard_from_uuid(agent_id)
    base_path = Path(base_path)

    # Create agent directories
    agent_dir = base_path / "agents" / shard / agent_id
    (agent_dir / "g4_information").mkdir(parents=True, exist_ok=True)
    (agent_dir / "g5_information").mkdir(parents=True, exist_ok=True)
    (agent_dir / "g3_information").mkdir(parents=True, exist_ok=True)  # For future format

    # Create empty session
    session = {
        "agent_uuid": agent_id,
        "created": datetime.utcnow().isoformat(),
        "last_checkpoint": None,
        "phase": 0,
        "cycle_index": 0,  # Changed from cycle_count to cycle_index for consistency
    }
    with open(agent_dir / "g5_information" / "session.json", "w") as f:
        json.dump(session, f, indent=2)

    return agent_id


def set_active_agent(
    agent_uuid: str, base_path: Union[str, Path] = "s2_information"
) -> IntelligenceEngine:
    """
    Create and return an IntelligenceEngine for the specified agent.

    Args:
        agent_uuid: UUID of the agent to activate
        base_path: Base path for S2 storage (default: "s2_information")

    Returns:
        Initialized IntelligenceEngine
    """
    return IntelligenceEngine(agent_uuid=agent_uuid, base_path=base_path)


def process_stream(
    data_bytes: bytes,
    agent_uuid: Optional[str] = None,
    base_path: Union[str, Path] = "s2_information",
) -> Dict[str, List[Any]]:
    """
    Process a byte stream through the GyroSI pipeline.

    This is a convenience function that creates a temporary engine,
    processes the data, and then closes the engine.

    Args:
        data_bytes: Raw bytes to process
        agent_uuid: Optional agent UUID
        base_path: Base path for S2 storage (default: "s2_information")

    Returns:
        Format of all emitted artifacts
    """
    if agent_uuid is None:
        raise ValueError(
            "process_stream requires an explicit agent_uuid. Refusing to generate a new agent automatically."
        )
    engine = IntelligenceEngine(agent_uuid=agent_uuid, base_path=base_path)
    try:
        return engine.process_stream(data_bytes)
    finally:
        engine.close()


def generate_text(
    prompt: str = "",
    max_length: int = 100,
    agent_uuid: Optional[str] = None,
    base_path: Union[str, Path] = "s2_information",
) -> str:
    """Generate text using the GyroSI Baby LM with proper codec."""
    if agent_uuid is None:
        raise ValueError(
            "generate_text requires an explicit agent_uuid. Refusing to generate a new agent automatically."
        )
    engine = IntelligenceEngine(agent_uuid=agent_uuid, base_path=base_path)
    try:
        if prompt:
            prompt_tokens = list(prompt)
            prompt_bytes = engine.encode_tokens_to_bytes(prompt_tokens)
            engine.process_stream(prompt_bytes)
        generated_bytes = engine.generate(b"", max_length)
        generated_tokens = engine.decode_bytes_to_tokens(generated_bytes)
        return "".join(str(t) for t in generated_tokens)
    finally:
        engine.close()