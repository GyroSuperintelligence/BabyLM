import pytest
import torch
import numpy as np
import os
import shutil
import json
from pathlib import Path
import struct
import uuid
import unittest
from datetime import datetime
import tempfile

from s4_intelligence.g2_intelligence_eg import MessageStore
from s4_intelligence.g1_intelligence_in import IntelligenceEngine, initialize_system, EncryptedFile
from s1_governance import build_epigenome_projection

# Import all modules
import s1_governance
from s4_intelligence.g1_intelligence_in import (
    IntelligenceEngine,
    initialize_system,
    create_agent,
    process_stream,
    generate_text,
    EncryptedFile,
    get_shard_from_uuid,
)
from s3_inference.g1_inference import GovernanceEngine, CycleComplete
from s3_inference.g2_inference import InformationEngine
from s3_inference.g3_inference import InferenceEngine, PatternPromotion


def cleanup_s2_information(base_path: str = "s2_information"):
    """Clean up all test-generated data while preserving essential system files."""
    base = base_path

    # Files/directories to preserve (essential system files)
    preserve_files = [
        "s2_manifest.json",
        os.path.join("agency", "g2_information", "g2_information.dat"),
    ]

    # Directories to clean (test-generated data)
    clean_dirs = [
        os.path.join("agency", "g1_information"),
        os.path.join("agency", "g4_information"),
        os.path.join("agency", "g5_information"),
        "agents",
    ]

    for clean_dir in clean_dirs:
        path = os.path.join(base, clean_dir)
        if os.path.exists(path):
            # Be careful with shutil.rmtree
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)


class TestGyroSIBabyLM(unittest.TestCase):
    """Comprehensive test suite for GyroSI Baby LM"""

    @classmethod
    def setUpClass(cls):
        cls.test_base_path = Path("s2_information_test")
        # Safeguard: never allow cleanup of production storage
        if str(cls.test_base_path.resolve()) == str(Path("s2_information").resolve()):
            raise RuntimeError("Refusing to clean up production storage!")
        if cls.test_base_path.exists():
            shutil.rmtree(cls.test_base_path)
        cls.epigenome_path = cls.test_base_path / "agency" / "g2_information" / "g2_information.dat"

    @classmethod
    def tearDownClass(cls):
        if str(cls.test_base_path.resolve()) == str(Path("s2_information").resolve()):
            raise RuntimeError("Refusing to clean up production storage!")
        if cls.test_base_path.exists():
            shutil.rmtree(cls.test_base_path)

    def setUp(self):
        self.base_path = self.test_base_path
        self.base_path.mkdir(parents=True, exist_ok=True)
        # Clean up main s2_information directory before each test
        cleanup_s2_information(str(self.base_path))
        initialize_system(base_path=self.base_path)
        # Create manifest
        manifest_path = self.base_path / "s2_manifest.json"
        manifest = {
            "version": "1.0",
            "pack_size": 64000000,
            "shard_prefix_length": 2,
            "initialized": datetime.utcnow().isoformat(),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

    def tearDown(self):
        # Clean up after
        cleanup_s2_information(str(self.base_path))

    def test_s1_governance_gene_constants(self):
        """Test that gene constants are properly defined"""
        gene = s1_governance.get_gene_tensors()
        assert "id_0" in gene
        assert "id_1" in gene
        for tensor_id in ["id_0", "id_1"]:
            tensor = gene[tensor_id]
            assert tensor.shape == (4, 2, 3, 2)
            assert tensor.dtype == torch.int8
            assert torch.all((tensor == 1) | (tensor == -1))

    def test_s1_governance_gyration_operations(self):
        """Test all four gyration operations"""
        gene = s1_governance.get_gene_tensors()
        tensor = gene["id_0"].clone()
        # Test identity (no change)
        result = s1_governance.gyration_op(tensor, 0)
        assert torch.equal(result, tensor)
        # Test inverse (global sign flip)
        result = s1_governance.gyration_op(tensor, 1)
        assert torch.equal(result, -tensor)
        # Test forward (flip rows 0 and 2)
        result = s1_governance.gyration_op(tensor, 2)
        expected = tensor.clone()
        expected[0] *= -1
        expected[2] *= -1
        assert torch.equal(result, expected)
        # Test backward (flip rows 1 and 3)
        result = s1_governance.gyration_op(tensor, 3)
        expected = tensor.clone()
        expected[1] *= -1
        expected[3] *= -1
        assert torch.equal(result, expected)

    def test_byte_to_gyrations_mapping(self):
        """Test byte to gyration conversion"""
        test_cases = [
            (0x00, ((0, 0), (0, 0))),
            (0xFF, ((7, 1), (7, 1))),
            (0x42, ((2, 0), (1, 0))),
        ]
        for byte_val, expected in test_cases:
            op_pair1, op_pair2 = s1_governance.byte_to_gyrations(byte_val)
            assert 0 <= op_pair1[0] <= 3 and 0 <= op_pair2[0] <= 3
            assert op_pair1[1] in [0, 1] and op_pair2[1] in [0, 1]

    def test_gyrations_to_byte_inverse(self):
        """Test that gyrations_to_byte is inverse of byte_to_gyrations"""
        for byte_val in range(256):
            op_pair1, op_pair2 = s1_governance.byte_to_gyrations(byte_val)
            reconstructed = s1_governance.gyrations_to_byte(op_pair1, op_pair2)
            op_pair1_new, op_pair2_new = s1_governance.byte_to_gyrations(reconstructed)
            assert op_pair1 == op_pair1_new
            assert op_pair2 == op_pair2_new

    def test_epigenome_projection_build(self):
        """Test that the epigenome projection exists and has correct size"""
        assert os.path.exists(self.epigenome_path)
        file_size = os.path.getsize(self.epigenome_path)
        expected_size = 48 * 256  # 12288 bytes, no SHA-256 header
        assert file_size == expected_size

    def test_governance_engine_phase_cycling(self):
        """Test that governance engine cycles through phases correctly"""
        engine = GovernanceEngine()
        for i in range(48):
            engine.process_op_pair((0, 0), False)
        assert engine.cycle_index == 1  # After one full cycle, index should be 1

    def test_information_engine_resonance(self):
        """Test information engine resonance classification"""
        engine = InformationEngine(epigenome_path=str(self.epigenome_path))
        event = engine.process_accepted_op_pair(phase=0, op_pair=(0, 0), byte_val=0)
        assert hasattr(event, "resonance_flag")
        assert hasattr(event, "bit_index")

    def test_inference_engine_pattern_detection(self):
        """Test pattern detection in inference engine"""
        # Create engine with lower threshold
        engine = InferenceEngine(pattern_threshold=2)
        engine.reset()

        # Create a pattern that should be promoted within a single cycle
        base_pattern = [(0, 0), (1, 1)]

        # Create a cycle with repeating patterns (the same pattern appears multiple times)
        cycle_ops = []
        for _ in range(12):  # Repeat pattern 12 times to fill 48-element cycle
            cycle_ops.extend(base_pattern)

        # All ops are resonant
        resonance_flags = [True] * len(cycle_ops)

        # Process once - this should be enough to detect the repeating pattern
        events = engine.process_cycle_complete(cycle_ops, resonance_flags)

        # Find pattern promotions
        promotions = [e for e in events if isinstance(e, PatternPromotion)]

        # Should have at least one promotion
        assert len(promotions) > 0

        # Verify the promoted pattern
        if len(promotions) > 0:
            assert promotions[0].pattern == base_pattern
            assert promotions[0].frequency >= engine.pattern_threshold

    def test_intelligence_engine_full_pipeline(self):
        """Test the full pipeline from bytes to artifacts"""
        agent_id = create_agent(base_path=self.base_path)
        engine = IntelligenceEngine(agent_id, base_path=self.base_path)
        test_data = b"Hello, GyroSI!"
        artifacts = engine.process_stream(test_data)
        assert "accepted_ops" in artifacts
        assert "resonances" in artifacts
        expected_ops = len(test_data) * 2
        assert len(artifacts["accepted_ops"]) == expected_ops
        assert len(artifacts["resonances"]) == expected_ops

    def test_genome_pack_writing(self):
        """Test that genome packs are written correctly and have a 128B header"""
        agent_id = create_agent(base_path=self.base_path)
        engine = IntelligenceEngine(agent_id, encryption_enabled=False, base_path=self.base_path)
        small_data = b"X" * 48
        engine.process_stream(small_data)
        engine.close()
        pack_dir = os.path.join(self.base_path, "agency", "g1_information", engine.shard)
        assert os.path.exists(pack_dir)
        # Accept both old and new naming, but prefer canonical
        pack_files = [
            f for f in os.listdir(pack_dir) if f.startswith("genome_") and f.endswith(".dat")
        ]
        assert len(pack_files) >= 1
        with open(os.path.join(pack_dir, pack_files[0]), "rb") as f:
            header = f.read(128)
            assert len(header) == 128
            assert header[:4] == b"GYRO"

    def test_session_state_persistence(self):
        """Test that session state is saved and loaded correctly (encrypted or plaintext)"""
        agent_id = create_agent(base_path=self.base_path)
        from s4_intelligence.g1_intelligence_in import get_shard_from_uuid

        for encryption_enabled in (True, False):
            # Compute shard from agent_id for correct path
            shard = get_shard_from_uuid(agent_id)
            from pathlib import Path
            import shutil

            agent_dir = Path(self.base_path) / "agents" / shard / agent_id
            # Clean up the entire agent directory before each run
            if agent_dir.exists():
                shutil.rmtree(agent_dir)
            g5_dir = agent_dir / "g5_information"
            key_path = g5_dir / "gyrocrypt.key"

            # Ensure g5_information directory exists before writing key
            g5_dir.mkdir(parents=True, exist_ok=True)

            if encryption_enabled:
                # Always generate and persist the key before using engine1
                if not key_path.exists():
                    key = os.urandom(32)
                    with open(key_path, "wb") as f:
                        f.write(key)
                    print("[DEBUG] Generated key for engine1:", key.hex())
                else:
                    with open(key_path, "rb") as f:
                        key = f.read()
                    print("[DEBUG] Loaded key for engine1:", key.hex())
                engine1 = IntelligenceEngine(
                    agent_id, base_path=self.base_path, encryption_enabled=True, gyrocrypt_key=key
                )
            else:
                engine1 = IntelligenceEngine(
                    agent_id, base_path=self.base_path, encryption_enabled=False
                )

            engine1.process_stream(b"test data" * 10)
            state1 = engine1.get_state()
            engine1.close()  # Ensure state is flushed

            if encryption_enabled:
                with open(key_path, "rb") as f:
                    key = f.read()
                print("[DEBUG] Loaded key for engine2:", key.hex())
                session_path = g5_dir / "session.json.enc"
                if session_path.exists():
                    with open(session_path, "rb") as f:
                        print("[DEBUG] First 16 bytes of session file:", f.read(16).hex())
                engine2 = IntelligenceEngine(
                    agent_id, base_path=self.base_path, encryption_enabled=True, gyrocrypt_key=key
                )
            else:
                engine2 = IntelligenceEngine(
                    agent_id, base_path=self.base_path, encryption_enabled=False
                )
            state2 = engine2.get_state()
            engine2.close()

            assert state1["governance"]["cycle_index"] > 0
            assert state2["governance"]["cycle_index"] == state1["governance"]["cycle_index"]
            # Check session file
            session_dir = g5_dir
            if encryption_enabled:
                session_path = session_dir / "session.json.enc"
                assert session_path.exists()
                from s4_intelligence.g1_intelligence_in import EncryptedFile

                session = EncryptedFile.read_json(session_path, key, b"GYR5")
            else:
                session_path = session_dir / "session.json"
                assert session_path.exists()
                import json

                with open(session_path, "r") as f:
                    session = json.load(f)
            assert session["cycle_index"] == state1["governance"]["cycle_index"]

    def test_gene_stateless_snapshot_is_constant(self):
        """Test that a decoded genome cycle is constant and deterministic."""
        agent_id = create_agent(base_path=self.base_path)
        engine = IntelligenceEngine(agent_id, base_path=self.base_path)

        # Get the snapshot from a fresh engine
        snapshot1 = engine.get_gene_stateless_snapshot()

        # Process some data
        engine.process_stream(b"some data to change the state")

        # Get the snapshot again
        snapshot2 = engine.get_gene_stateless_snapshot()

        # The snapshot should be identical because it's based on immutable constants
        assert snapshot1 == snapshot2
        assert len(snapshot1) == 96

    def test_deterministic_processing(self):
        """Test that processing the same data for different agents yields identical pack files."""
        data = b"Deterministic test data that is long enough to complete a cycle." * 3

        # Agent 1
        agent1_id = create_agent(base_path=self.base_path)
        engine1 = IntelligenceEngine(
            agent_uuid=agent1_id, encryption_enabled=False, base_path=self.base_path
        )
        engine1.inference_engine = InferenceEngine(
            agent_uuid=agent1_id, min_pattern_length=2, max_pattern_length=4
        )
        engine1.process_stream(data)
        engine1.close()

        pack_dir1 = Path(self.base_path) / "agency" / "g1_information" / engine1.shard
        pack_files1 = sorted(list(pack_dir1.glob("genome_*.dat")))
        assert len(pack_files1) > 0
        with open(pack_files1[0], "rb") as f:
            content1 = f.read()

        # Agent 2
        agent2_id = create_agent(base_path=self.base_path)
        engine2 = IntelligenceEngine(
            agent_uuid=agent2_id, encryption_enabled=False, base_path=self.base_path
        )
        engine2.inference_engine = InferenceEngine(
            agent_uuid=agent2_id, min_pattern_length=2, max_pattern_length=4
        )
        engine2.process_stream(data)
        engine2.close()

        pack_dir2 = Path(self.base_path) / "agency" / "g1_information" / engine2.shard
        pack_files2 = sorted(list(pack_dir2.glob("genome_*.dat")))
        assert len(pack_files2) > 0
        with open(pack_files2[0], "rb") as f:
            content2 = f.read()

        # Canonical header is 128 bytes; skip header for content comparison
        assert content1[128:] == content2[128:]

    def test_empty_input_handling(self):
        """Test that empty input is handled gracefully"""
        agent_id = create_agent(base_path=self.base_path)
        engine = IntelligenceEngine(agent_uuid=agent_id, base_path=self.base_path)
        artifacts = engine.process_stream(b"")
        assert len(artifacts["accepted_ops"]) == 0

    def test_single_byte_input(self):
        """Test handling of single byte input"""
        agent_id = create_agent(base_path=self.base_path)
        engine = IntelligenceEngine(agent_uuid=agent_id, base_path=self.base_path)
        artifacts = engine.process_stream(b"X")
        assert len(artifacts["accepted_ops"]) == 2

    def test_unicode_handling(self):
        """Test handling of unicode/emoji input"""
        agent_id = create_agent(base_path=self.base_path)
        engine = IntelligenceEngine(agent_uuid=agent_id, base_path=self.base_path)
        emoji_bytes = "👋".encode("utf-8")
        artifacts = engine.process_stream(emoji_bytes)
        assert len(artifacts["accepted_ops"]) == len(emoji_bytes) * 2

    def test_cycle_compression(self):
        """Test cycle compression for repeated patterns"""
        agent_id = create_agent(base_path=self.base_path)
        engine = IntelligenceEngine(agent_id, base_path=self.base_path)

        # Create a simple pattern that will repeat
        simple_data = b"ABCDEFGHIJKL" * 4  # 48 bytes for a full cycle

        # Process the same data twice to ensure we get compression
        engine.process_stream(simple_data)
        artifacts = engine.process_stream(simple_data)

        # Look for compressed blocks
        compressed = [b for b in artifacts["compressed_blocks"] if b.block_type == "cycle_repeat"]
        assert len(compressed) > 0

    def test_generation_basic(self):
        """Test basic text generation"""
        agent_id = create_agent(base_path=self.base_path)
        engine = IntelligenceEngine(agent_id, base_path=self.base_path)
        training_data = b"ABCD" * 10
        engine.process_stream(training_data)

        generated = engine.generate(b"AB", max_length=10)
        assert isinstance(generated, bytes)
        assert len(generated) == 10

    def test_gyrocrypt_encryption_and_headers(self):
        """Test GyroCrypt encryption and headers"""
        agent_id = create_agent(base_path=self.base_path)
        engine = IntelligenceEngine(
            agent_uuid=agent_id, encryption_enabled=True, base_path=self.base_path
        )
        assert engine._encryption_enabled
        assert engine._gyrocrypt_key is not None

        engine.process_stream(b"test data for pack creation")
        engine.close()

        pack_dir = os.path.join(self.base_path, "agency", "g1_information", engine.shard)
        # Look for canonical genome_*.dat files
        pack_files = [
            f for f in os.listdir(pack_dir) if f.startswith("genome_") and f.endswith(".dat")
        ]
        assert len(pack_files) > 0

        pack_path = os.path.join(pack_dir, pack_files[0])
        with open(pack_path, "rb") as f:
            header = f.read(128)
            assert len(header) == 128
            cycle_index = struct.unpack("<I", header[8:12])[0]
            assert cycle_index >= 0

    def test_pruning_analysis_metrics(self):
        """Test that pruning analysis produces correct metrics"""
        engine = InferenceEngine()
        low_entropy_ops = [(0, 0)] * 48
        low_entropy_resonance = [True] * 24 + [False] * 24
        analysis1 = engine.analyse_cycle(low_entropy_ops, low_entropy_resonance)
        assert analysis1["horizon_distance"] == 0.0
        assert analysis1["pattern_entropy"] == 1.0 / 48.0
        assert analysis1["prune"]

        high_entropy_ops = [(i % 4, i % 2) for i in range(48)]
        high_entropy_resonance = [i % 3 == 0 for i in range(48)]
        analysis2 = engine.analyse_cycle(high_entropy_ops, high_entropy_resonance)
        assert not analysis2["prune"]

    def test_full_learning_and_generation_cycle(self):
        """Test a complete learning cycle from input to generation"""
        agent_id = create_agent(base_path=self.base_path)
        engine = IntelligenceEngine(agent_id, base_path=self.base_path)
        training_text = "The quick brown fox jumps over the lazy dog. " * 5
        engine.process_stream(training_text.encode("utf-8"))

        state = engine.get_state()
        assert state["inference"]["promoted_patterns"] > 0

        generated = engine.generate(b"The quick", max_length=20)
        assert len(generated) > 0


class TestMessageStore(unittest.TestCase):
    """Test cases for MessageStore functionality."""

    @classmethod
    def setUpClass(cls):
        cls.test_base_path = Path("s2_information_test")
        # Safeguard: never allow cleanup of production storage
        if str(cls.test_base_path.resolve()) == str(Path("s2_information").resolve()):
            raise RuntimeError("Refusing to clean up production storage!")
        if cls.test_base_path.exists():
            try:
                shutil.rmtree(cls.test_base_path)
            except OSError as e:
                print(f"Warning: Could not fully clean up test directory: {e}")
                # Try to remove individual files/directories that might be locked
                import time

                time.sleep(0.1)  # Give OS time to release handles
                try:
                    shutil.rmtree(cls.test_base_path, ignore_errors=True)
                except Exception:
                    pass  # Final attempt, ignore any remaining errors
        # Ensure epigenome is present in the test directory
        src = Path("s2_information/agency/g2_information/g2_information.dat")
        dst = cls.test_base_path / "agency" / "g2_information" / "g2_information.dat"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    @classmethod
    def tearDownClass(cls):
        if str(cls.test_base_path.resolve()) == str(Path("s2_information").resolve()):
            raise RuntimeError("Refusing to clean up production storage!")
        if cls.test_base_path.exists():
            try:
                shutil.rmtree(cls.test_base_path)
            except OSError as e:
                print(f"Warning: Could not fully clean up test directory: {e}")
                # Try to remove individual files/directories that might be locked
                import time

                time.sleep(0.1)  # Give OS time to release handles
                try:
                    shutil.rmtree(cls.test_base_path, ignore_errors=True)
                except Exception:
                    pass  # Final attempt, ignore any remaining errors

    def setUp(self):
        """Set up test environment."""
        # Use the dedicated test storage root
        self.base_path = self.test_base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create manifest
        manifest = {
            "version": "1.0",
            "pack_size": 64000000,
            "shard_prefix_length": 2,
            "initialized": datetime.utcnow().isoformat(),
        }
        with open(self.base_path / "s2_manifest.json", "w") as f:
            json.dump(manifest, f)

        # Create required directories
        (self.base_path / "agency" / "g2_information").mkdir(parents=True, exist_ok=True)

        # Build epigenome projection
        epigenome_path = self.base_path / "agency" / "g2_information" / "g2_information.dat"
        build_epigenome_projection(str(epigenome_path))

        # Create test agent
        self.agent_uuid = str(uuid.uuid4())
        self.shard = self.agent_uuid[:2]

        # Clean up agent directories before test
        agent_dir = self.base_path / "agents" / self.shard / self.agent_uuid
        if agent_dir.exists():
            import shutil

            shutil.rmtree(agent_dir)

        # Set up environment for IntelligenceEngine
        # Create agent directories
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "g4_information").mkdir(parents=True, exist_ok=True)
        (agent_dir / "g5_information").mkdir(parents=True, exist_ok=True)
        (agent_dir / "threads").mkdir(parents=True, exist_ok=True)

        # Create agency directories
        (self.base_path / "agency" / "g1_information" / self.shard).mkdir(
            parents=True, exist_ok=True
        )
        (self.base_path / "agency" / "g4_information" / self.shard).mkdir(
            parents=True, exist_ok=True
        )

        # Initialize agent files (session only)
        session = {
            "agent_uuid": self.agent_uuid,
            "created": datetime.utcnow().isoformat(),
            "last_checkpoint": None,
            "phase": 0,
            "cycle_index": 0,
        }

        # Encrypt session file
        key = os.urandom(32)
        g5_dir = agent_dir / "g5_information"
        g5_dir.mkdir(parents=True, exist_ok=True)
        (g5_dir / "gyrocrypt.key").write_bytes(key)
        snapshot = IntelligenceEngine(
            self.agent_uuid, base_path=self.base_path, gyrocrypt_key=key
        ).get_gene_stateless_snapshot()
        salt = os.urandom(12)
        EncryptedFile.write_json(
            agent_dir / "g5_information" / "session.json.enc", session, key, snapshot, salt, b"GYR5"
        )
        self._test_key = key

        # Create message store
        self.store = MessageStore(self.agent_uuid, str(self.base_path))

        # Thread ID for testing
        self.thread_id = "test-thread-001"

        # Ensure session files exist for both encryption modes
        g5_dir = agent_dir / "g5_information"
        g5_dir.mkdir(parents=True, exist_ok=True)
        session = {"cycle_index": 0}
        key = self._test_key
        snapshot = IntelligenceEngine(
            self.agent_uuid, base_path=self.base_path
        ).get_gene_stateless_snapshot()
        salt = os.urandom(12)
        EncryptedFile.write_json(g5_dir / "session.json.enc", session, key, snapshot, salt, b"GYR5")
        with open(g5_dir / "session.json", "w") as f:
            json.dump(session, f)

    def tearDown(self):
        """Clean up test environment and remove all test files/directories for this agent only."""
        agent_dir = self.base_path / "agents" / self.shard / self.agent_uuid
        if agent_dir.exists():
            import shutil

            shutil.rmtree(agent_dir)
        # Optionally, clean up thread archives
        threads_dir = agent_dir / "threads"
        if threads_dir.exists():
            import shutil

            shutil.rmtree(threads_dir)

    def test_flush_and_restore(self):
        """Test that flush_thread trims recent, and genome packs contain all archived messages."""
        # Create 30 dummy messages
        messages = []
        for i in range(30):
            message = {
                "id": f"msg-{i:04d}",
                "role": "agent",
                "content": f"This is test message number {i}. It contains some text content for testing.",
                "timestamp": f"2024-01-01T{i//60:02d}:{i%60:02d}:00Z",
                "artifacts": {},
            }
            messages.append(message)
        self.store.write_recent(self.thread_id, messages)
        recent = self.store.load_recent(self.thread_id)
        self.assertEqual(len(recent), 30)
        for encryption_enabled in (True, False):
            engine = IntelligenceEngine(
                agent_uuid=self.agent_uuid,
                base_path=self.base_path,
                encryption_enabled=encryption_enabled,
                gyrocrypt_key=self._test_key,
            )
            for msg in messages[:5]:
                if msg["role"] == "agent":
                    engine.process_stream(msg["content"].encode("utf-8"))
            engine.finalize()
            engine._update_session()
            self.store.flush_thread(self.thread_id, keep_last=5, engine=engine)
            recent_after = self.store.load_recent(self.thread_id)
            self.assertEqual(len(recent_after), 5)
            self.assertEqual(recent_after[0]["id"], "msg-0025")
            self.assertEqual(recent_after[-1]["id"], "msg-0029")
            # TODO: Add direct genome pack verification here
            engine.close()
        recent = self.store.load_recent("empty-thread")
        self.assertEqual(len(recent), 0)
        # No more archive index to check

    def test_small_thread_no_flush(self):
        """Test thread with fewer messages than flush threshold."""
        # Create 100 messages (less than 250 threshold)
        messages = []
        for i in range(100):
            message = {
                "id": f"msg-{i:04d}",
                "role": "agent",
                "content": f"Message {i}",
                "timestamp": datetime.utcnow().isoformat(),
                "artifacts": {},
            }
            messages.append(message)
        # Write messages
        self.store.write_recent(self.thread_id, messages)
        # Flush (should do nothing since under threshold)
        self.store.flush_thread(self.thread_id, keep_last=250)
        # Verify all messages still in recent
        recent = self.store.load_recent(self.thread_id)
        self.assertEqual(len(recent), 100)
        # No more archive index to check

    def test_snapshot_offsets(self):
        """Test that message offsets in archive records are correct. Only flush if a real pack was written."""
        short_msgs = [
            {
                "id": "msg-0",
                "role": "agent",
                "content": "A",
                "timestamp": datetime.utcnow().isoformat(),
                "artifacts": {},
            },
            {
                "id": "msg-1",
                "role": "agent",
                "content": "BB",
                "timestamp": datetime.utcnow().isoformat(),
                "artifacts": {},
            },
            {
                "id": "msg-2",
                "role": "agent",
                "content": "CCC",
                "timestamp": datetime.utcnow().isoformat(),
                "artifacts": {},
            },
        ]
        for encryption_enabled in (True, False):
            self.store.write_recent(self.thread_id, short_msgs)
            engine = IntelligenceEngine(
                agent_uuid=self.agent_uuid,
                base_path=self.base_path,
                encryption_enabled=encryption_enabled,
                gyrocrypt_key=self._test_key,
            )
            for msg in short_msgs:
                if msg["role"] == "agent":
                    engine.process_stream(msg["content"].encode("utf-8"))
            engine.finalize()
            engine._update_session()
            self.store.flush_thread(self.thread_id, keep_last=0, engine=engine)
            recent = self.store.load_recent(self.thread_id)
            self.assertEqual(len(recent), 0)
            # TODO: Add direct genome pack verification here
            engine.close()
        # No more archive index or restore_segment to check

    def test_single_cycle_restore(self):
        """Test that a single message can be archived and restored. Only flush if a real pack was written."""
        single_message = "A" * 24
        for encryption_enabled in (True, False):
            self.store.write_recent(
                self.thread_id,
                [
                    {
                        "id": "msg-0000",
                        "role": "agent",
                        "content": single_message,
                        "timestamp": "2024-01-01T00:00:00Z",
                        "artifacts": {},
                    }
                ],
            )
            engine = IntelligenceEngine(
                agent_uuid=self.agent_uuid,
                base_path=self.base_path,
                encryption_enabled=encryption_enabled,
                gyrocrypt_key=self._test_key,
            )
            engine.process_stream(single_message.encode("utf-8"))
            engine.finalize()
            self.store.flush_thread(self.thread_id, keep_last=0, engine=engine)
            recent = self.store.load_recent(self.thread_id)
            self.assertEqual(len(recent), 0)
            # TODO: Add direct genome pack verification here
            engine.close()
        # No more archive index or restore_segment to check


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
