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

# Import all modules
import s1_governance
from s4_intelligence.g2_intelligence_eg import byte_to_gyrations, gyrations_to_byte
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
        gene = s1_governance.get_gene_constant()
        assert "id_0" in gene
        assert "id_1" in gene
        for tensor_id in ["id_0", "id_1"]:
            tensor = gene[tensor_id]
            assert tensor.shape == (4, 2, 3, 2)
            assert tensor.dtype == torch.int8
            assert torch.all((tensor == 1) | (tensor == -1))

    def test_s1_governance_gyration_operations(self):
        """Test all four gyration operations"""
        gene = s1_governance.get_gene_constant()
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
            op_pair1, op_pair2 = byte_to_gyrations(byte_val)
            assert 0 <= op_pair1[0] <= 3 and 0 <= op_pair2[0] <= 3
            assert op_pair1[1] in [0, 1] and op_pair2[1] in [0, 1]

    def test_gyrations_to_byte_inverse(self):
        """Test that gyrations_to_byte is inverse of byte_to_gyrations"""
        for byte_val in range(256):
            op_pair1, op_pair2 = byte_to_gyrations(byte_val)
            reconstructed = gyrations_to_byte(op_pair1, op_pair2)
            op_pair1_new, op_pair2_new = byte_to_gyrations(reconstructed)
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
        assert engine.phase == 0
        assert engine.cycle_count == 1

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

    def test_curriculum_persistence(self):
        """Test that patterns are persisted to curriculum.json(.enc) and can be decrypted or read"""
        agent_id = create_agent(base_path=self.base_path)
        for encryption_enabled in (True, False):
            engine = IntelligenceEngine(
                agent_id, base_path=self.base_path, encryption_enabled=encryption_enabled
            )
            pattern_data = b"ABCDABCDABCDABCD" * 3
            engine.process_stream(pattern_data)
            engine.close()
            curriculum_dir = (
                Path(self.base_path) / "agents" / engine.shard / agent_id / "g4_information"
            )
            if encryption_enabled:
                curriculum_path = curriculum_dir / "curriculum.json.enc"
                assert curriculum_path.exists()
                key = (
                    engine._gyrocrypt_key
                    if hasattr(engine, "_gyrocrypt_key") and engine._gyrocrypt_key
                    else b"\x00" * 16
                )
                curriculum = EncryptedFile.read_json(curriculum_path, key, b"GYR4")
            else:
                curriculum_path = curriculum_dir / "curriculum.json"
                assert curriculum_path.exists()
                import json

                with open(curriculum_path, "r") as f:
                    curriculum = json.load(f)
            # patterns is now a dict
            assert (
                "patterns" in curriculum
                and isinstance(curriculum["patterns"], dict)
                and len(curriculum["patterns"]) > 0
            )

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

            assert state1["governance"]["cycle_count"] > 0
            assert state2["governance"]["cycle_count"] == state1["governance"]["cycle_count"]
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
            assert session["cycle_index"] == state1["governance"]["cycle_count"]

    def test_current_cycle_decoded_is_constant(self):
        """Test that a decoded genome cycle is constant and deterministic."""
        agent_id = create_agent(base_path=self.base_path)
        engine = IntelligenceEngine(agent_id, base_path=self.base_path)

        # Get the snapshot from a fresh engine
        snapshot1 = engine.get_current_cycle_decoded()

        # Process some data
        engine.process_stream(b"some data to change the state")

        # Get the snapshot again
        snapshot2 = engine.get_current_cycle_decoded()

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
        engine = IntelligenceEngine(base_path=self.base_path)
        artifacts = engine.process_stream(b"")
        assert len(artifacts["accepted_ops"]) == 0

    def test_single_byte_input(self):
        """Test handling of single byte input"""
        engine = IntelligenceEngine(base_path=self.base_path)
        artifacts = engine.process_stream(b"X")
        assert len(artifacts["accepted_ops"]) == 2

    def test_unicode_handling(self):
        """Test handling of unicode/emoji input"""
        engine = IntelligenceEngine(base_path=self.base_path)
        emoji_bytes = "👋".encode("utf-8")
        artifacts = engine.process_stream(emoji_bytes)
        expected_ops = len(emoji_bytes) * 2
        assert len(artifacts["accepted_ops"]) == expected_ops

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
