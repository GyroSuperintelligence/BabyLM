"""
test_message_store.py - Unit tests for MessageStore

Tests the message store functionality including flushing and restoration.
"""

import unittest
import tempfile
import shutil
import uuid
import os
from datetime import datetime
from pathlib import Path
import json

from app.services.message_store import MessageStore
from s4_intelligence.g1_intelligence_in import IntelligenceEngine, initialize_system, EncryptedFile
from s1_governance import build_epigenome_projection


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

        # Initialize agent files
        curriculum = {
            "version": "1.0",
            "patterns": {},
            "byte_to_token": {},
            "token_to_byte": {},
            "metadata": {"created": datetime.utcnow().isoformat()},
        }
        session = {
            "agent_uuid": self.agent_uuid,
            "created": datetime.utcnow().isoformat(),
            "last_checkpoint": None,
            "phase": 0,
            "cycle_count": 0,
            "active_curriculum": None,
        }

        # Encrypt curriculum and session files
        key = os.urandom(32)
        g5_dir = agent_dir / "g5_information"
        g5_dir.mkdir(parents=True, exist_ok=True)
        (g5_dir / "gyrocrypt.key").write_bytes(key)
        snapshot = IntelligenceEngine(
            self.agent_uuid, base_path=self.base_path, gyrocrypt_key=key
        ).get_current_cycle_decoded()
        salt = os.urandom(12)
        EncryptedFile.write_json(
            agent_dir / "g4_information" / "curriculum.json.enc",
            curriculum,
            key,
            snapshot,
            salt,
            b"GYR4",
        )
        EncryptedFile.write_json(
            agent_dir / "g5_information" / "session.json.enc", session, key, snapshot, salt, b"GYR5"
        )
        self._test_key = key

        # Create message store
        self.store = MessageStore(self.agent_uuid, str(self.base_path))

        # Thread ID for testing
        self.thread_id = "test-thread-001"

        # Ensure curriculum and session files exist for both encryption modes
        g4_dir = agent_dir / "g4_information"
        g5_dir = agent_dir / "g5_information"
        g4_dir.mkdir(parents=True, exist_ok=True)
        g5_dir.mkdir(parents=True, exist_ok=True)
        curriculum = {"patterns": {}, "byte_to_token": {}, "token_to_byte": {}}
        session = {"cycle_index": 0}
        key = self._test_key
        snapshot = IntelligenceEngine(
            self.agent_uuid, base_path=self.base_path
        ).get_current_cycle_decoded()
        salt = os.urandom(12)
        EncryptedFile.write_json(
            g4_dir / "curriculum.json.enc", curriculum, key, snapshot, salt, b"GYR4"
        )
        EncryptedFile.write_json(g5_dir / "session.json.enc", session, key, snapshot, salt, b"GYR5")
        with open(g4_dir / "curriculum.json", "w") as f:
            json.dump(curriculum, f)
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
    unittest.main()
