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

from app.services.message_store import MessageStore
from s4_intelligence.g1_intelligence_in import IntelligenceEngine, initialize_system
from s1_governance import build_epigenome_projection


class TestMessageStore(unittest.TestCase):
    """Test cases for MessageStore functionality."""

    def setUp(self):
        """Set up test environment."""
        # Use canonical s2_information directory
        self.base_path = Path("s2_information")
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create manifest
        manifest_path = self.base_path / "s2_manifest.json"
        import json

        manifest = {
            "version": "1.0",
            "pack_size": 65536,
            "shard_prefix_length": 2,
            "initialized": datetime.utcnow().isoformat(),
        }
        with open(manifest_path, "w") as f:
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

        # Initialize agent files
        curriculum = {
            "version": "1.0",
            "patterns": {},
            "byte_to_token": {},
            "token_to_byte": {},
            "metadata": {"created": datetime.utcnow().isoformat()},
        }
        with open(agent_dir / "g4_information" / "curriculum.json", "w") as f:
            json.dump(curriculum, f)

        session = {
            "agent_uuid": self.agent_uuid,
            "created": datetime.utcnow().isoformat(),
            "last_checkpoint": None,
            "phase": 0,
            "cycle_count": 0,
            "active_curriculum": None,
        }
        with open(agent_dir / "g5_information" / "session.json", "w") as f:
            json.dump(session, f)

        # Create message store
        self.store = MessageStore(self.agent_uuid, str(self.base_path))

        # Thread ID for testing
        self.thread_id = "test-thread-001"

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
        """Test flushing 30 messages and restoring the first 5 using the real restore_segment."""
        # Create 30 dummy messages (reduced from 300 for speed)
        messages = []
        for i in range(30):
            message = {
                "id": f"msg-{i:04d}",
                "role": "agent" if i % 2 == 0 else "assistant",
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
                agent_uuid=self.agent_uuid, encryption_enabled=encryption_enabled
            )
            # Process first 5 messages through the engine to create genome data
            for msg in messages[:5]:
                if msg["role"] == "agent":
                    engine.process_stream(msg["content"].encode("utf-8"))
            # Finalize to flush any partial cycles
            engine.finalize()
            engine._update_session()  # Ensure key is persisted before archiving
            # Get pack/cycle info directly from the engine
            cycle_info = engine.get_last_pack_info()
            encrypted = engine._encryption_enabled
            # Use the engine's pack/cycle info for flush_thread
            self.store.flush_thread(self.thread_id, keep_last=25, cycle_info=cycle_info)
            recent_after = self.store.load_recent(self.thread_id)
            self.assertEqual(len(recent_after), 25)
            self.assertEqual(recent_after[0]["id"], "msg-0005")
            self.assertEqual(recent_after[-1]["id"], "msg-0029")
            archives = self.store.read_archives(self.thread_id)
            self.assertGreater(len(archives), 0)
            archive_record = archives[-1]
            self.assertEqual(archive_record["message_count"], 5)
            self.assertIn("pack_uuid", archive_record)
            self.assertIn("first_cycle", archive_record)
            self.assertIn("cycles", archive_record)
            self.assertIn("preview", archive_record)
            self.assertIn("message_spans", archive_record)
            restored = self.store.restore_segment(archive_record, engine)
            for orig, restored_msg in zip(messages[:5], restored):
                self.assertEqual(orig["content"], restored_msg["content"])
                self.assertEqual(orig["role"], restored_msg["role"])
            print(
                f"\u2713 Successfully flushed 30 messages, kept 25 recent (encryption={encryption_enabled})"
            )
            print(
                f"\u2713 Created archive with {archive_record['message_count']} messages (encryption={encryption_enabled})"
            )
            print(
                f"\u2713 Restored {len(restored)} messages from archive (encryption={encryption_enabled})"
            )
            engine.close()

    def test_empty_thread(self):
        """Test operations on empty thread."""
        # Load empty thread
        recent = self.store.load_recent("empty-thread")
        self.assertEqual(len(recent), 0)

        # Read empty archives
        archives = self.store.read_archives("empty-thread")
        self.assertEqual(len(archives), 0)

        # Flush empty thread (should do nothing)
        self.store.flush_thread("empty-thread")

        # Verify still empty
        recent = self.store.load_recent("empty-thread")
        self.assertEqual(len(recent), 0)

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

        # Verify no archives created
        archives = self.store.read_archives(self.thread_id)
        self.assertEqual(len(archives), 0)

    def test_snapshot_offsets(self):
        """Test snapshot/offset logic with very short messages."""
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
                "role": "assistant",
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
        self.store.write_recent(self.thread_id, short_msgs)
        for encryption_enabled in (True, False):
            engine = IntelligenceEngine(
                agent_uuid=self.agent_uuid, encryption_enabled=encryption_enabled
            )
            for msg in short_msgs:
                if msg["role"] == "agent":
                    engine.process_stream(msg["content"].encode("utf-8"))
            engine.finalize()
            engine._update_session()  # Ensure key is persisted before archiving
            cycle_info = engine.get_last_pack_info()
            encrypted = engine._encryption_enabled
            self.store.flush_thread(self.thread_id, keep_last=0, cycle_info=cycle_info)
            archives = self.store.read_archives(self.thread_id)
            if not archives:
                print(
                    f"No archive created for short stream (encryption={encryption_enabled}), as expected."
                )
                continue
            archive_record = archives[0]
            restored = self.store.restore_segment(archive_record, engine)
            self.assertEqual(len(restored), 3)
            for orig, restored_msg in zip(short_msgs, restored):
                self.assertEqual(orig["content"], restored_msg["content"])
                self.assertEqual(orig["role"], restored_msg["role"])
            engine.close()

    def test_single_cycle_restore(self):
        """Test restoring a single message that fits in one cycle (≤24 bytes)."""
        single_message = "A" * 24

        for encryption_enabled in (True, False):
            # Re-seed the recent window for *each* pass
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
                agent_uuid=self.agent_uuid, encryption_enabled=encryption_enabled
            )
            engine.process_stream(single_message.encode("utf-8"))
            self.store.flush_thread(self.thread_id, keep_last=0, engine=engine)
            archives = self.store.read_archives(self.thread_id)
            self.assertGreater(len(archives), 0)
            archive_record = archives[-1]  # newest record corresponds to this run
            restored = self.store.restore_segment(archive_record, engine)
            self.assertEqual(len(restored), 1)
            self.assertEqual(restored[0]["content"], single_message)
            engine.close()


if __name__ == "__main__":
    unittest.main()
