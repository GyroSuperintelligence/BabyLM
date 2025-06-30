"""
message_store.py - Thread and message history management

Manages thread/message storage using the genome as canonical source for archived content.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import os

from s4_intelligence.g1_intelligence_in import get_shard_from_uuid, IntelligenceEngine
from s2_information.s2_manifest import get_manifest
from s4_intelligence.g2_intelligence_eg import byte_to_gyrations, VOID_OP_PAIR


# Minimal atomic JSON write helper
def atomic_json_write(file_path: Path, data: Union[Dict[str, Any], List[Any]]) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(".tmp")
    backup_path = file_path.with_suffix(".bak")
    try:
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        if file_path.exists():
            try:
                if backup_path.exists():
                    backup_path.unlink()
                os.replace(file_path, backup_path)
            except:
                pass
        os.replace(temp_path, file_path)
    except Exception as e:
        print(f"Error in atomic_json_write: {e}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        raise


ARCHIVE_SHARD_LIMIT = get_manifest().get("archive_shard_limit", 10000)


class MessageStore:
    """
    Manages message storage for threads, using recent windows and archived segments.

    Storage structure:
    agents/<shard>/<agent_uuid>/
    ├── g5_information/
    │   └── session.json             # stays tiny, only thread headers
    └── threads/
        ├── <thread_id>-recent.json  # most recent N messages
        └── <thread_id>-archives.json# list of archived segment records
    """

    def __init__(self, agent_uuid: str, base_path: str = "s2_information"):
        """
        Initialize MessageStore for a specific agent.

        Args:
            agent_uuid: Agent UUID
            base_path: Base path for S2 storage
        """
        self.agent_uuid = agent_uuid
        self.shard = get_shard_from_uuid(agent_uuid)
        self.base_path = Path(base_path)

        # Ensure thread directory exists
        self.thread_dir = self.base_path / "agents" / self.shard / agent_uuid / "threads"
        self.thread_dir.mkdir(parents=True, exist_ok=True)

    def load_recent(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Load the recent message window for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            List of recent messages (empty list if file doesn't exist)
        """
        recent_path = self.thread_dir / f"{thread_id}-recent.json"

        if not recent_path.exists():
            return []

        try:
            with open(recent_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading recent messages: {e}")
            return []

    def write_recent(self, thread_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Write the recent message window for a thread.

        Args:
            thread_id: Thread identifier
            messages: List of messages to write
        """
        recent_path = self.thread_dir / f"{thread_id}-recent.json"
        atomic_json_write(recent_path, messages)

    def _get_archive_path(self, thread_id: str, shard_index: int) -> Path:
        return self.thread_dir / f"{thread_id}-archives-{shard_index:05d}.json"

    def _find_latest_archive_shard(self, thread_id: str) -> int:
        # Find the highest existing archive shard index for this thread
        shard_index = 1
        while self._get_archive_path(thread_id, shard_index).exists():
            shard_index += 1
        return shard_index - 1 if shard_index > 1 else 1

    def append_archive(self, thread_id: str, record: Dict[str, Any]) -> None:
        """
        Append an archive record to the thread's archive list, sharding if needed.
        """
        # Find the latest archive file for this thread
        shard_index = self._find_latest_archive_shard(thread_id)
        current_path = self._get_archive_path(thread_id, shard_index)
        # Load or create
        if current_path.exists():
            with open(current_path) as f:
                archives = json.load(f)
        else:
            archives = []
        # If limit reached, start new shard
        if len(archives) >= ARCHIVE_SHARD_LIMIT:
            shard_index += 1
            current_path = self._get_archive_path(thread_id, shard_index)
            archives = []
        archives.append(record)
        atomic_json_write(current_path, archives)

    def read_archives(self, thread_id: str) -> List[Dict[str, Any]]:
        """
        Read the list of archive records for a thread (across all shards).
        """
        archives = []
        shard_index = 1
        while True:
            path = self._get_archive_path(thread_id, shard_index)
            if not path.exists():
                break
            try:
                with open(path, "r") as f:
                    archives.extend(json.load(f))
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading archives from {path}: {e}")
            shard_index += 1
        return archives

    def flush_thread(
        self,
        thread_id: str,
        keep_last: int = 250,
        cycle_info: Optional[dict] = None,
        engine: Optional[IntelligenceEngine] = None,
    ) -> None:
        """
        Flush old messages from recent to archived, keeping only the last N messages.
        Args:
            thread_id: Thread identifier
            keep_last: Number of recent messages to keep
            cycle_info: dict from engine.get_last_pack_info() (must include pack_uuid, cycle_index_start, cycle_index_end)
            engine: IntelligenceEngine instance (optional, but if provided, will call finalize() before archiving)
        Contract:
            Must be called immediately after engine.finalize() and engine.get_last_pack_info().
            If no new cycles were written (cycle_index_end == cycle_index_start), this is a no-op.
        """
        # Always finalize engine if provided, to flush any pending cycles
        if engine is not None:
            engine.finalize()
            engine._update_session()
            cycle_info = engine.get_last_pack_info()
        # First: if there are no messages (or fewer than keep_last), do nothing.
        messages = self.load_recent(thread_id)
        print(f"[DEBUG] flush_thread: len(messages)={len(messages)}, keep_last={keep_last}")
        if len(messages) <= keep_last:
            print("[DEBUG] flush_thread: Not enough messages to archive.")
            return
        # Only now enforce having valid cycle_info, and bail if no new cycles were written.
        print(f"[DEBUG] flush_thread: cycle_info={cycle_info}")
        if (
            not cycle_info
            or not cycle_info.get("pack_uuid")
            or cycle_info["cycle_index_end"] is None
            or cycle_info["cycle_index_start"] is None
        ):
            print("[DEBUG] flush_thread: Invalid cycle_info.")
            raise ValueError("flush_thread requires valid cycle_info after engine.finalize()")
        if cycle_info["cycle_index_end"] == cycle_info["cycle_index_start"]:
            print("[DEBUG] flush_thread: No new cycles written, skipping archive.")
            return
        # Handle keep_last=0 edge case
        if keep_last == 0:
            to_archive = messages
            to_keep = []
        else:
            to_archive = messages[:-keep_last]
            to_keep = messages[-keep_last:]
        print(f"[DEBUG] flush_thread: archiving {len(to_archive)} messages.")
        if not to_archive:
            print("[DEBUG] flush_thread: Nothing to archive after slicing.")
            return
        start_cycle = cycle_info["cycle_index_start"]
        end_cycle = cycle_info["cycle_index_end"]
        pack_uuid = cycle_info["pack_uuid"]
        num_cycles = end_cycle - start_cycle
        # Calculate per-message cycle/offsets for precise restoration
        message_spans = []  # List[List[Tuple[cycle_index, start, end]]]
        byte_cursor = 0
        total_bytes = b"".join(
            [msg["content"].encode("utf-8") if msg.get("content") else b"" for msg in to_archive]
        )
        total_length = len(total_bytes)
        cycle_size = 24
        # Build a map of which bytes go in which cycle
        for msg in to_archive:
            content_bytes = msg["content"].encode("utf-8") if msg.get("content") else b""
            msg_len = len(content_bytes)
            msg_spans = []
            remaining = msg_len
            while remaining > 0:
                cycle_index = byte_cursor // cycle_size
                in_cycle_offset = byte_cursor % cycle_size
                take = min(remaining, cycle_size - in_cycle_offset)
                msg_spans.append((cycle_index, in_cycle_offset, in_cycle_offset + take))
                byte_cursor += take
                remaining -= take
            message_spans.append(msg_spans)
        preview_text = ""
        if to_archive and to_archive[0].get("content"):
            preview_bytes = to_archive[0]["content"].encode("utf-8")[:30]
            preview_text = preview_bytes.decode("utf-8", errors="ignore")
        archive_record = {
            "id": str(uuid.uuid4()),
            "pack_uuid": pack_uuid,
            "first_cycle": start_cycle,
            "cycles": num_cycles,
            "message_count": len(to_archive),
            "preview": preview_text,
            "created": datetime.utcnow().isoformat(),
            "message_spans": message_spans,
            "messages_metadata": [
                {"id": msg.get("id"), "role": msg.get("role"), "timestamp": msg.get("timestamp")}
                for msg in to_archive
            ],
        }
        self.append_archive(thread_id, archive_record)
        self.write_recent(thread_id, to_keep)

    def restore_segment(
        self, record: Dict[str, Any], engine: IntelligenceEngine
    ) -> List[Dict[str, Any]]:
        """
        Restore messages from an archived segment using only the archive record's info.
        Args:
            record: Archive record with pack info
            engine: IntelligenceEngine instance to use for canonical reading
        Returns:
            List of restored messages with id, role, content, timestamp, artifacts
        """
        pack_uuid = record.get("pack_uuid")
        first_cycle = record.get("first_cycle", 0)
        num_cycles = record.get("cycles", 0)
        messages_metadata = record.get("messages_metadata", [])
        message_spans = record.get("message_spans", [])
        if not pack_uuid or num_cycles == 0:
            return []
        try:
            all_bytes = engine.read_genome_segment(pack_uuid, first_cycle, num_cycles)
            restored_messages = []
            for i, meta in enumerate(messages_metadata):
                spans = message_spans[i] if i < len(message_spans) else []
                msg_bytes = b""
                for cycle_index, start, end in spans:
                    cycle_start = cycle_index * 24
                    msg_bytes += all_bytes[cycle_start + start : cycle_start + end]
                # Strip trailing nulls (void op-pair padding)
                msg_bytes = msg_bytes.rstrip(b"\x00")

                # Also strip trailing bytes that decode to the void op-pair (7, 0)
                # This is a bitwise operation: for each pair of bytes, check if they decode to (7, 0)
                def strip_void_op_pairs(data: bytes) -> bytes:
                    # Remove trailing bytes that decode to VOID_OP_PAIR
                    while len(data) >= 1:
                        try:
                            op1, op2 = byte_to_gyrations(data[-1])
                            if op1 == VOID_OP_PAIR and op2 == VOID_OP_PAIR:
                                data = data[:-1]
                            else:
                                break
                        except Exception:
                            break
                    return data

                msg_bytes = strip_void_op_pairs(msg_bytes)
                try:
                    content_text = msg_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    content_text = msg_bytes.decode("latin-1")
                message = {
                    "id": meta.get("id", str(uuid.uuid4())),
                    "role": meta.get("role", "agent"),
                    "content": content_text,
                    "timestamp": meta.get("timestamp", datetime.utcnow().isoformat()),
                    "artifacts": {},
                }
                restored_messages.append(message)
            restored_messages.sort(key=lambda m: m.get("timestamp", ""))
            return restored_messages
        except Exception as e:
            print(f"Error restoring segment: {e}")
            return []
