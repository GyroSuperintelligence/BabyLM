"""
g2_intelligence_eg.py - Thread and message history management

Manages thread/message storage using the genome as canonical source for archived content.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import os
from typing import TYPE_CHECKING

from s1_governance import get_shard_from_uuid, VOID_OP_PAIR, is_void
from s2_information.s2_manifest import get_manifest

# Import from S1 governance
from s1_governance import (
    get_gene_tensors,
    gyration_op,
    byte_to_gyrations,
    gyrations_to_byte,
    build_epigenome_projection,
)

if TYPE_CHECKING:
    from s4_intelligence.g1_intelligence_in import IntelligenceEngine


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
    Manages message storage for threads, using recent windows and genome packs as the sole source of archived content.

    Storage structure:
    agents/<shard>/<agent_uuid>/
    ├── g5_information/
    │   └── session.json             # stays tiny, only thread headers
    └── threads/
        ├── <thread_id>-recent.json  # most recent N messages
    Genome packs are canonical for all archived content.
    """

    def __init__(self, agent_uuid: str, base_path: Optional[str] = None):
        if base_path is None:
            raise ValueError("MessageStore requires explicit base_path to avoid path mismatches.")
        self.agent_uuid = agent_uuid
        self.shard = get_shard_from_uuid(agent_uuid)
        self.base_path = Path(base_path)
        self.thread_dir = self.base_path / "agents" / self.shard / agent_uuid / "threads"
        self.thread_dir.mkdir(parents=True, exist_ok=True)

    def load_recent(self, thread_id: str) -> List[Dict[str, Any]]:
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
        recent_path = self.thread_dir / f"{thread_id}-recent.json"
        atomic_json_write(recent_path, messages)

    def flush_thread(
        self,
        thread_id: str,
        keep_last: int = 250,
        cycle_info: Optional[dict] = None,
        engine: Optional["IntelligenceEngine"] = None,
    ) -> None:
        """
        Trim the recent window, keeping only the last N messages. All archiving is handled by genome packs.
        """
        messages = self.load_recent(thread_id)
        total = len(messages)
        if total == 0 or (keep_last >= total and keep_last >= 0):
            return
        if keep_last == 0:
            to_keep = []
        else:
            to_keep = messages[-keep_last:]
        self.write_recent(thread_id, to_keep)

    def restore_thread(
        self,
        thread_id: str,
        engine: "IntelligenceEngine",
        start_cycle: int,
        num_cycles: int,
    ) -> List[Dict[str, Any]]:
        """
        Restore messages for a thread by reading cycles from genome packs, using deterministic offsets.
        """
        # This assumes a deterministic mapping from cycles to messages (e.g., one message per cycle, or a known encoding)
        # You may need to adapt this logic to your actual encoding scheme
        # For now, this is a placeholder for the new restoration logic
        raise NotImplementedError(
            "Restoration from genome packs must be implemented based on your encoding."
        )
