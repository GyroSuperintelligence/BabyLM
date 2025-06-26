"""
g2_memory.py - Epigenetic Memory (G2)

Agent-centric, pack-based, RLE-compressed navigation history.
Replaces the former NavigationLog with a scalable, robust solution.
"""

import os
import json
import hashlib
import threading
from typing import Iterator, Tuple, List, Dict, Any, Optional

# RLE compression constants
_RLE_FLAG_NIBBLE = 0xE  # High nibble 1110 marks RLE run
_IDENTITY_EVENT = 0x00  # Double-identity event (compressible)

class G2_Memory:
    """
    Epigenetic Memory: Complete, lossless navigation history for an agent.
    Stores the complete history of navigation events in a pack-based structure,
    with RLE compression for identity events.
    """
    PACK_SIZE_LIMIT = 16 * 1024 * 1024  # 16 MB per pack

    def __init__(self, agent_uuid: str):
        self.agent_uuid = agent_uuid
        self.base_dir = os.path.join("memory", "agents", agent_uuid, "g2_memory")
        self.manifest_path = os.path.join(self.base_dir, "manifest.json")
        self.active_path = os.path.join(self.base_dir, "active.dat")
        self._lock = threading.RLock()
        self._identity_run_count = 0
        self._total_events = 0
        self._is_dirty = False
        os.makedirs(self.base_dir, exist_ok=True)
        self.manifest = self._load_manifest()
        if not os.path.exists(self.active_path):
            open(self.active_path, "wb").close()

    def append(self, id0_code: int, id1_code: int) -> None:
        """
        Append a navigation event with RLE compression.
        Args:
            id0_code: Operator code (0-3) for id_0 tensor
            id1_code: Operator code (0-3) for id_1 tensor
        """
        if not (0 <= id0_code <= 3 and 0 <= id1_code <= 3):
            raise ValueError(f"Operator codes must be 0-3, got {id0_code}, {id1_code}")
        packed_byte = (id1_code << 4) | id0_code
        with self._lock:
            if packed_byte == 0x00:
                self._identity_run_count += 1
                if self._identity_run_count == 17:
                    self._flush_identity_run()
            else:
                self._flush_identity_run()
                self._append_raw_byte(packed_byte)
            self._total_events += 1
            self._is_dirty = True
            if os.path.getsize(self.active_path) >= self.PACK_SIZE_LIMIT:
                self.rotate_pack()

    def _flush_identity_run(self) -> None:
        if self._identity_run_count == 0:
            return
        if self._identity_run_count == 1:
            self._append_raw_byte(0x00)
        else:
            count_nibble = (self._identity_run_count - 2) & 0x0F
            rle_byte = (_RLE_FLAG_NIBBLE << 4) | count_nibble
            self._append_raw_byte(rle_byte)
        self._identity_run_count = 0

    def _append_raw_byte(self, byte_val: int) -> None:
        with open(self.active_path, "ab") as f:
            f.write(bytes([byte_val]))

    def rotate_pack(self) -> None:
        with self._lock:
            self._flush_identity_run()
            packs = self.manifest.get("packs", [])
            next_index = len(packs) + 1
            pack_name = f"pack-{next_index:05d}.dat"
            pack_path = os.path.join(self.base_dir, pack_name)
            event_start = 0
            if packs:
                last_pack = packs[-1]
                event_start = last_pack.get("event_start", 0) + last_pack.get("event_count", 0)
            with open(self.active_path, "rb") as f:
                pack_data = f.read()
            sha256 = hashlib.sha256(pack_data).hexdigest()
            event_count = self._count_events(pack_data)
            os.rename(self.active_path, pack_path)
            pack_info = {
                "file": pack_name,
                "event_start": event_start,
                "event_count": event_count,
                "sha256": sha256
            }
            packs.append(pack_info)
            self.manifest["packs"] = packs
            self.manifest["total_events"] = self._total_events
            self._save_manifest()
            open(self.active_path, "wb").close()

    def _count_events(self, data: bytes) -> int:
        count = 0
        i = 0
        while i < len(data):
            byte = data[i]
            i += 1
            high_nibble = (byte >> 4) & 0x0F
            low_nibble = byte & 0x0F
            if high_nibble == _RLE_FLAG_NIBBLE:
                count += low_nibble + 2
            else:
                count += 1
        return count

    def iter_steps(self, reverse: bool = False) -> Iterator[Tuple[int, int]]:
        """
        Iterate over navigation events as (id0_code, id1_code) tuples.
        Args:
            reverse: If True, iterate in reverse order
        Yields:
            Tuples of (id0_code, id1_code)
        """
        with self._lock:
            self._flush_identity_run()
            events = []
            for pack in self.manifest.get("packs", []):
                pack_path = os.path.join(self.base_dir, pack["file"])
                with open(pack_path, "rb") as f:
                    events.extend(self._decode_events(f.read()))
            if os.path.exists(self.active_path):
                with open(self.active_path, "rb") as f:
                    events.extend(self._decode_events(f.read()))
            if reverse:
                events.reverse()
            for event in events:
                yield event

    def _decode_events(self, data: bytes) -> List[Tuple[int, int]]:
        events = []
        i = 0
        while i < len(data):
            byte = data[i]
            i += 1
            high_nibble = (byte >> 4) & 0x0F
            low_nibble = byte & 0x0F
            if high_nibble == _RLE_FLAG_NIBBLE:
                run_length = low_nibble + 2
                for _ in range(run_length):
                    events.append((0, 0))
            else:
                id0_code = byte & 0x0F
                id1_code = (byte >> 4) & 0x0F
                events.append((id0_code, id1_code))
        return events

    @property
    def event_count(self) -> int:
        return self._total_events

    def shutdown(self) -> None:
        with self._lock:
            self._flush_identity_run()
            if os.path.getsize(self.active_path) > 0:
                self.rotate_pack()

    def _load_manifest(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r") as f:
                    manifest = json.load(f)
                    self._total_events = manifest.get("total_events", 0)
                    return manifest
            except Exception as e:
                print(f"Error loading manifest: {e}")
        return {
            "version": "1.0",
            "total_events": 0,
            "packs": []
        }

    def _save_manifest(self) -> None:
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def fork(self, new_agent_uuid: str) -> "G2_Memory":
        self.shutdown()
        new_memory = G2_Memory(new_agent_uuid)
        for pack in self.manifest.get("packs", []):
            src_path = os.path.join(self.base_dir, pack["file"])
            dst_path = os.path.join(new_memory.base_dir, pack["file"])
            with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
                dst.write(src.read())
        new_memory.manifest = self.manifest.copy()
        new_memory._total_events = self._total_events
        new_memory._save_manifest()
        return new_memory