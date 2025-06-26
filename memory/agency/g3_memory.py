import os
import json
from typing import List, Dict, Any

class G3_Memory:
    """
    Structural Memory (G3): Append-only, pack-based inference/event log for an agent.
    Manifest includes inference stats.
    """
    PACK_SIZE_LIMIT = 16 * 1024 * 1024  # 16 MB per pack

    def __init__(self, agent_uuid: str):
        self.agent_uuid = agent_uuid
        self.base_dir = os.path.join("memory", "agents", agent_uuid, "g3_memory")
        self.manifest_path = os.path.join(self.base_dir, "manifest.json")
        self.active_path = os.path.join(self.base_dir, "active.dat")
        self._ensure_dirs()
        self.manifest = self.load_manifest()

    def _ensure_dirs(self):
        os.makedirs(self.base_dir, exist_ok=True)

    def append_event(self, event_bytes: bytes):
        with open(self.active_path, "ab") as f:
            f.write(event_bytes)
        if os.path.getsize(self.active_path) >= self.PACK_SIZE_LIMIT:
            self.rotate_pack()

    def rotate_pack(self):
        packs = [p for p in os.listdir(self.base_dir) if p.startswith("pack-") and p.endswith(".dat")]
        next_index = len(packs) + 1
        pack_name = f"pack-{next_index:05d}.dat"
        pack_path = os.path.join(self.base_dir, pack_name)
        os.rename(self.active_path, pack_path)
        # Update manifest
        pack_info = {
            "file": pack_name,
            # Add more metadata as needed (e.g., event_start, event_count, sha256)
        }
        self.manifest.setdefault("packs", []).append(pack_info)
        self.save_manifest()
        # Create new active file
        open(self.active_path, "wb").close()

    def load_manifest(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        # Enhanced manifest with inference_stats
        return {
            "version": "1.0",
            "inference_stats": {
                "total_patterns_detected": 0,
                "unique_signatures": 0,
                "last_gyro_state": None
            },
            "packs": []
        }

    def save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def update_inference_stats(self, total_patterns: int, unique_signatures: int, last_gyro_state: str):
        self.manifest["inference_stats"] = {
            "total_patterns_detected": total_patterns,
            "unique_signatures": unique_signatures,
            "last_gyro_state": last_gyro_state
        }
        self.save_manifest()

    def read_events(self) -> List[bytes]:
        events = []
        for pack in self.manifest.get("packs", []):
            pack_path = os.path.join(self.base_dir, pack["file"])
            with open(pack_path, "rb") as f:
                events.append(f.read())
        if os.path.exists(self.active_path):
            with open(self.active_path, "rb") as f:
                events.append(f.read())
        return events
