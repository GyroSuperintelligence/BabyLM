import os
import json
from typing import Any, Dict

class G5_Memory:
    """
    Immunity Memory (G5): Session snapshots for an agent, one JSON file per session UUID.
    """
    def __init__(self, agent_uuid: str):
        self.agent_uuid = agent_uuid
        self.base_dir = os.path.join("memory", "agents", agent_uuid, "g5_memory")
        self._ensure_dirs()

    def _ensure_dirs(self):
        os.makedirs(self.base_dir, exist_ok=True)

    def load_session(self, session_uuid: str) -> Dict[str, Any]:
        path = os.path.join(self.base_dir, f"session-{session_uuid}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Session {session_uuid} not found for agent {self.agent_uuid}")
        with open(path, "r") as f:
            return json.load(f)

    def save_session(self, session_uuid: str, data: Dict[str, Any]):
        path = os.path.join(self.base_dir, f"session-{session_uuid}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
