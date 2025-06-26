import os
import json
from typing import Any, Dict

class G4_Memory:
    """
    Somatic Memory (G4): Curriculum/learning materials for an agent, stored as JSON documents.
    """
    def __init__(self, agent_uuid: str):
        self.agent_uuid = agent_uuid
        self.base_dir = os.path.join("memory", "agents", agent_uuid, "g4_memory")
        self._ensure_dirs()

    def _ensure_dirs(self):
        os.makedirs(self.base_dir, exist_ok=True)

    def load_curriculum(self, curriculum_id: str) -> Dict[str, Any]:
        path = os.path.join(self.base_dir, f"{curriculum_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Curriculum {curriculum_id} not found for agent {self.agent_uuid}")
        with open(path, "r") as f:
            return json.load(f)

    def save_curriculum(self, curriculum_id: str, data: Dict[str, Any]):
        path = os.path.join(self.base_dir, f"{curriculum_id}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
