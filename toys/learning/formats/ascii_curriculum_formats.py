# toys/learning/formats/ascii_curriculum_formats.py
# This script generates the full 256-character ASCII curriculum format.

import uuid
import datetime
import unicodedata
from typing import cast
from baby.types import FormatMetadata
from baby.information import store_format, get_memory_preferences, shard_path
from baby.governance import classify_pattern_resonance
from pathlib import Path
import os
import json

# Namespace and UUID config
CURRICULUM_NAMESPACE = uuid.UUID("b6e0e1e2-1c2d-4e3f-8a9b-123456789abc")
ASCII_CURRICULUM_NAME = "ascii_curriculum_v1.0.0"
ASCII_CURRICULUM_UUID = str(uuid.uuid5(CURRICULUM_NAMESPACE, ASCII_CURRICULUM_NAME))

# Build the full 256-character ASCII table
patterns = []
for i in range(256):
    char = chr(i)
    try:
        name = unicodedata.name(char)
        description = name
    except ValueError:
        if i < 32 or i == 127:
            description = f"CONTROL CHARACTER {i}"
        else:
            description = f"ASCII {i}"
    patterns.append({
        "index": i,
        "character": char,  # *** This is now ALWAYS the real character ***
        "description": description,
        "type": "ASCII",
        "count": 0,
        "first_cycle": None,
        "last_cycle": None,
        "gyration_feature": classify_pattern_resonance(i),
        "confidence": 0.0,
    })

timestamp = datetime.datetime.now().isoformat()
format_data = {
    "format_uuid": ASCII_CURRICULUM_UUID,
    "format_name": "ascii_256",
    "format_version": "1.0.0",
    "stability": "stable",
    "compatibility": {"min_format_version": "1.0.0"},
    "metadata": {
        "description": "Full 256-character ASCII table. 'character' is the real character.",
        "author": "curriculum_generator",
        "timestamp": timestamp
    },
    "cgm_policies": {},
    "patterns": patterns,
}

if __name__ == "__main__":
    base_memories_dir = "memories"
    prefs = get_memory_preferences(base_memories_dir)
    formats_dir = os.path.join(base_memories_dir, "public/formats")
    format_shard = shard_path(Path(formats_dir), ASCII_CURRICULUM_UUID, prefs)
    format_shard.mkdir(parents=True, exist_ok=True)
    file_path = format_shard / f"format-ascii_256-{ASCII_CURRICULUM_UUID}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(format_data, f, ensure_ascii=False, indent=2)
    print("\n🎉✅ ASCII Curriculum Format Learned! ✅🎉")
    print(f"🆔  UUID: {ASCII_CURRICULUM_UUID}")
    print(f"📦  Location: {file_path}")
    print("🔤  Characters: 256 (raw bytes, all valid ASCII)")
    print("📝  Each entry includes: character, description, type, gyration_feature, and stats.")
    print("✨  Ready for learning, encryption, and curriculum composition! ✨\n")
