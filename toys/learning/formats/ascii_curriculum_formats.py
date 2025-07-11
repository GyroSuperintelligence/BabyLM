
# toys/learning/formats/ascii_curriculum.py
# This toy is used to generate the ASCII curriculum format.

import uuid
import datetime
import unicodedata
from typing import cast
from baby.types import FormatMetadata
from baby.information import store_format
from baby.governance import classify_pattern_resonance

FORMAT_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000000")

# Generate ASCII 256 characters with special handling for control chars
ascii_chars = []
for i in range(256):
    if i < 32 or i == 127:  # Control characters
        char = chr(i)
        try:
            name = unicodedata.name(char)
        except ValueError:
            name = f"CONTROL CHARACTER {i}" if i < 32 else "DELETE"
        category = unicodedata.category(char)
        ascii_chars.append(
            {"character": f"\\x{i:02x}" if not char.isprintable() else char, "description": name, "type": category}
        )
    else:
        char = chr(i)
        name = unicodedata.name(char, f"ASCII {i}")
        category = unicodedata.category(char)
        ascii_chars.append({"character": char, "description": name, "type": category})

# Build the patterns list with proper resonance classification
patterns = [
    {
        "index": i,
        "character": entry["character"],
        "description": entry["description"],
        "type": entry["type"],
        "count": 0,
        "first_cycle": None,
        "last_cycle": None,
        "gyration_feature": classify_pattern_resonance(i),
        "confidence": 0.0,
    }
    for i, entry in enumerate(ascii_chars)
]

# Build the format metadata
timestamp = datetime.datetime.now().isoformat()
format_data = {
    "format_uuid": str(uuid.uuid5(FORMAT_NAMESPACE, "ascii_curriculum")),
    "format_name": "ascii_256",
    "format_version": "1.0.0",
    "stability": "stable",
    "compatibility": {
        "min_format_version": "1.0.0",
        "max_format_version": "1.0.0",
        "depends_on": [],
        "conflicts_with": [],
    },
    "metadata": {
        "author": "curriculum_init",
        "description": "Complete ASCII 256 character set mapping with Unicode names and types",
        "tags": ["ascii", "foundational", "curriculum"],
        "created_at": timestamp,
        "last_updated": timestamp,
        "usage_count": 0,
        "validation_status": "verified",
    },
    "cgm_policies": {
        "governance": {"operation": "L0", "bits": [0, 7], "policy": "traceability"},
        "information": {"operation": "LI", "bits": [1, 6], "policy": "variety"},
        "inference": {"operation": "FG", "bits": [2, 5], "policy": "accountability"},
        "intelligence": {"operation": "BG", "bits": [3, 4], "policy": "integrity"},
    },
    "patterns": patterns,
}

# Store the format using the system's helper
if __name__ == "__main__":
    from baby.information import get_memory_preferences

    base_memories_dir = "memories"
    prefs = get_memory_preferences(base_memories_dir)
    format_uuid = store_format(cast(FormatMetadata, format_data), prefs, base_memories_dir)
    print("\n🎉✅ ASCII Curriculum Format Learned! ✅🎉")
    print(f"🆔  UUID: {format_uuid}")
    print("📦  Location: memories/public/formats/")
    print("🔤  Characters: 256 (ASCII + Unicode extensions)")
    print("📝  Each entry includes: character, description, type, gyration_feature, and stats.")
    print("✨  Ready for learning, encryption, and curriculum composition! ✨\n")
