import uuid
import datetime
import unicodedata
from typing import cast
from baby.types import FormatMetadata
from baby.information import store_format
from baby.governance import classify_pattern_resonance

# Use the same namespace as ascii/emoji
CURRICULUM_NAMESPACE = uuid.UUID("b6e0e1e2-1c2d-4e3f-8a9b-123456789abc")
MATH_CURRICULUM_NAME = "math_curriculum_v1.0.0"
MATH_CURRICULUM_UUID = str(uuid.uuid5(CURRICULUM_NAMESPACE, MATH_CURRICULUM_NAME))

# Exclude ASCII codepoints
ascii_codepoints = set(range(256))

# Collect all Unicode characters with category 'Sm' (Symbol, Math), excluding ASCII
all_math_symbols = []
for codepoint in range(0x0000, 0x110000):  # Unicode range
    if codepoint in ascii_codepoints:
        continue
    char = chr(codepoint)
    if unicodedata.category(char) == "Sm":
        all_math_symbols.append(char)

patterns = []
for i, char in enumerate(all_math_symbols):
    try:
        name = unicodedata.name(char)
    except ValueError:
        name = "UNKNOWN"
    category = unicodedata.category(char)
    patterns.append({
        "index": i % 256,  # cycles every 256
        "character": char,
        "description": name,
        "type": category,
        "count": 0,
        "first_cycle": None,
        "last_cycle": None,
        "gyration_feature": classify_pattern_resonance(i),
        "confidence": 0.0,
    })

timestamp = datetime.datetime.now().isoformat()
format_data = {
    "format_uuid": MATH_CURRICULUM_UUID,
    "format_name": "math_all",
    "format_version": "1.0.0",
    "stability": "stable",
    "compatibility": {"min_format_version": "1.0.0"},
    "metadata": {
        "description": "All Unicode math symbols (excluding ASCII).",
        "author": "curriculum_generator",
        "timestamp": timestamp,
    },
    "cgm_policies": {},
    "patterns": patterns,
}

if __name__ == "__main__":
    from baby.information import get_memory_preferences, shard_path
    import os
    base_memories_dir = "memories"
    prefs = get_memory_preferences(base_memories_dir)
    formats_dir = os.path.join(base_memories_dir, "public/formats")
    from pathlib import Path
    format_shard = shard_path(Path(formats_dir), MATH_CURRICULUM_UUID, prefs)
    format_shard.mkdir(parents=True, exist_ok=True)
    file_path = format_shard / f"format-math_all-{MATH_CURRICULUM_UUID}.json"
    with open(file_path, "w") as f:
        import json
        f.write(json.dumps(format_data))
    print("\n🎉✅ Math Curriculum Format Learned! ✅🎉")
    print(f"🆔  UUID: {MATH_CURRICULUM_UUID}")
    print(f"📦  Location: {file_path}")
    print(f"🧮  Symbols: {len(patterns)} (Unicode math symbols, non-ASCII)")
    print("📝  Each entry includes: character, description, type, gyration_feature, and stats.")
    print("✨  Ready for learning, encryption, and curriculum composition! ✨\n")
