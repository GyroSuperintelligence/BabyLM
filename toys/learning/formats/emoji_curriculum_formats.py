import uuid
import datetime
import unicodedata
from baby.types import FormatMetadata
from baby.information import store_format, get_memory_preferences, shard_path
from baby.governance import classify_pattern_resonance

try:
    import emoji
except ImportError:
    raise ImportError("Please install the 'emoji' package: pip install emoji")

CURRICULUM_NAMESPACE = uuid.UUID("b6e0e1e2-1c2d-4e3f-8a9b-123456789abc")
EMOJI_CURRICULUM_NAME = "emoji_curriculum_v1.0.0"
EMOJI_CURRICULUM_UUID = str(uuid.uuid5(CURRICULUM_NAMESPACE, EMOJI_CURRICULUM_NAME))

# Gather all emoji characters
all_emoji = list(emoji.EMOJI_DATA.keys())
patterns = []
for i, char in enumerate(all_emoji):
    name = emoji.EMOJI_DATA[char].get('en', char)
    patterns.append({
        "index": i % 256,  # cycles every 256
        "character": char,  # actual emoji character
        "description": name,
        "type": "EMOJI",
        "count": 0,
        "first_cycle": None,
        "last_cycle": None,
        "gyration_feature": classify_pattern_resonance(i),
        "confidence": 0.0,
    })

timestamp = datetime.datetime.now().isoformat()
format_data = {
    "format_uuid": EMOJI_CURRICULUM_UUID,
    "format_name": "emoji_all",
    "format_version": "1.0.0",
    "stability": "stable",
    "compatibility": {"min_format_version": "1.0.0"},
    "metadata": {
        "description": "All emoji characters.",
        "author": "curriculum_generator",
        "timestamp": timestamp
    },
    "cgm_policies": {},
    "patterns": patterns,
}

if __name__ == "__main__":
    import os
    base_memories_dir = "memories"
    prefs = get_memory_preferences(base_memories_dir)
    formats_dir = os.path.join(base_memories_dir, "public/formats")
    from pathlib import Path
    format_shard = shard_path(Path(formats_dir), EMOJI_CURRICULUM_UUID, prefs)
    format_shard.mkdir(parents=True, exist_ok=True)
    file_path = format_shard / f"format-emoji_all-{EMOJI_CURRICULUM_UUID}.json"
    with open(file_path, "w") as f:
        import json
        f.write(json.dumps(format_data))
    print("\n🎉✅ Emoji Curriculum Format Learned! ✅🎉")
    print(f"🆔  UUID: {EMOJI_CURRICULUM_UUID}")
    print(f"📦  Location: {file_path}")
    print(f"😀  Emojis: {len(patterns)} (Unicode emoji set)")
    print("📝  Each entry includes: character, description, type, gyration_feature, and stats.")
    print("✨  Ready for learning, encryption, and curriculum composition! ✨\n")
