import uuid
import datetime
import unicodedata
from typing import cast
from baby.types import FormatMetadata
from baby.information import store_format
from baby.governance import classify_pattern_resonance

try:
    import emoji
except ImportError:
    raise ImportError("Please install the 'emoji' package: pip install emoji")

# Get the full list of emoji characters
all_emoji = list(emoji.EMOJI_DATA.keys())
pack_size = 256
num_packs = (len(all_emoji) + pack_size - 1) // pack_size


def make_format_name(pack_num: int) -> str:
    return f"emoji_pack_{pack_num+1}"


def make_description(pack_num: int, total_packs: int) -> str:
    return f"Emoji curriculum pack {pack_num+1} of {total_packs} (Unicode emoji set)"


for pack_num in range(num_packs):
    from baby.information import get_memory_preferences
    base_memories_dir = "memories"
    prefs = get_memory_preferences(base_memories_dir)
    start = pack_num * pack_size
    end = min(start + pack_size, len(all_emoji))
    emoji_slice = all_emoji[start:end]
    patterns = []
    for i, char in enumerate(emoji_slice):
        # Handle single and multi-codepoint emoji for name
        if len(char) == 1:
            try:
                name = unicodedata.name(char)
            except ValueError:
                name = emoji.EMOJI_DATA[char].get("en", "EMOJI") if "en" in emoji.EMOJI_DATA[char] else "EMOJI"
        else:
            try:
                name = " + ".join(unicodedata.name(c) for c in char)
            except ValueError:
                name = emoji.EMOJI_DATA[char].get("en", "EMOJI") if "en" in emoji.EMOJI_DATA[char] else "EMOJI"
        # Use the category of the first codepoint as type
        category = unicodedata.category(char[0]) if char else "So"
        patterns.append(
            {
                "index": i,
                "character": char,
                "description": name,
                "type": category,
                "count": 0,
                "first_cycle": None,
                "last_cycle": None,
                "gyration_feature": classify_pattern_resonance(i),
                "confidence": 0.0,
            }
        )
    timestamp = datetime.datetime.now().isoformat()
    format_data = {
        "format_uuid": str(uuid.uuid4()),
        "format_name": make_format_name(pack_num),
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
            "description": make_description(pack_num, num_packs),
            "tags": ["emoji", "foundational", "curriculum", f"pack_{pack_num+1}"],
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
    format_uuid = store_format(cast(FormatMetadata, format_data), prefs, base_memories_dir)
    print("\n��😃 Emoji Curriculum Format Learned! 😃🎉")
    print(f"🆔  UUID: {format_uuid}")
    print("📦  Location: memories/public/formats/")
    print(f"😀  Pack: {pack_num+1} of {num_packs} | Emojis: {len(patterns)}")
    print(f"🏷️  Format Name: {make_format_name(pack_num)}")
    print("📝  Each entry includes: character, description, type, gyration_feature, and stats.")
    print("✨  Ready for learning, encryption, and curriculum composition! ✨\n")
