import os
import sys
import uuid
from pathlib import Path
import json

# Use same namespace as ascii/emoji/math
CURRICULUM_NAMESPACE = uuid.UUID("b6e0e1e2-1c2d-4e3f-8a9b-123456789abc")
LEMMA_FORMAT_NAME = "wordnet_lemmas_v1.0.0"
SYNSET_FORMAT_NAME = "wordnet_synsets_v1.0.0"
LEMMA_FORMAT_UUID = str(uuid.uuid5(CURRICULUM_NAMESPACE, LEMMA_FORMAT_NAME))
SYNSET_FORMAT_UUID = str(uuid.uuid5(CURRICULUM_NAMESPACE, SYNSET_FORMAT_NAME))

# Add project root to path to import from 'baby'
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    print("NLTK not found. Please run 'pip install nltk'")
    sys.exit(1)

from baby.information import (
    get_memory_preferences,
    shard_path,
    update_registry,
    json_dumps,
    atomic_write,
)
from baby.types import FormatMetadata, PatternMetadata
from typing import cast, List

MEMORIES_DIR = Path("memories")

def _generate_base_format_dict(format_uuid: str, format_name: str, description: str) -> dict:
    """Creates the boilerplate for a FormatMetadata file."""
    patterns: List[PatternMetadata] = []
    for i in range(256):
        p_meta: PatternMetadata = {
            "index": i, "character": None, "description": "Unassigned", "type": "Cn",
            "count": 0, "first_cycle": None, "last_cycle": None,
            "gyration_feature": "identity", # Placeholder
            "confidence": 0.0,
        }
        patterns.append(p_meta)

    format_data = {
        "format_uuid": format_uuid,
        "format_name": format_name,
        "format_version": "1.0.0",
        "stability": "stable",
        "compatibility": {"min_format_version": "1.0.0"},
        "metadata": {"description": description, "author": "curriculum_generator"},
        "cgm_policies": {
            "governance": {"operation": "L0", "bits": [0, 7], "policy": "traceability"},
            "information": {"operation": "LI", "bits": [1, 6], "policy": "variety"},
            "inference": {"operation": "FG", "bits": [2, 5], "policy": "accountability"},
            "intelligence": {"operation": "BG", "bits": [3, 4], "policy": "integrity"},
        },
        "patterns": patterns
    }
    return format_data

def save_format_file(format_data: dict, prefs: dict):
    """Saves a format file to the correct sharded location and updates registries."""
    format_uuid = format_data["format_uuid"]
    format_name = format_data["format_name"]
    formats_dir = MEMORIES_DIR / "public/formats"
    format_shard_path = shard_path(formats_dir, format_uuid, prefs)
    format_shard_path.mkdir(parents=True, exist_ok=True)
    file_path = format_shard_path / f"format-{format_name}-{format_uuid}.json"
    atomic_write(file_path, json_dumps(format_data).encode('utf-8'))
    print(f"  Saved format to: {file_path}")
    update_registry(format_shard_path, format_uuid)
    print(f"  Updated registry for shard: {format_shard_path.name}")

def get_lemma_definition(lemma_name: str) -> str:
    synsets = wn.synsets(lemma_name)
    if not synsets:
        return f"No definition found for lemma: {lemma_name}"
    return synsets[0].definition() # type: ignore[reportOptionalMemberAccess]

def create_lemma_format(prefs: dict):
    print(f"\nCreating Lemma Format (UUID: {LEMMA_FORMAT_UUID})...")
    unique_lemmas = sorted(list(wn.all_lemma_names()))
    print(f"  Found {len(unique_lemmas)} unique lemmas. Mapping all.")
    format_dict = _generate_base_format_dict(
        LEMMA_FORMAT_UUID,
        "wordnet_lemmas_v1",
        "Maps all unique WordNet lemmas to pattern indices with their primary definitions."
    )
    patterns = []
    for i, lemma_name in enumerate(unique_lemmas):
        definition = get_lemma_definition(lemma_name)
        patterns.append({
            "index": i % 256,  # cycles every 256
            "character": lemma_name,
            "description": definition,
            "type": "WNL",
            "count": 0,
            "first_cycle": None,
            "last_cycle": None,
            "gyration_feature": "identity",
            "confidence": 0.0,
        })
    format_dict["patterns"] = patterns
    print(f"  Sample entries:")
    for i in range(min(3, len(patterns))):
        char = patterns[i]["character"]
        desc = patterns[i]["description"]
        print(f"    [{i}] '{char}': {desc[:60]}...")
    save_format_file(format_dict, prefs)

def create_synset_format(prefs: dict):
    print(f"\nCreating Synset Format (UUID: {SYNSET_FORMAT_UUID})...")
    all_synsets = list(wn.all_synsets())
    print(f"  Found {len(all_synsets)} synsets. Mapping all.")
    format_dict = _generate_base_format_dict(
        SYNSET_FORMAT_UUID,
        "wordnet_synsets_v1",
        "Maps all WordNet synsets to pattern indices with their definitions."
    )
    patterns = []
    for i, synset in enumerate(all_synsets):
        patterns.append({
            "index": i % 256,  # cycles every 256
            "character": synset.name(),
            "description": synset.definition(),  # type: ignore[reportOptionalMemberAccess]
            "type": "WNS",
            "count": 0,
            "first_cycle": None,
            "last_cycle": None,
            "gyration_feature": "identity",
            "confidence": 0.0,
        })
    format_dict["patterns"] = patterns
    print(f"  Sample entries:")
    for i in range(min(3, len(patterns))):
        char = patterns[i]["character"]
        desc = patterns[i]["description"]
        print(f"    [{i}] '{char}': {desc[:60]}...")
    save_format_file(format_dict, prefs)

def main():
    try:
        wn.all_synsets()
    except Exception:
        print("Downloading NLTK WordNet data... (this happens once)")
        import nltk
        nltk.download('wordnet')
        print("Download complete.")

    prefs = get_memory_preferences(str(MEMORIES_DIR))
    print("=== WordNet Format Generation ===")
    create_lemma_format(prefs)
    create_synset_format(prefs)
    unique_lemmas = sorted(list(wn.all_lemma_names()))
    all_synsets = list(wn.all_synsets())
    print("\n🎉✅ WordNet Curriculum Formats Learned! ✅🎉")
    print(f"🆔  Lemma UUID: {LEMMA_FORMAT_UUID}")
    print(f"🆔  Synset UUID: {SYNSET_FORMAT_UUID}")
    print("📦  Location: memories/public/formats/")
    print(f"🔤  Lemmas: {len(unique_lemmas)} | Synsets: {len(all_synsets)}")
    print("📝  Each entry includes: character, description, type, gyration_feature, and stats.")
    print("✨  Ready for learning, encryption, and curriculum composition! ✨\n")
    print("Next step: Run the ingestion script to train on WordNet data.")

if __name__ == "__main__":
    main()
