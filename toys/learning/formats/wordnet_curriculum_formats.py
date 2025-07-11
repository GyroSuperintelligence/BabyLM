import os
import sys
import uuid
from pathlib import Path
import json

# Add project root to path to import from 'baby'
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    print("NLTK not found. Please run 'pip install nltk'")
    sys.exit(1)

# Import ONLY the necessary file/registry helpers. No engines.
from baby.information import (
    get_memory_preferences,
    shard_path,
    update_registry,
    json_dumps,
    atomic_write,
)
from baby.types import FormatMetadata, PatternMetadata
from typing import cast, List

# --- Configuration ---
MEMORIES_DIR = Path("memories")
LEMMA_FORMAT_UUID = "0003-wordnet-lemmas-v1-0000-000000000000"
SYNSET_FORMAT_UUID = "0004-wordnet-synsets-v1-0000-000000000000"

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
    formats_dir = MEMORIES_DIR / "public/formats"
    format_shard_path = shard_path(formats_dir, format_uuid, prefs)
    
    # Ensure directory exists
    format_shard_path.mkdir(parents=True, exist_ok=True)
    
    file_path = format_shard_path / f"format-{format_uuid}.json"
    
    # Use atomic write for safety
    atomic_write(file_path, json_dumps(format_data).encode('utf-8'))
    
    print(f"  Saved format to: {file_path}")
    
    # Update the registries
    update_registry(format_shard_path, format_uuid)
    print(f"  Updated registry for shard: {format_shard_path.name}")

def get_lemma_definition(lemma_name: str) -> str:
    """Get the definition of the most common synset containing this lemma."""
    synsets = wn.synsets(lemma_name)
    if not synsets:
        return f"No definition found for lemma: {lemma_name}"
    
    # Use the first synset (most common sense)
    return synsets[0].definition()  # type: ignore[attr-defined]

def create_lemma_format(prefs: dict):
    """Generates the WordNet Lemma format file."""
    print(f"\nCreating Lemma Format (UUID: {LEMMA_FORMAT_UUID})...")
    
    # 1. Get unique lemma names
    unique_lemmas = sorted(list(wn.all_lemma_names()))
    print(f"  Found {len(unique_lemmas)} unique lemmas. Using the first 256.")

    # 2. Get base format structure
    format_dict = _generate_base_format_dict(
        LEMMA_FORMAT_UUID,
        "wordnet_lemmas_v1",
        "Maps the first 256 unique WordNet lemmas to pattern indices with their primary definitions."
    )
    
    # 3. Populate patterns with lemmas and their definitions
    for i in range(256):
        if i < len(unique_lemmas):
            lemma_name = unique_lemmas[i]
            definition = get_lemma_definition(lemma_name)
            format_dict["patterns"][i].update({
                "character": lemma_name,
                "description": definition,
                "type": "WNL"
            })
    
    print(f"  Sample entries:")
    for i in range(min(3, len(unique_lemmas))):
        char = format_dict["patterns"][i]["character"]
        desc = format_dict["patterns"][i]["description"]
        print(f"    [{i}] '{char}': {desc[:60]}...")
    
    # 4. Save the file
    save_format_file(format_dict, prefs)

def create_synset_format(prefs: dict):
    """Generates the WordNet Synset format file."""
    print(f"\nCreating Synset Format (UUID: {SYNSET_FORMAT_UUID})...")
    
    # 1. Get all synsets
    all_synsets = list(wn.all_synsets())
    print(f"  Found {len(all_synsets)} synsets. Using the first 256.")

    # 2. Get base format structure
    format_dict = _generate_base_format_dict(
        SYNSET_FORMAT_UUID,
        "wordnet_synsets_v1",
        "Maps the first 256 WordNet synsets to pattern indices with their definitions."
    )

    # 3. Populate patterns with synsets and their definitions
    for i in range(256):
        if i < len(all_synsets):
            synset = all_synsets[i]  # type: ignore
            format_dict["patterns"][i].update({
                "character": synset.name(),
                "description": synset.definition(),
                "type": "WNS"
            })

    print(f"  Sample entries:")
    for i in range(min(3, len(all_synsets))):
        char = format_dict["patterns"][i]["character"] 
        desc = format_dict["patterns"][i]["description"]
        print(f"    [{i}] '{char}': {desc[:60]}...")

    # 4. Save the file
    save_format_file(format_dict, prefs)

def main():
    # --- NLTK Setup ---
    try:
        wn.all_synsets()
    except Exception:
        print("Downloading NLTK WordNet data... (this happens once)")
        nltk.download('wordnet')
        print("Download complete.")

    # Load memory preferences to get sharding rules
    prefs = get_memory_preferences(str(MEMORIES_DIR))

    print("=== WordNet Format Generation ===")
    
    # --- Generate Formats ---
    create_lemma_format(prefs)
    create_synset_format(prefs)
    
    print("\n=== WordNet format generation complete ===")
    print("Next step: Run the ingestion script to train on WordNet data.")
    
if __name__ == "__main__":
    main()