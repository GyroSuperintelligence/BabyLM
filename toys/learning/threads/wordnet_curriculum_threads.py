#!/usr/bin/env python3
"""
Serializes the entire WordNet database into a single, flat text file.
This output file is the curriculum input for the high-performance trainer.
"""

import sys
from pathlib import Path

try:
    import nltk
    from nltk.corpus import wordnet as wn
    from tqdm import tqdm
except ImportError:
    print("Required packages not found. Please run 'pip install nltk tqdm'")
    sys.exit(1)

# Define the output path for the generated corpus file
OUTPUT_DIR = Path("toys/learning/threads/corpus")
OUTPUT_FILE = OUTPUT_DIR / "wordnet_corpus.txt"

def serialize_synset(synset) -> str:
    """
    Convert a WordNet synset into a structured, clean text representation.
    Each synset is a self-contained block of text.
    """
    parts = [f"[SYNSET ID={synset.name()}]\n"]
    for lemma in synset.lemmas():
        parts.append(f"LEMMA: {lemma.name().replace('_', ' ')}\n")
    
    parts.append(f"DEFINITION: {synset.definition()}\n")
    
    for example in synset.examples():
        parts.append(f"EXAMPLE: {example}\n")
    
    for hypernym in synset.hypernyms():
        parts.append(f"IS_A: {hypernym.name()}\n") # Using clearer 'IS_A' relation
        
    parts.append("[END]\n\n")
    return "".join(parts)

def main():
    """Main function to generate the corpus file."""
    print("=== WordNet Corpus Serialization ===")
    
    # --- NLTK Setup ---
    try:
        wn.all_synsets()
    except LookupError:
        print("Downloading NLTK WordNet data (this happens once)...")
        nltk.download('wordnet')
    
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_synsets = list(wn.all_synsets())
    
    print(f"Serializing {len(all_synsets):,} synsets to '{OUTPUT_FILE}'...")
    
    # Open the file and write all synsets
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for synset in tqdm(all_synsets, desc="Writing Synsets", unit="synset"):
            serialized_text = serialize_synset(synset)
            f.write(serialized_text)
            
    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    
    print("\n🎉✅ WordNet Corpus File Created! ✅🎉")
    print(f"   - File location: {OUTPUT_FILE}")
    print(f"   - File size: {file_size_mb:.2f} MB")
    print("\nNext step: Use the bulk trainer to learn from this file:")
    print(f"   python toys/learning/trainer.py {OUTPUT_FILE}")

if __name__ == "__main__":
    main()