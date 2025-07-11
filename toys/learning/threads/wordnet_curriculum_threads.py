import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    print("NLTK not found. Please run 'pip install nltk'")
    sys.exit(1)

from baby.intelligence import initialize_intelligence_engine

# --- Configuration ---
MEMORIES_DIR = "memories"

def serialize_synset(synset) -> str:
    """
    Convert a WordNet synset into a structured text representation.
    """
    parts = ["[SYNSET_START]\n"]
    
    # Basic synset info
    parts.append(f"ID: {synset.name()}\n")
    
    # All lemmas in this synset
    for lemma in synset.lemmas():
        lemma_name = lemma.name().replace('_', ' ')
        parts.append(f"LEMMA: {lemma_name}\n")
    
    # Definition
    parts.append(f"DEFINITION: {synset.definition()}\n")
    
    # Examples
    for example in synset.examples():
        parts.append(f"EXAMPLE: {example}\n")
    
    # Hypernyms (more general concepts)
    for hypernym in synset.hypernyms():
        parts.append(f"HYPERNYM: {hypernym.name()}\n")
    
    # Hyponyms (more specific concepts) - limit to avoid huge output
    hyponyms = synset.hyponyms()[:5]  # First 5 to keep manageable
    for hyponym in hyponyms:
        parts.append(f"HYPONYM: {hyponym.name()}\n")
    
    # Meronyms (parts)
    for meronym in synset.part_meronyms()[:3]:
        parts.append(f"PART: {meronym.name()}\n")
    
    # Holonyms (wholes)  
    for holonym in synset.part_holonyms()[:3]:
        parts.append(f"WHOLE: {holonym.name()}\n")
    
    parts.append("[SYNSET_END]\n\n")
    
    return "".join(parts)

def ingest_wordnet_data(engine):
    """
    Traverse WordNet and ingest all data through the intelligence engine.
    """
    print("Starting WordNet data ingestion...")
    
    # Start a new public thread for the curriculum
    root_thread_uuid = engine.start_new_thread(privacy="public")
    print(f"Started curriculum thread: {root_thread_uuid}")
    
    # Get all synsets, starting with root synsets for better organization
    root_synsets = [s for s in wn.all_synsets() if not s.hypernyms()]
    
    # Use BFS traversal for logical ordering  
    queue = list(root_synsets)
    visited = set(root_synsets)
    
    count = 0
    total_bytes = 0
    
    print(f"Found {len(root_synsets)} root synsets. Starting BFS traversal...")
    
    while queue:
        synset = queue.pop(0)
        
        # Serialize this synset
        serialized_text = serialize_synset(synset)
        serialized_bytes = serialized_text.encode('utf-8')
        
        # Process through the intelligence engine
        # This will update pattern resonances, generate keys, and update format metadata
        input_stream, intermediate_ciphertext = engine.process_input_stream(
            serialized_bytes, 
            privacy="public"
        )
        
        total_bytes += len(serialized_bytes)
        count += 1
        
        # Progress reporting
        if count % 100 == 0:
            print(f"  Processed {count} synsets ({total_bytes:,} bytes total)")
            print(f"    Current thread: {engine.thread_uuid}")
            print(f"    Current thread size: {engine.current_thread_size:,} bytes")
        
        # Add hyponyms to queue for continued traversal
        for hyponym in synset.hyponyms():
            if hyponym not in visited:
                visited.add(hyponym)
                queue.append(hyponym)
        
        # Add other related synsets to ensure full coverage
        for related in (synset.similar_tos() + synset.also_sees()):
            if related not in visited:
                visited.add(related)
                queue.append(related)
    
    # Finalize the last thread
    print("\nFinalizing curriculum threads...")
    engine.finalize_and_save_thread(privacy="public")
    
    print(f"\n=== WordNet ingestion complete ===")
    print(f"Processed {count} synsets")
    print(f"Total data processed: {total_bytes:,} bytes")
    print(f"Final thread: {engine.thread_uuid}")

def main():
    # --- NLTK Setup ---
    try:
        wn.all_synsets()
    except Exception:
        print("Downloading NLTK WordNet data... (this happens once)")
        nltk.download('wordnet')
        print("Download complete.")
    
    print("=== WordNet Curriculum Ingestion ===")
    print("Initializing GyroSI Engine in public mode...")
    
    # Initialize in public mode - this will load all available formats
    engine = initialize_intelligence_engine(base_memories_dir=MEMORIES_DIR)
    
    print(f"Loaded {len(engine.formats)} formats:")
    for uuid, fmt in engine.formats.items():
        name = fmt.get('format_name', 'Unknown')
        desc = fmt.get('metadata', {}).get('description', '')[:50]
        print(f"  - {name} ({uuid[:8]}...): {desc}...")
    
    # Set active format - you can choose which format to make primary
    # For this curriculum, we might want to use a mixed approach or pick one
    if engine.formats:
        # Use the first available format as the active one
        # The engine will still consider all formats during resonance calculation
        first_format_uuid = next(iter(engine.formats.keys()))
        engine.format_uuid = first_format_uuid
        print(f"\nSet active format: {engine.formats[first_format_uuid].get('format_name')}")
    
    # Ingest WordNet data
    ingest_wordnet_data(engine)

if __name__ == "__main__":
    main()