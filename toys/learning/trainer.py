#!/usr/bin/env python3
"""
A scalable, robust bulk trainer for GyroSI.

This script reads a large corpus file and correctly performs all three
training tasks in a single, efficient pass:
1.  Updates all Format file statistics (semantic learning).
2.  Creates a persistent, chunked public Thread of the curriculum (macro-context).
3.  Creates a corresponding stream of Gene Keys for the thread (micro-context).
"""

import sys
import os
import uuid
import datetime
import base64
from pathlib import Path
import numpy as np

# Ensure project root is on the path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found. Please run 'pip install tqdm'")
    sys.exit(1)

from baby.inference import InferenceEngine
from baby.information import (
    list_formats, load_format, store_format, get_memory_preferences,
    create_thread, json_dumps, shard_path, json_loads, store_gene_keys
)
from baby.types import GeneKeysMetadata, FormatMetadata

# --- Configuration ---
CHUNK_SIZE = 1 << 22  # 4 MiB for a good balance of memory and I/O
MEMORIES_DIR = "memories"

class TrainingSession:
    """Manages the state of a bulk training session."""

    def __init__(self, ie: InferenceEngine, prefs: dict):
        self.ie = ie
        self.prefs = prefs
        self.formats = self._load_all_formats()
        
        # Session State
        self.active_thread_uuid: str | None = None
        self.active_thread_handle = None
        self.current_thread_size = 0
        self.parent_thread_uuid: str | None = None
        
        # Gene keys buffer for batch writing
        self.gene_keys_buffer = []
        
        # Use the first available text format for logging context
        self.primary_format_uuid = self._find_primary_format()

    def _load_all_formats(self) -> dict[str, FormatMetadata]:
        """Loads all valid formats from disk."""
        formats = {u: load_format(u, MEMORIES_DIR) for u in list_formats(MEMORIES_DIR)}
        valid_formats = {u: f for u, f in formats.items() if f and "patterns" in f}
        print(f"✓ Loaded {len(valid_formats)} valid formats for training.")
        return valid_formats

    def _find_primary_format(self) -> str:
        """Finds a suitable text format to be the primary context."""
        for u, f in self.formats.items():
            if "ascii" in f.get("format_name", ""):
                return u
        return next(iter(self.formats.keys()))

    def start_new_training_thread(self):
        """Finalizes the previous thread and starts a new one, resuming if files exist."""
        if self.active_thread_uuid:
            self.finalize_current_thread()

        new_uuid = create_thread(
            privacy="public",
            parent_uuid=self.parent_thread_uuid,
            format_uuid=self.primary_format_uuid,
            prefs=self.prefs,
            base_memories_dir=MEMORIES_DIR,
            thread_name="wordnet_curriculum_part",
            tags=["wordnet", "curriculum"]
        )
        
        self.parent_thread_uuid = self.active_thread_uuid
        self.active_thread_uuid = new_uuid
        self.current_thread_size = 0
        self.gene_keys_buffer = []
        print(f"\nSwitching to new thread: {self.active_thread_uuid}")

        # Open thread file in append mode if it exists, else write mode
        threads_root = Path(MEMORIES_DIR) / "public/threads"
        thread_shard = shard_path(threads_root, new_uuid, self.prefs)
        thread_path = thread_shard / f"thread-{new_uuid}.ndjson"
        if thread_path.exists():
            self.active_thread_handle = open(thread_path, "a", encoding="utf-8")
            self.current_thread_size = thread_path.stat().st_size
            print(f"Resuming existing thread file: {thread_path}")
        else:
            self.active_thread_handle = open(thread_path, "w", encoding="utf-8")
            print(f"Creating new thread file: {thread_path}")

        # No need to open gene key file handle directly; store_gene_keys will append if file exists

    def process_chunk(self, chunk_buffer: bytes):
        """Processes a chunk of data, updating all artifacts."""
        batch_size = self.prefs.get("stream_config", {}).get("train_batch_size", 4096)
        
        for offset in range(0, len(chunk_buffer), batch_size):
            batch_bytes = chunk_buffer[offset : offset + batch_size]
            if not batch_bytes: continue
            
            p_batch = np.frombuffer(batch_bytes, dtype=np.uint8)

            # 1. INFERENCE: The fast, batched physics calculation.
            key_indices, resonances = self.ie.process_batch(p_batch)

            # 2. FORMAT LEARNING: Update format stats in memory.
            self._update_formats(key_indices, resonances)

            # 3. THREAD WRITING: Write the raw content to the thread file.
            self._write_thread_content(batch_bytes)
            
            # 4. GENE KEY WRITING: Buffer the event log for batch writing.
            self._buffer_gene_keys(p_batch, key_indices, resonances)

    def _update_formats(self, key_indices: np.ndarray, resonances: np.ndarray):
        """Updates all formats based on a batch of results."""
        freq = np.bincount(key_indices, minlength=256)
        
        # To calculate average resonance, we need to sum them up per index
        # We can do this efficiently without a Python loop.
        resonance_sums = np.bincount(key_indices, weights=resonances, minlength=256)
        
        for fmt in self.formats.values():
            if "patterns" not in fmt:
                continue
            for p_entry in fmt["patterns"]:
                if "index" not in p_entry:
                    continue
                idx = p_entry["index"]
                
                # Update Count
                count_increase = int(freq[idx])
                if count_increase == 0: continue
                
                current_count = p_entry.get("count", 0)
                new_count = current_count + count_increase
                p_entry["count"] = new_count
                
                # Update Confidence (moving average)
                current_confidence = p_entry.get("confidence", 0.0)
                batch_avg_resonance = resonance_sums[idx] / count_increase
                batch_confidence = 1.0 - (batch_avg_resonance / np.pi)
                
                # Weight the update by the size of the batch vs. previous count
                alpha = count_increase / new_count
                p_entry["confidence"] = (1 - alpha) * current_confidence + alpha * batch_confidence

    def _write_thread_content(self, content_bytes: bytes):
        """Writes a content event to the active thread file."""
        max_size = self.prefs["storage_config"]["max_thread_size_mb"] * 1024 * 1024
        if self.current_thread_size >= max_size:
            self.start_new_training_thread()
            
        event = {"type": "input", "data": base64.b64encode(content_bytes).decode("utf-8")}
        line = json_dumps(event) + '\n'
        if self.active_thread_handle:
            bytes_written = self.active_thread_handle.write(line)
        self.current_thread_size += bytes_written

    def _buffer_gene_keys(self, source_bytes: np.ndarray, key_indices: np.ndarray, resonances: np.ndarray):
        """Buffers gene key events for batch writing."""
        now = datetime.datetime.now().isoformat()
        for i in range(len(source_bytes)):
            if self.active_thread_uuid is None:
                continue
            key_event: GeneKeysMetadata = {
                "cycle": self.ie.cycle_counter - len(source_bytes) + i,
                "pattern_index": int(key_indices[i]),
                "thread_uuid": self.active_thread_uuid,
                "format_uuid": self.primary_format_uuid,
                "event_type": "INPUT",
                "source_byte": int(source_bytes[i]),
                "resonance": float(resonances[i]),
                "created_at": now,
                "privacy": "public",
                "agent_uuid": None
            }
            self.gene_keys_buffer.append(key_event)
            
            # Write gene keys in batches to avoid memory buildup
            if len(self.gene_keys_buffer) >= 1000:
                self._flush_gene_keys()

    def _flush_gene_keys(self):
        """Writes buffered gene keys to disk using the proper store_gene_keys function."""
        if not self.gene_keys_buffer:
            return
            
        if self.active_thread_uuid:
            store_gene_keys(
                thread_uuid=self.active_thread_uuid,
                gene_keys=self.gene_keys_buffer,
                privacy="public",
                prefs=self.prefs,
                agent_secret=None,
                agent_uuid=None,
                base_memories_dir=MEMORIES_DIR
            )
        self.gene_keys_buffer = []

    def finalize_current_thread(self):
        """Closes handles and updates metadata for the currently active thread."""
        if not self.active_thread_uuid: return
        
        # Flush any remaining gene keys
        self._flush_gene_keys()
        
        # Close thread handle
        if self.active_thread_handle: 
            self.active_thread_handle.close()
        
        # Update metadata
        threads_root = Path(MEMORIES_DIR) / "public/threads"
        thread_shard = shard_path(threads_root, self.active_thread_uuid, self.prefs)
        meta_path = thread_shard / f"thread-{self.active_thread_uuid}.json"
        
        if meta_path.exists():
            with open(meta_path, "r+") as f:
                meta = json_loads(f.read())
                meta["size_bytes"] = self.current_thread_size
                meta["last_updated"] = datetime.datetime.now().isoformat()
                f.seek(0)
                f.truncate()
                f.write(json_dumps(meta))

    def checkpoint(self):
        """Saves the current state of all formats to disk."""
        print("\n--- Checkpointing learned formats to disk... ---")
        for fmt in self.formats.values():
            for p in fmt.get("patterns", []):
                if "count" in p:
                    p["count"] = int(p["count"])
                if "confidence" in p:
                    p["confidence"] = float(p["confidence"])
            store_format(fmt, self.prefs, MEMORIES_DIR)
        print("--- Checkpoint complete. ---")

def main(filepath: str):
    """Main training function."""
    print("=== GyroSI Scalable Bulk Trainer ===")
    
    prefs = get_memory_preferences(MEMORIES_DIR)
    ie = InferenceEngine(base_memories_dir=MEMORIES_DIR)
    
    session = TrainingSession(ie, prefs)
    
    # Start the first thread of the curriculum
    session.start_new_training_thread()
    
    file_size = Path(filepath).stat().st_size
    
    with open(filepath, "rb") as fh, tqdm(total=file_size, unit="B", unit_scale=True, desc="Training") as pbar:
        while True:
            buf = fh.read(CHUNK_SIZE)
            if not buf:
                break
            
            session.process_chunk(buf)
            pbar.update(len(buf))
            
            # Optional: checkpoint every N chunks
            if pbar.n // (CHUNK_SIZE * 100) > (pbar.last_print_n // (CHUNK_SIZE * 100)):
                session.checkpoint()
    
    # Finalize all artifacts
    session.finalize_current_thread()
    session.checkpoint()
    
    print("\n🎉✅ Training Complete! All artifacts (Threads, Keys, Formats) created successfully. ✅🎉")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_corpus_file>")
        sys.exit(1)
    main(sys.argv[1])