#!/usr/bin/env python3
"""
Quick-n-dirty bulk trainer.  Point it at a large text file; it
updates all loaded formats using the new batch API.
"""

import sys, pathlib, numpy as np
from baby.inference import InferenceEngine
from baby.information import list_formats, load_format, store_format, get_memory_preferences

CHUNK = 1 << 20          # 1 MiB
MEM  = "memories"        # base dir

def main(filepath: str):
    prefs = get_memory_preferences(MEM)
    batch_size = prefs.get("stream_config", {}).get("train_batch_size", 4096)
    ie = InferenceEngine(base_memories_dir=MEM)  # ensures shared mask path

    # Load every format once (keep in RAM)
    formats = {u: load_format(u, MEM) for u in list_formats(MEM)}
    
    # Filter out None formats and ensure patterns exist
    valid_formats = {}
    for uuid, fmt in formats.items():
        if fmt is not None and "patterns" in fmt:
            valid_formats[uuid] = fmt

    # Optional: checkpoint tracking for long training sessions
    last_gb_mark = 0

    with open(filepath, "rb") as fh:
        while True:
            buf = fh.read(CHUNK)
            if not buf:
                break
            # Walk through the chunk in batches so we reuse the fast path
            for off in range(0, len(buf), batch_size):
                batch = np.frombuffer(buf, dtype=np.uint8, count=batch_size, offset=off)

                # ---- inference (vectorised) ----
                key_indices, _ = ie.process_batch(batch)

                # ---- frequency tally in pure NumPy, O(n) once ----
                freq = np.bincount(key_indices, minlength=256)

                # ---- update every loaded format in a single pass ----
                for fmt in valid_formats.values():
                    counts = np.array([p.get("count", 0) for p in fmt["patterns"]], dtype=np.int64)
                    counts += freq              # vector add
                    for i, c in enumerate(counts):
                        fmt["patterns"][i]["count"] = int(c)

            # Optional: checkpoint every gigabyte
            if (fh.tell() // (1 << 30)) != last_gb_mark:
                last_gb_mark = fh.tell() // (1 << 30)
                for u, fmt in valid_formats.items():
                    store_format(fmt, prefs, MEM)
                print(f"checkpoint @ {last_gb_mark} GiB")

    # Persist changed formats
    for u, fmt in valid_formats.items():
        store_format(fmt, prefs, MEM)
    print("✓  Training complete")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_with_batches.py <big_text_file>")
        sys.exit(1)
    main(sys.argv[1]) 