#!/usr/bin/env python3
"""
Build index for knowledge file to enable O(1) lookups.
"""

import os
import struct
from typing import Any


def _unpack_phenotype(buf: bytes, offset: int = 0) -> tuple[dict[str, Any], int]:
    """Unpack a phenotype entry from buffer."""
    _STRUCT_SIZE = 12
    if offset + _STRUCT_SIZE > len(buf):
        raise struct.error("Incomplete record")

    # Unpack the 12-byte struct
    state_idx, token_id, mask = struct.unpack("<IIf", buf[offset : offset + _STRUCT_SIZE])

    entry = {"key": (state_idx, token_id), "mask": mask, "conf": 1.0}  # Default confidence

    return entry, offset + _STRUCT_SIZE


def build_knowledge_index(knowledge_path: str) -> bool:
    """Build index for knowledge file."""
    print(f"Building index for: {knowledge_path}")

    if not os.path.exists(knowledge_path):
        print(f"Error: File not found: {knowledge_path}")
        return False

    file_size = os.path.getsize(knowledge_path)
    print(f"File size: {file_size} bytes")

    # Read the entire file and build index
    index = {}
    offset = 0
    entry_count = 0

    with open(knowledge_path, "rb") as f:
        buf = f.read()

        while offset < len(buf):
            try:
                entry, new_offset = _unpack_phenotype(buf, offset)
                if "key" in entry:
                    context_key = entry["key"]
                    size = new_offset - offset
                    index[context_key] = (offset, size)
                    entry_count += 1
                offset = new_offset
            except struct.error:
                print(f"Warning: Corrupt record at offset {offset}")
                break

    print(f"Found {entry_count} valid entries")

    # Write index file in JSON format
    index_path = knowledge_path + ".idx"
    import json

    # Convert to the format expected by the system: string keys "(state, token_id)" and tuple values (offset, size)
    json_index = {str(context_key): (offset, size) for context_key, (offset, size) in index.items()}
    with open(index_path, "wb") as f:
        f.write(json.dumps(json_index).encode("utf-8"))

    print(f"Index written to: {index_path}")
    return True


if __name__ == "__main__":
    knowledge_path = "memories/public/knowledge/knowledge.bin"
    success = build_knowledge_index(knowledge_path)
    if success:
        print("✓ Index built successfully")
    else:
        print("✗ Failed to build index")
