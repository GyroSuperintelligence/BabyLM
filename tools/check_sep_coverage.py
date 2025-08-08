#!/usr/bin/env python3
"""
Check SEP coverage in the knowledge store.

This script counts how many SEP entries exist in the store and how many unique pre-states
have SEP associations.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from baby.information import SEP_ID
from baby.policies import PhenotypeStore


def check_sep_coverage(store_path: str) -> None:
    """Check SEP coverage in the given store."""
    print(f"Checking SEP coverage in: {store_path}")

    # Open the store
    store = PhenotypeStore(store_path)

    # Count SEP entries
    sep_entries = 0
    unique_pre_states = set()

    for key, entry in store.iter_entries():
        state_idx, token_id = key
        if token_id == SEP_ID:
            sep_entries += 1
            unique_pre_states.add(state_idx)

    print(f"SEP entries: {sep_entries}")
    print(f"unique pre-states: {len(unique_pre_states)}")

    if sep_entries > 0:
        print("✅ SEP is being learned!")
    else:
        print("❌ No SEP entries found - this indicates a learning issue")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tools/check_sep_coverage.py <store_path>")
        print("Example: python tools/check_sep_coverage.py memories/public/knowledge/knowledge.bin")
        sys.exit(1)

    store_path = sys.argv[1]
    check_sep_coverage(store_path)
