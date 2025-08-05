#!/usr/bin/env python3
"""
Test script for multi-knowledge functionality.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from baby.policies import create_multi_knowledge_view


def test_multi_knowledge():
    """Test the multi-knowledge functionality."""
    print("Testing multi-knowledge functionality...")

    # Test with the public knowledge directory
    knowledge_dir = "memories/public/knowledge"

    # Create multi-knowledge view
    multi_view = create_multi_knowledge_view(knowledge_dir, "knowledge_*.bin")

    print(f"Found {len(multi_view.stores)} knowledge files")

    if multi_view.stores:
        print("✅ Multi-knowledge view created successfully")

        # Test getting some data
        total_entries = 0
        for store in multi_view.stores:
            entries = len(store.data)
            total_entries += entries
            print(f"  Store: {entries} entries")

        print(f"Total entries across all stores: {total_entries}")

        # Test iter_entries
        unique_entries = 0
        seen_keys = set()
        for key, entry in multi_view.iter_entries():
            if key not in seen_keys:
                unique_entries += 1
                seen_keys.add(key)

        print(f"Unique entries from iter_entries: {unique_entries}")

        multi_view.close()
        print("✅ Multi-knowledge view closed successfully")
    else:
        print("⚠️  No knowledge files found matching pattern 'knowledge_*.bin'")
        print("This is expected if no training has been done yet.")


if __name__ == "__main__":
    test_multi_knowledge()
