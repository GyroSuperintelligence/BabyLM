#!/usr/bin/env python3
"""
Check SEP candidates for specific prompts.

This script checks if the knowledge store has any candidates (including SEP) for a given prompt.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from baby.information import SEP_ID, encode_text
from baby.policies import PhenotypeStore


def check_prompt_candidates(store_path: str, prompt: str) -> None:
    """Check candidates for a specific prompt."""
    print(f"Checking candidates for prompt: '{prompt}'")
    print(f"Store: {store_path}")

    # Open the store
    store = PhenotypeStore(store_path)

    # Encode the prompt
    prompt_bytes = encode_text(prompt, name="bert-base-uncased")

    # For now, we'll just check if SEP exists in the store
    # In a full implementation, we'd need to process the prompt through the engine
    # to get the final state, then check candidates for that state

    sep_entries = 0
    total_entries = 0

    for key, entry in store.iter_entries():
        state_idx, token_id = key
        total_entries += 1
        if token_id == SEP_ID:
            sep_entries += 1

    print(f"Total entries in store: {total_entries}")
    print(f"SEP entries: {sep_entries}")

    if sep_entries > 0:
        print("✅ SEP candidates exist in the store")
        has_sep = True
    else:
        print("❌ No SEP candidates found")
        has_sep = False

    print(f"has_sep={has_sep}")

    # This is a simplified check - in practice you'd need to:
    # 1. Process the prompt through the engine to get the final state
    # 2. Check candidates for that specific state
    # 3. See if any of those candidates include SEP

    print("\nNote: This is a simplified check. For accurate results, you'd need to:")
    print("1. Process the prompt through the engine")
    print("2. Get the final state after processing")
    print("3. Check candidates for that specific state")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tools/sep_for_prompt.py <store_path> <prompt>")
        print("Example: python tools/sep_for_prompt.py memories/public/knowledge/knowledge.bin 'Hello'")
        sys.exit(1)

    store_path = sys.argv[1]
    prompt = sys.argv[2]
    check_prompt_candidates(store_path, prompt)
