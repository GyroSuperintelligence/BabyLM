#!/usr/bin/env python3
"""
Quick probe to check candidate counts for the current state.
Run this after making a request to see how many choices the system has.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from toys.communication.external_adapter import agent_pool


def probe_candidates():
    """Probe the current state's candidate count."""
    try:
        a = agent_pool.get("assistant").engine
        rep = (
            int(a.phenomenology_map[a.current_state_index])
            if a.phenomenology_map is not None
            else a.current_state_index
        )
        keys = list(a.operator.store.iter_keys_for_state(rep))
        print(f"Current state index: {a.current_state_index}")
        print(f"Canonical state index: {rep}")
        print(f"Candidates for current state: {len(keys)}")
        if keys:
            print(f"Sample candidates: {keys[:10]}")
        else:
            print("No candidates found - this explains poor coherence!")
    except Exception as e:
        print(f"Probe failed: {e}")


if __name__ == "__main__":
    probe_candidates()
