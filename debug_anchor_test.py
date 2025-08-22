#!/usr/bin/env python3
"""
Debug script to test anchor application.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.kernel.gyro_core import GyroEngine
from baby.tokenizer import get_tokenizer

def debug_anchor_test():
    print("🔍 Debug Anchor Application")
    print("=" * 50)
    
    # Initialize engine
    engine = GyroEngine(atlas_paths={
        "ontology_keys": "memories/public/meta/ontology_keys.npy",
        "epistemology": "memories/public/meta/epistemology.npy", 
        "phenomenology_map": "memories/public/meta/phenomenology_map.npy",
        "orbit_sizes": "memories/public/meta/orbit_sizes.npy",
        "theta": "memories/public/meta/theta.npy"
    })
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Learn some tokens
    test_text = "Hello world"
    tokens = tokenizer.encode(test_text)
    
    state = engine.start_state()
    print(f"Start state: {state}")
    
    # Learn tokens
    for token_id in tokens:
        if engine.should_learn_from_token(token_id, "user"):
            state = engine.learn_on_user(state, token_id)
            print(f"Learned token {token_id} ('{tokenizer.decode([token_id])}') -> state {state}")
    
    # Test multiple emissions to see if state changes
    print(f"\nTesting multiple emissions from state {state}:")
    current_state_idx = engine.state_to_index[state]
    
    for i in range(5):
        result = engine.emit_next(current_state_idx)
        if result:
            token_id, new_state_idx = result
            new_state = int(engine.keys[new_state_idx])
            decoded = tokenizer.decode([token_id])
            print(f"  Emission {i+1}: token {token_id} ('{decoded}') -> state {new_state}")
            current_state_idx = new_state_idx
        else:
            print(f"  Emission {i+1}: No emission")
            break

if __name__ == "__main__":
    debug_anchor_test()
