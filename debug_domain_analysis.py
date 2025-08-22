#!/usr/bin/env python3
"""
Debug domain analysis for different states.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.kernel.gyro_core import GyroEngine
from baby.tokenizer import get_tokenizer

def debug_domain_analysis():
    print("🔍 Debug Domain Analysis")
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
    
    # Test domain for different states
    test_states = [state, 17661175009296, 268435472]
    
    for i, test_state in enumerate(test_states):
        print(f"\n--- Domain for state {i+1}: {test_state} ---")
        test_state_idx = engine.state_to_index[test_state]
        rep_idx = engine.orbit_rep_index(test_state_idx)
        candidates = engine.tokens_by_addr_rep_user.get(rep_idx, set())
        
        print(f"  State index: {test_state_idx}")
        print(f"  Orbit representative: {rep_idx}")
        print(f"  Available candidates: {len(candidates)} tokens")
        
        if candidates:
            print(f"  Candidates: {sorted(candidates)}")
            for token_id in sorted(candidates):
                decoded = tokenizer.decode([token_id])
                print(f"    {token_id} ('{decoded}')")
        else:
            print(f"  No candidates available!")

if __name__ == "__main__":
    debug_domain_analysis()
