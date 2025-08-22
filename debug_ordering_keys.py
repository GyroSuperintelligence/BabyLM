#!/usr/bin/env python3
"""
Debug ordering keys for different tokens and states.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.kernel.gyro_core import GyroEngine
from baby.tokenizer import get_tokenizer

def debug_ordering_keys():
    print("🔍 Debug Ordering Keys")
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
    
    # Learn tokens
    for token_id in tokens:
        if engine.should_learn_from_token(token_id, "user"):
            state = engine.learn_on_user(state, token_id)
    
    # Test ordering keys for different states
    test_states = [state, 17661175009296, 268435472]
    test_tokens = [2375, 13225]  # The two tokens in the domain
    
    for i, test_state in enumerate(test_states):
        print(f"\n--- Ordering keys for state {i+1}: {test_state} ---")
        test_state_idx = engine.state_to_index[test_state]
        
        candidate_info = []
        for token_id in test_tokens:
            # Get token's canonical address info
            addr_int = engine.address_of_token(token_id)
            addr_idx = engine.state_to_index[addr_int]
            
            # Compute ordering key components
            geom = bin(int(engine.keys[test_state_idx]) ^ int(engine.keys[addr_idx])).count('1')
            size = int(engine.orbit_sizes[addr_idx])
            mask = engine.passive_mask.get((addr_idx, token_id), 0)
            state_diversity = bin(int(engine.keys[test_state_idx]) ^ addr_int).count('1')
            
            # Ordering key: (geometry, state_diversity, orbit_size, passive_mask, token_id)
            order_key = (geom, state_diversity, size, mask, token_id)
            candidate_info.append((order_key, token_id))
            
            decoded = tokenizer.decode([token_id])
            print(f"  Token {token_id} ('{decoded}'): key={order_key}")
            print(f"    geom={geom}, state_diversity={state_diversity}, size={size}, mask={mask}")
        
        # Check for ties
        candidate_info.sort(key=lambda x: x[0])
        best_key = candidate_info[0][0]
        best_candidates = [c for c in candidate_info if c[0] == best_key]
        
        print(f"  Best key: {best_key}")
        print(f"  Candidates with best key: {len(best_candidates)}")
        for _, token_id in best_candidates:
            decoded = tokenizer.decode([token_id])
            print(f"    {token_id} ('{decoded}')")

if __name__ == "__main__":
    debug_ordering_keys()
