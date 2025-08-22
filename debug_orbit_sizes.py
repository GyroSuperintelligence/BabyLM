#!/usr/bin/env python3
"""
Debug orbit sizes for learned tokens.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.kernel.gyro_core import GyroEngine
from baby.tokenizer import get_tokenizer

def debug_orbit_sizes():
    print("🔍 Debug Orbit Sizes for Learned Tokens")
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
    
    # Test tokens
    test_tokens = [637, 49085, 326, 7595, 11222, 11, 448, 22184, 382, 261, 16281, 328, 68524, 15543, 13]
    
    print("Token analysis:")
    token_info = []
    for token_id in test_tokens:
        addr_int = engine.address_of_token(token_id)
        addr_idx = engine.state_to_index[addr_int]
        orbit_size = int(engine.orbit_sizes[addr_idx])
        decoded = tokenizer.decode([token_id])
        token_info.append((orbit_size, token_id, decoded))
        print(f"  Token {token_id} ('{decoded}'): orbit_size={orbit_size}")
    
    # Sort by orbit size
    token_info.sort()
    print(f"\nSorted by orbit size (ascending):")
    for orbit_size, token_id, decoded in token_info:
        print(f"  {orbit_size}: token {token_id} ('{decoded}')")
    
    # Check which tokens advance from state 0
    print(f"\nState advancement test from state 0:")
    for orbit_size, token_id, decoded in token_info:
        from baby.kernel.gyro_core import token_to_introns
        introns = token_to_introns(token_id)
        test_idx = 0
        for intron in introns:
            test_idx = engine.apply_intron_index(test_idx, intron)
        
        if test_idx == 0:
            print(f"  Token {token_id} ('{decoded}'): NO ADVANCE")
        else:
            print(f"  Token {token_id} ('{decoded}'): ADVANCES to state {int(engine.keys[test_idx])}")

if __name__ == "__main__":
    debug_orbit_sizes()
