#!/usr/bin/env python3
"""
Debug orbit diversity for learned tokens.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.kernel.gyro_core import GyroEngine
from baby.tokenizer import get_tokenizer

def debug_orbit_diversity():
    print("🔍 Debug Orbit Diversity")
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
    
    print("Token address and orbit analysis:")
    token_info = []
    for token_id in test_tokens:
        addr_int = engine.address_of_token(token_id)
        addr_idx = engine.state_to_index[addr_int]
        orbit_rep_idx = engine.orbit_rep_index(addr_idx)
        decoded = tokenizer.decode([token_id])
        token_info.append((orbit_rep_idx, token_id, decoded, addr_int))
        print(f"  Token {token_id} ('{decoded}'): addr={addr_int:012X}, orbit_rep={orbit_rep_idx}")
    
    # Group by orbit representative
    orbit_groups = {}
    for orbit_rep_idx, token_id, decoded, addr_int in token_info:
        if orbit_rep_idx not in orbit_groups:
            orbit_groups[orbit_rep_idx] = []
        orbit_groups[orbit_rep_idx].append((token_id, decoded, addr_int))
    
    print(f"\nOrbit distribution:")
    for orbit_rep_idx, tokens in orbit_groups.items():
        print(f"  Orbit {orbit_rep_idx}: {len(tokens)} tokens")
        for token_id, decoded, addr_int in tokens:
            print(f"    {token_id} ('{decoded}') -> {addr_int:012X}")
    
    # Check if we have diversity
    if len(orbit_groups) > 1:
        print(f"\n✅ Orbit diversity: {len(orbit_groups)} different orbits")
    else:
        print(f"\n❌ No orbit diversity: all tokens in orbit {list(orbit_groups.keys())[0]}")

if __name__ == "__main__":
    debug_orbit_diversity()
