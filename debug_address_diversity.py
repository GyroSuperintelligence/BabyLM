#!/usr/bin/env python3
"""
Debug script to check address diversity and binding patterns.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.kernel.gyro_core import GyroEngine

def debug_address_diversity():
    print("🔍 Debugging Address Diversity and Binding Patterns")
    print("=" * 60)
    
    # Initialize engine
    engine = GyroEngine(atlas_paths={
        "ontology_keys": "memories/public/meta/ontology_keys.npy",
        "epistemology": "memories/public/meta/epistemology.npy", 
        "phenomenology_map": "memories/public/meta/phenomenology_map.npy",
        "orbit_sizes": "memories/public/meta/orbit_sizes.npy",
        "theta": "memories/public/meta/theta.npy"
    })
    print(f"✅ Engine initialized")
    
    # Check a broader range of tokens
    test_tokens = [0, 1, 2, 10, 100, 326, 637, 1000, 7595, 11222, 49085]
    
    print(f"\n📊 Address Binding Analysis:")
    addresses_seen = set()
    
    for token_id in test_tokens:
        addr = engine.address_of_token(token_id)
        addresses_seen.add(addr)
        
        # Check if this is a "boring" address (repeated patterns)
        bytes_list = [(addr >> (i * 8)) & 0xFF for i in range(6)]
        unique_bytes = set(bytes_list)
        
        print(f"   Token {token_id:5d}: addr=0x{addr:012X}, bytes={[f'{x:02X}' for x in bytes_list]}, unique_bytes={len(unique_bytes)}")
    
    print(f"\n📈 Address Diversity Summary:")
    print(f"   Total tokens tested: {len(test_tokens)}")
    print(f"   Unique addresses: {len(addresses_seen)}")
    print(f"   Address diversity ratio: {len(addresses_seen)/len(test_tokens):.2f}")
    
    # Check some sample orbit representatives
    print(f"\n🌍 Sample Orbit Representatives:")
    for i, rep_idx in enumerate(engine.orbit_reps[:10]):
        rep_addr = int(engine.keys[rep_idx])
        bytes_list = [(rep_addr >> (j * 8)) & 0xFF for j in range(6)]
        unique_bytes = set(bytes_list)
        print(f"   Rep {i}: addr=0x{rep_addr:012X}, bytes={[f'{x:02X}' for x in bytes_list]}, unique_bytes={len(unique_bytes)}")

if __name__ == "__main__":
    debug_address_diversity()
