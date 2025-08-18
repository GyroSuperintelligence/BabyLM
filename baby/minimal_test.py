#!/usr/bin/env python3
"""
Minimal test to isolate performance bottleneck in knowledge injection.
"""

import time
import sys
from pathlib import Path

# Add baby module to path
sys.path.insert(0, str(Path(__file__).parent))

from baby.kernel.gyro_core import GyroEngine
from baby.tokenizer import get_tokenizer

def test_address_computation():
    """Test address computation performance for a few tokens."""
    print("Loading GyroEngine...")
    
    # Load engine with config from config.json
    atlas_paths = {
        'epistemology': 'memories/public/meta/epistemology.npy',
        'ontology_keys': 'memories/public/meta/ontology_keys.npy',
        'theta': 'memories/public/meta/theta.npy',
        'phenomenology_map': 'memories/public/meta/phenomenology_map.npy',
        'orbit_sizes': 'memories/public/meta/orbit_sizes.npy'
    }
    
    store_paths = {
        'address_memory': 'memories/public/knowledge/address_memory.dat',
        'passive_memory': 'memories/public/knowledge/passive_memory.bin'
    }
    
    runtime = {
        'max_nudges': 6,
        'enable_self_reinforcement': False
    }
    
    version_info = {
        'atlas_version': 'v1.2.0',
        'address_version': 'v1.1.0',
        'config_version': 'v1.0.0'
    }
    
    start_time = time.time()
    engine = GyroEngine(atlas_paths, store_paths, runtime, version_info)
    load_time = time.time() - start_time
    print(f"✓ Engine loaded in {load_time:.2f}s")
    
    # Test tokenizer
    tokenizer = get_tokenizer()
    test_text = "The quick brown fox"
    tokens = tokenizer.encode(test_text)
    print(f"Test text: '{test_text}'")
    print(f"Tokens: {tokens[:10]}...")  # Show first 10 tokens
    
    # Test address computation for first few tokens
    print("\nTesting address computation:")
    for i, token_id in enumerate(tokens[:5]):  # Test only first 5 tokens
        start_time = time.time()
        address = engine.address_of_token(token_id)
        compute_time = time.time() - start_time
        print(f"Token {i+1} (id={token_id}): {compute_time:.4f}s -> 0x{address:012X}")
        
        if compute_time > 1.0:  # If any token takes more than 1 second
            print(f"⚠️  Token {token_id} took {compute_time:.2f}s - this is too slow!")
            break
    
    print(f"\nCache size: {len(engine._address_cache)} entries")

if __name__ == "__main__":
    test_address_computation()