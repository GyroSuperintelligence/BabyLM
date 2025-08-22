#!/usr/bin/env python3
"""
Simple debug script to check what's being learned.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.kernel.gyro_core import GyroEngine
from baby.tokenizer import get_tokenizer

def debug_simple():
    print("🔍 Simple Debug - What's Being Learned?")
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
    
    # Check quote token
    quote_token = tokenizer.encode('"')[0]
    print(f"Quote character '\"' = token {quote_token}")
    
    # Learn from a simple sentence
    test_text = "Hello world"
    tokens = tokenizer.encode(test_text)
    print(f"\nText: '{test_text}'")
    print(f"Tokens: {tokens}")
    
    # Decode each token to see what they are
    for i, token_id in enumerate(tokens):
        decoded = tokenizer.decode([token_id])
        print(f"  Token {token_id}: '{decoded}'")
    
    # Learn the tokens
    state = engine.start_state()
    print(f"\nLearning process:")
    print(f"Start state: {state}")
    
    for token_id in tokens:
        if 0 <= token_id < engine.vocab_size:
            state = engine.learn_on_user(state, token_id)
            print(f"  Learned token {token_id} ('{tokenizer.decode([token_id])}') -> state {state}")
    
    # Check what was learned
    print(f"\nLearned tokens by orbit:")
    for rep_idx, token_set in engine.tokens_by_addr_rep_user.items():
        print(f"  Orbit {rep_idx}: tokens {sorted(token_set)}")
        for token_id in sorted(token_set):
            decoded = tokenizer.decode([token_id])
            print(f"    Token {token_id}: '{decoded}'")
    
    # Test emission
    print(f"\nTesting emission from state {state}:")
    result = engine.emit_next(engine.state_to_index[state])
    if result:
        token_id, new_state_idx = result
        new_state = int(engine.keys[new_state_idx])
        decoded = tokenizer.decode([token_id])
        print(f"  Emitted token {token_id} ('{decoded}') -> state {new_state}")
    else:
        print(f"  No emission (no candidates)")

if __name__ == "__main__":
    debug_simple()
