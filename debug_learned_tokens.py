#!/usr/bin/env python3
"""
Debug what tokens are actually learned after filtering.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.kernel.gyro_core import GyroEngine
from baby.tokenizer import get_tokenizer

def debug_learned_tokens():
    print("🔍 Debug Learned Tokens After Filtering")
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
    
    # Test with the same text that knowledge_test uses
    test_text = "In mathematics and computer science, an algorithm is a sequence of rigorous instructions."
    tokens = tokenizer.encode(test_text)
    print(f"Text: '{test_text}'")
    print(f"Tokens: {tokens}")
    
    # Check which tokens will be learned
    print(f"\nFiltered tokens (that will be learned):")
    learned_tokens = []
    for token_id in tokens:
        if engine.should_learn_from_token(token_id, "user"):
            decoded = tokenizer.decode([token_id])
            print(f"  Token {token_id}: '{decoded}' - LEARN")
            learned_tokens.append(token_id)
        else:
            decoded = tokenizer.decode([token_id])
            print(f"  Token {token_id}: '{decoded}' - SKIP")
    
    print(f"\nTotal tokens to learn: {len(learned_tokens)}")
    
    if len(learned_tokens) == 0:
        print("❌ NO TOKENS WILL BE LEARNED! This explains the empty domain.")
        return
    
    # Learn the tokens
    state = engine.start_state()
    for token_id in learned_tokens:
        state = engine.learn_on_user(state, token_id)
    
    # Check what was actually learned
    total_learned = sum(len(tokens) for tokens in engine.tokens_by_addr_rep_user.values())
    print(f"Actually learned: {total_learned} tokens")
    
    if total_learned == 0:
        print("❌ NOTHING WAS LEARNED! Domain is empty.")
    else:
        print("✅ Some tokens were learned.")
        print("Learned tokens by orbit:")
        for rep_idx, token_set in engine.tokens_by_addr_rep_user.items():
            print(f"  Orbit {rep_idx}: {sorted(token_set)}")
        
    # Try emission
    current_state_idx = engine.state_to_index[state]
    result = engine.emit_next(current_state_idx)
    if result:
        token_id, _ = result
        decoded = tokenizer.decode([token_id])
        print(f"✅ Can emit: token {token_id} ('{decoded}')")
    else:
        print("❌ Cannot emit: no candidates")

if __name__ == "__main__":
    debug_learned_tokens()
