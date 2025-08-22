#!/usr/bin/env python3
"""
Debug script to test learning process directly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.kernel.gyro_core import GyroEngine
from baby.tokenizer import get_tokenizer
from baby.constants.harmony_tokens import ALL_CONTROL_TOKENS

def debug_learning():
    print("🔍 Debugging Learning Process")
    print("=" * 50)
    
    # Initialize engine
    engine = GyroEngine(atlas_paths={
        "ontology_keys": "memories/public/meta/ontology_keys.npy",
        "epistemology": "memories/public/meta/epistemology.npy", 
        "phenomenology_map": "memories/public/meta/phenomenology_map.npy",
        "orbit_sizes": "memories/public/meta/orbit_sizes.npy",
        "theta": "memories/public/meta/theta.npy"
    })
    print(f"✅ Engine initialized")
    print(f"   - Start state: {engine.start_state()}")
    print(f"   - Vocab size: {engine.vocab_size}")
    
    # Test with a simple user message
    test_text = "In mathematics and computer science, an algorithm is a sequence of rigorous instructions."
    print(f"\n📝 Test text: {test_text}")
    
    # Tokenize
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(test_text)
    print(f"   - Tokens: {tokens[:10]}... (total: {len(tokens)})")
    
    # Check which tokens are user vs control
    user_tokens = []
    control_tokens = []
    for tok in tokens:
        if tok in ALL_CONTROL_TOKENS:
            control_tokens.append(tok)
        else:
            user_tokens.append(tok)
    
    print(f"   - User tokens: {len(user_tokens)}")
    print(f"   - Control tokens: {len(control_tokens)}")
    
    # Test learning on first few user tokens
    state = engine.start_state()
    print(f"\n🧠 Learning process:")
    print(f"   - Initial state: {state}")
    
    for i, token_id in enumerate(user_tokens[:5]):
        print(f"\n   Step {i+1}: Token {token_id}")
        
        # Check if token would be learned
        if token_id in ALL_CONTROL_TOKENS:
            print(f"     ❌ Control token - skipping")
            continue
            
        # Get canonical address
        addr_int = engine.address_of_token(token_id)
        addr_idx = engine.state_to_index[addr_int]
        addr_rep_idx = engine.orbit_rep_index(addr_idx)
        
        print(f"     - Canonical address: {addr_int}")
        print(f"     - Address index: {addr_idx}")
        print(f"     - Orbit representative: {addr_rep_idx}")
        
        # Check current passive mask
        key = (addr_idx, token_id)
        prev_mask = engine.passive_mask.get(key, 0)
        print(f"     - Previous passive mask: {prev_mask}")
        
        # Learn the token
        new_state = engine.learn_on_user(state, token_id)
        print(f"     - New state: {new_state}")
        
        # Check updated passive mask
        new_mask = engine.passive_mask.get(key, 0)
        print(f"     - New passive mask: {new_mask}")
        
        # Check if token was registered
        user_domain = engine.tokens_by_addr_rep_user.get(addr_rep_idx, set())
        print(f"     - User domain size: {len(user_domain)}")
        print(f"     - Token in domain: {token_id in user_domain}")
        
        # Check if token 7595 is in the domain after this step
        if token_id == 7595:
            print(f"     🔍 SPECIAL CHECK - Token 7595:")
            print(f"        - In user domain: {7595 in user_domain}")
            print(f"        - All tokens in domain: {sorted(user_domain)}")
        
        state = new_state
    
    # Check final state
    print(f"\n📊 Final Results:")
    print(f"   - Final state: {state}")
    print(f"   - Total passive memory entries: {len(engine.passive_mask)}")
    print(f"   - Total user domain entries: {sum(len(domain) for domain in engine.tokens_by_addr_rep_user.values())}")
    
    # Check each orbit representative
    for rep_idx, tokens in engine.tokens_by_addr_rep_user.items():
        print(f"   - Orbit {rep_idx}: {len(tokens)} tokens")
        if tokens:
            print(f"     Tokens: {list(tokens)[:5]}...")
    
    # Special check for token 7595
    print(f"\n🔍 Special check for token 7595:")
    addr_int = engine.address_of_token(7595)
    addr_idx = engine.state_to_index[addr_int]
    addr_rep_idx = engine.orbit_rep_index(addr_idx)
    user_domain = engine.tokens_by_addr_rep_user.get(addr_rep_idx, set())
    print(f"   - Token 7595 address: {addr_int}")
    print(f"   - Token 7595 orbit rep: {addr_rep_idx}")
    print(f"   - Token 7595 in user domain: {7595 in user_domain}")
    print(f"   - All tokens in orbit {addr_rep_idx}: {sorted(user_domain)}")

if __name__ == "__main__":
    debug_learning()
