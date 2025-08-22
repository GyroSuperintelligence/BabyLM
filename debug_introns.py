#!/usr/bin/env python3
"""
Debug script to check token introns and state transitions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.kernel.gyro_core import GyroEngine
from baby.kernel.gyro_core import token_to_introns

def debug_introns():
    print("🔍 Debugging Token Introns and State Transitions")
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
    
    # Test token 7595 specifically
    token_id = 7595
    print(f"\n🔍 Analyzing token {token_id}:")
    
    # Get introns
    introns = token_to_introns(token_id)
    print(f"   Introns: {introns}")
    
    # Test state transitions from state 0
    state_idx = 0
    current_state = int(engine.keys[state_idx])
    print(f"   Starting state: {current_state} (index: {state_idx})")
    
    for i, intron in enumerate(introns):
        print(f"   Step {i+1}: Applying intron {intron}")
        new_idx = engine.apply_intron_index(state_idx, intron)
        new_state = int(engine.keys[new_idx])
        print(f"     State: {current_state} -> {new_state} (index: {state_idx} -> {new_idx})")
        
        if new_idx == state_idx:
            print(f"     ⚠️  No state change!")
        else:
            print(f"     ✅ State changed")
        
        state_idx = new_idx
        current_state = new_state
    
    print(f"   Final state: {current_state} (index: {state_idx})")
    
    # Check if this token is in the user domain
    addr_int = engine.address_of_token(token_id)
    addr_idx = engine.state_to_index[addr_int]
    addr_rep_idx = engine.orbit_rep_index(addr_idx)
    
    print(f"\n📍 Token {token_id} address analysis:")
    print(f"   Canonical address: {addr_int}")
    print(f"   Address index: {addr_idx}")
    print(f"   Orbit representative: {addr_rep_idx}")
    
    user_domain = engine.tokens_by_addr_rep_user.get(addr_rep_idx, set())
    print(f"   In user domain: {token_id in user_domain}")
    
    # Check passive memory
    key = (addr_idx, token_id)
    passive_mask = engine.passive_mask.get(key, 0)
    print(f"   Passive mask: {passive_mask}")

if __name__ == "__main__":
    debug_introns()
