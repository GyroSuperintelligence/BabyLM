#!/usr/bin/env python3
"""
Debug script to check token introns and state transitions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.kernel.gyro_core import GyroEngine, token_to_introns

def debug_token_introns():
    print("🔍 Debugging Token Introns and State Transitions")
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
    
    # Test the problematic token
    token_id = 326
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
    
    # Test a few other tokens to see if they advance the state
    test_tokens = [637, 7595, 11222, 49085]
    print(f"\n🔍 Testing other tokens from state 0:")
    
    for tok in test_tokens:
        introns = token_to_introns(tok)
        new_idx = 0
        for intron in introns:
            new_idx = engine.apply_intron_index(new_idx, intron)
        
        if new_idx == 0:
            print(f"   Token {tok}: NO ADVANCE (stays at state 0)")
        else:
            print(f"   Token {tok}: ADVANCES to state {int(engine.keys[new_idx])} (index {new_idx})")

if __name__ == "__main__":
    debug_token_introns()
