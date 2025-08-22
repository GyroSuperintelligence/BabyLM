#!/usr/bin/env python3
"""
Debug script to test pure resonance emission.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.kernel.gyro_core import GyroEngine
from baby.tokenizer import get_tokenizer

def debug_emission():
    print("🔍 Debugging Pure Resonance Emission")
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
    
    # Learn some tokens first
    test_text = "In mathematics and computer science, an algorithm is a sequence of rigorous instructions."
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(test_text)
    
    print(f"\n📚 Learning tokens: {tokens[:5]}...")
    state = engine.start_state()
    
    # Learn first 5 tokens
    for i, token_id in enumerate(tokens[:5]):
        if token_id not in [0, 1, 2]:  # Skip control tokens
            state = engine.learn_on_user(state, token_id)
            print(f"   Learned token {token_id}, new state: {state}")
    
    print(f"\n🧠 Learning complete:")
    print(f"   - Final state: {state}")
    print(f"   - Passive memory entries: {len(engine.passive_mask)}")
    print(f"   - User domain entries: {sum(len(domain) for domain in engine.tokens_by_addr_rep_user.values())}")
    
    # Test emission with detailed debugging
    print(f"\n🌀 Testing Pure Resonance Emission:")
    state_idx = engine.state_to_index[state]
    
    for i in range(5):
        print(f"\n   --- Step {i+1} ---")
        print(f"   Current state: {state} (index: {state_idx})")
        
        # Check what candidates are available
        rep_idx = engine.orbit_rep_index(state_idx)
        candidates = engine.tokens_by_addr_rep_user.get(rep_idx, set())
        print(f"   Orbit representative: {rep_idx}")
        print(f"   Available candidates: {candidates}")
        
        if not candidates:
            print(f"   No candidates available!")
            break
        
        # Calculate defects for all candidates
        print(f"   Resonance defects:")
        defects = []
        for token_id in candidates:
            defect = engine._resonance_defect(state_idx, token_id)
            defects.append((defect, token_id))
            print(f"     Token {token_id}: defect {defect}")
        
        # Find minimum defect
        min_defect = min(defects, key=lambda x: x[0])
        tok = min_defect[1]
        print(f"   Selected token: {tok} (defect: {min_defect[0]})")
        
        # Apply token
        result = engine.emit_next(state_idx)
        if result is None:
            print(f"   emit_next returned None!")
            break
        
        token_id, new_idx = result
        new_state = int(engine.keys[new_idx])
        
        print(f"   Applied token {token_id} -> new state: {new_state} (index: {new_idx})")
        
        state_idx = new_idx
        state = new_state

if __name__ == "__main__":
    debug_emission()
