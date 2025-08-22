#!/usr/bin/env python3
"""
Debug script to check path memory evolution and exon calculations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.kernel.gyro_core import GyroEngine
from baby.tokenizer import get_tokenizer

def debug_path_memory():
    print("🔍 Debugging Path Memory and Exon Calculations")
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
    print(f"   Initial path memory: {engine.path_memory} (0x{engine.path_memory:02X})")
    
    # Test with specific tokens
    test_tokens = [637, 49085, 326, 7595, 11222]
    
    print(f"\n📊 Token Exon Analysis:")
    for token_id in test_tokens:
        addr_int = engine.address_of_token(token_id)
        exon = engine._compute_exon_from_state(addr_int)
        print(f"   Token {token_id}: address={addr_int}, exon={exon} (0x{exon:02X})")
    
    # Test learning and path memory evolution
    print(f"\n🧠 Learning and Path Memory Evolution:")
    state = engine.start_state()
    
    for i, token_id in enumerate(test_tokens[:3]):
        print(f"\n   Step {i+1}: Learning token {token_id}")
        print(f"     Before: path_memory={engine.path_memory} (0x{engine.path_memory:02X})")
        
        new_state = engine.learn_on_user(state, token_id)
        
        print(f"     After:  path_memory={engine.path_memory} (0x{engine.path_memory:02X})")
        print(f"     State: {state} -> {new_state}")
        
        state = new_state
    
    # Test resonance calculations after learning
    print(f"\n🌀 Resonance Analysis After Learning:")
    print(f"   Current path memory: {engine.path_memory} (0x{engine.path_memory:02X})")
    
    for token_id in test_tokens:
        defect = engine._resonance_defect(0, token_id)  # Use state index 0
        exon = engine._compute_exon_from_state(engine.address_of_token(token_id))
        
        print(f"   Token {token_id}: exon={exon:02X}, defect={defect}")
        
        # Show the actual fold calculation
        path_byte = engine.path_memory & 0xFF
        fold_result = engine.fold_sequence([exon], path_byte)
        print(f"     fold({path_byte:02X}, {exon:02X}) = {fold_result}")

if __name__ == "__main__":
    debug_path_memory()
