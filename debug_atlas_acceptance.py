#!/usr/bin/env python3
"""
Debug script to check atlas acceptance criteria.
"""

import sys
import os
import random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.kernel.gyro_core import GyroEngine
import numpy as np

def debug_atlas_acceptance():
    print("🔍 Atlas Acceptance Checks")
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
    
    # 0) Prove the atlas is usable (before touching logic)
    print(f"\n📊 Atlas Acceptance Checks:")
    
    # Check 1: No absorbing row at start
    print(f"\n1️⃣ No absorbing row at start:")
    start_unique = len(np.unique(engine.ep[0, :]))
    print(f"   |unique(ep[0, :])| = {start_unique}")
    if start_unique > 1:
        print(f"   ✅ PASS: Start state can move")
    else:
        print(f"   ❌ FAIL: Start state is absorbing - rebuild epistemology.npy")
    
    # Check 2: Transitions actually branch
    print(f"\n2️⃣ Transitions actually branch:")
    sample_indices = random.sample(range(len(engine.ep)), min(10, len(engine.ep)))
    transition_counts = []
    for i in sample_indices:
        unique_transitions = len(np.unique(engine.ep[i, :]))
        transition_counts.append(unique_transitions)
        print(f"   ep[{i}, :] has {unique_transitions} unique transitions")
    
    avg_transitions = np.mean(transition_counts)
    print(f"   Average unique transitions: {avg_transitions:.1f}")
    if avg_transitions > 10:
        print(f"   ✅ PASS: Introns explore the state space")
    else:
        print(f"   ❌ FAIL: Introns don't explore - atlas may be degenerate")
    
    # Check 3: Address binding has spread
    print(f"\n3️⃣ Address binding has spread:")
    batch_size = 50
    test_tokens = random.sample(range(engine.vocab_size), min(batch_size, engine.vocab_size))
    unique_addresses = set()
    
    for token_id in test_tokens:
        try:
            addr = engine.address_of_token(token_id)
            unique_addresses.add(addr)
        except:
            continue
    
    spread_ratio = len(unique_addresses) / len(test_tokens)
    print(f"   Tested {len(test_tokens)} tokens")
    print(f"   Unique addresses: {len(unique_addresses)}")
    print(f"   Address spread ratio: {spread_ratio:.3f}")
    
    if spread_ratio > 0.7:
        print(f"   ✅ PASS: Address binding has good spread")
    elif spread_ratio > 0.45:
        print(f"   ⚠️  WARNING: Address binding has moderate spread")
    else:
        print(f"   ❌ FAIL: Address binding collapses - rep→final push is degenerate")
    
    # Summary
    print(f"\n📋 Atlas Health Summary:")
    print(f"   Start state movement: {'✅' if start_unique > 1 else '❌'}")
    print(f"   Transition diversity: {'✅' if avg_transitions > 10 else '❌'}")
    print(f"   Address spread: {'✅' if spread_ratio > 0.7 else '⚠️' if spread_ratio > 0.45 else '❌'}")
    
    if start_unique > 1 and avg_transitions > 10 and spread_ratio > 0.45:
        print(f"\n🎉 Atlas appears healthy - proceed with emission fixes")
    else:
        print(f"\n🚨 Atlas has issues - fix atlas before emission logic")

if __name__ == "__main__":
    debug_atlas_acceptance()
