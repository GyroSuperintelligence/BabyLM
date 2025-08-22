#!/usr/bin/env python3
"""
Debug script to check exon calculation step by step.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baby.kernel.gyro_core import GyroEngine

def debug_exon_calculation():
    print("🔍 Debugging Exon Calculation Step by Step")
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
    
    # Test specific addresses
    test_addresses = [0, 256087037634792]
    
    for addr in test_addresses:
        print(f"\n📊 Analyzing address {addr} (0x{addr:012X}):")
        
        # Split into 6 bytes
        b = [(addr >> (i * 8)) & 0xFF for i in range(6)]
        print(f"   Bytes: {[f'{x:02X}' for x in b]}")
        
        # Fold opposites
        p1 = engine.fold(b[0], b[3])
        p2 = engine.fold(b[1], b[4])  
        p3 = engine.fold(b[2], b[5])
        
        print(f"   p1 = fold({b[0]:02X}, {b[3]:02X}) = {p1:02X}")
        print(f"   p2 = fold({b[1]:02X}, {b[4]:02X}) = {p2:02X}")
        print(f"   p3 = fold({b[2]:02X}, {b[5]:02X}) = {p3:02X}")
        
        # Final fold
        temp = engine.fold(p1, p2)
        exon = engine.fold(temp, p3)
        
        print(f"   temp = fold({p1:02X}, {p2:02X}) = {temp:02X}")
        print(f"   exon = fold({temp:02X}, {p3:02X}) = {exon:02X}")
        
        # Check fallback
        if exon == 0:
            exon = engine.fold(0xAA, 0x01)
            print(f"   Fallback: exon = fold(AA, 01) = {exon:02X}")
        
        print(f"   Final exon: {exon:02X}")
        
        # Verify with engine method
        engine_exon = engine._compute_exon_from_state(addr)
        print(f"   Engine result: {engine_exon:02X}")
        print(f"   Match: {exon == engine_exon}")

if __name__ == "__main__":
    debug_exon_calculation()
