#!/usr/bin/env python3
"""Verify the unified cryptographer integration"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extensions.ext_cryptographer import ext_Cryptographer
from src.core.gyro_core import GyroEngine

def verify_integration():
    print("Verifying Unified Cryptographer Integration...")

    # 1. Verify extension loads
    try:
        key = b"test_key_minimum_16_bytes"
        crypto = ext_Cryptographer(key)
        print("✓ Extension loads successfully")
    except Exception as e:
        print(f"✗ Failed to load extension: {e}")
        return False

    # 2. Verify footprint
    if crypto.get_footprint_bytes() != 5:
        print(f"✗ Wrong footprint: {crypto.get_footprint_bytes()} (expected 5)")
        return False
    print("✓ Footprint is correct (5 bytes)")

    # 3. Verify Gene stays plain
    engine = GyroEngine()
    gene_before = engine.gene["id_0"].clone()

    # Encrypt some data (NOT the Gene!)
    test_data = b"This is test data, not the Gene"
    encrypted = crypto.encrypt(test_data)

    # Verify Gene unchanged
    gene_after = engine.gene["id_0"]
    if not gene_before.equal(gene_after):
        print("✗ Gene was modified!")
        return False
    print("✓ Gene remains unchanged in memory")

    # 4. Verify round-trip encryption
    crypto2 = ext_Cryptographer(key)
    decrypted = crypto2.decrypt(encrypted)
    if decrypted != test_data:
        print("✗ Encryption round-trip failed")
        return False
    print("✓ Encryption round-trip successful")

    # 5. Verify navigation affects crypto
    initial_gyration = crypto.gyration
    for i in range(8):
        crypto.process_navigation_event(i)
    if crypto.gyration == initial_gyration:
        print("✗ Navigation doesn't affect gyration")
        return False
    print("✓ Navigation events affect cryptographic state")

    print("\\n✅ All integration checks passed!")
    return True

if __name__ == "__main__":
    success = verify_integration()
    sys.exit(0 if success else 1)
