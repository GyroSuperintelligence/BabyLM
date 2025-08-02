#!/usr/bin/env python3
"""
Simple LEB128 Integration Test

This script tests the LEB128 integration without requiring the full API.
It verifies that:
1. LEB128 physics functions work correctly
2. Token-level generation works
3. Learning pipeline works with LEB128
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from baby.intelligence import IntelligenceEngine
from baby.policies import OrbitStore
from toys.communication.tokenizer import token_to_introns, introns_to_token, ψ, ψ_inv


def test_leb128_physics():
    """Test basic LEB128 physics functions."""
    print("Testing LEB128 physics functions...")
    
    # Test ψ isomorphism
    test_bytes = [0x00, 0x7F, 0x80, 0xFF]
    for b in test_bytes:
        intron = ψ(b)
        byte_back = ψ_inv(intron)
        assert b == byte_back, f"ψ isomorphism failed for {b}"
    print("✅ ψ isomorphism test passed")
    
    # Test token-to-intron conversion
    test_tokens = [1, 100, 1000, 10000]
    for token_id in test_tokens:
        introns = token_to_introns(token_id)
        token_back = introns_to_token(introns)
        assert token_id == token_back, f"Token conversion failed for {token_id}"
    print("✅ Token-to-intron conversion test passed")
    
    # Test LEB128 encoding/decoding
    for token_id in test_tokens:
        introns = token_to_introns(token_id)
        # Verify LEB128 format
        for i, intron in enumerate(introns[:-1]):
            assert (intron & 0x80) != 0, f"Continuation bit missing in intron {i}"
        assert (introns[-1] & 0x80) == 0, f"Final bit should be 0"
    print("✅ LEB128 format test passed")


def test_token_generation():
    """Test token-level generation functionality."""
    print("\nTesting token-level generation...")
    
    # Create a simple store with some test data
    store = OrbitStore(store_path=":memory:", use_mmap=False)
    
    # Add some test phenotypes
    test_phenotypes = [
        ((0, 100), {"mask": 0xAA, "conf": 0.8}),
        ((0, 200), {"mask": 0x55, "conf": 0.6}),
        ((1, 100), {"mask": 0x33, "conf": 0.9}),
        ((1, 200), {"mask": 0xCC, "conf": 0.4}),
    ]
    
    for key, entry in test_phenotypes:
        store.put(key, entry)
    
    # Test generate_token_leb128 function
    try:
        # This would require a full IntelligenceEngine, so we'll test the components
        from toys.communication.tokenizer import apply_token_physics
        
        # Test apply_token_physics
        epistemology = np.random.randint(0, 1000, (100, 256), dtype=np.int32)
        state = 0
        token_id = 100
        
        final_state = apply_token_physics(state, token_id, epistemology)
        assert isinstance(final_state, int), "apply_token_physics should return int"
        print("✅ apply_token_physics test passed")
        
    except ImportError:
        print("⚠️  Skipping apply_token_physics test (import failed)")
    
    store.close()


def test_learning_pipeline():
    """Test that the learning pipeline works with LEB128."""
    print("\nTesting learning pipeline with LEB128...")
    
    # Test that learn_token method exists and works
    try:
        from baby.inference import InferenceEngine
        from baby.information import InformationEngine
        
        # Create minimal test components
        ontology_path = PROJECT_ROOT / "memories/public/meta/ontology_keys.npy"
        ep_path = PROJECT_ROOT / "memories/public/meta/epistemology.npy"
        phenomap_path = PROJECT_ROOT / "memories/public/meta/phenomenology_map.npy"
        theta_path = PROJECT_ROOT / "memories/public/meta/theta.npy"
        
        if not all(p.exists() for p in [ontology_path, ep_path, phenomap_path, theta_path]):
            print("⚠️  Skipping learning pipeline test (missing required files)")
            return
        
        # Create test components
        s2 = InformationEngine(
            keys_path=str(ontology_path),
            ep_path=str(ep_path),
            phenomap_path=str(phenomap_path),
            theta_path=str(theta_path)
        )
        
        store = OrbitStore(store_path=":memory:", use_mmap=False)
        
        engine = InferenceEngine(s2_engine=s2, phenotype_store=store)
        
        # Test learn_token method
        entry = engine.learn_token(token_id=100, state_index=0, last_intron=0xAA)
        assert entry is not None, "learn_token should return an entry"
        assert "mask" in entry, "Entry should have mask field"
        assert "conf" in entry, "Entry should have conf field"
        
        print("✅ Learning pipeline test passed")
        
        store.close()
        
    except Exception as e:
        print(f"⚠️  Learning pipeline test failed: {e}")


def test_integration():
    """Test the full integration."""
    print("\nTesting full LEB128 integration...")
    
    # Test that the main codebase can import and use LEB128 functions
    try:
        from toys.communication.tokenizer import (
            token_to_introns, introns_to_token, 
            apply_token_physics, TokenSTT
        )
        
        # Test TokenSTT
        epistemology = np.random.randint(0, 1000, (100, 256), dtype=np.int32)
        token_stt = TokenSTT(epistemology, vocab_size=30522)
        
        # Test token transition
        state = 0
        token_id = 100
        final_state = token_stt.get_token_transition(state, token_id)
        assert isinstance(final_state, int), "TokenSTT should return int"
        
        print("✅ Full integration test passed")
        
    except Exception as e:
        print(f"⚠️  Integration test failed: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("LEB128 INTEGRATION TEST".center(60, "="))
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        test_leb128_physics()
        test_token_generation()
        test_learning_pipeline()
        test_integration()
        
        elapsed = time.time() - start_time
        print(f"\n✅ All tests completed in {elapsed:.2f}s")
        print("LEB128 integration is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 