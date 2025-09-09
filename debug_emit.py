#!/usr/bin/env python3
import json
from baby.kernel.gyro_core import GyroEngine

def debug_emit():
    # Load config
    with open('baby/config.json') as f:
        config = json.load(f)

    # Create engine
    engine = GyroEngine(
        atlas_paths=config['atlas'],
        store_paths=config['stores'],
        runtime=config['runtime'],
        vocab_size=201_088
    )

    print("=== DEBUGGING EMIT ISSUE ===")
    
    # Test state
    state = engine.start_state()
    print(f'Start state: 0x{state:012X}')

    # Test with simple input
    test_inputs = ["hello", "algorithms are", "how are you"]
    
    for test_input in test_inputs:
        print(f"\n--- Testing: '{test_input}' ---")
        
        # Reset to start state
        current_state = engine.start_state()
        
        # Tokenize (simplified - just use some test tokens)
        if test_input == "hello":
            tokens = [15496, 995]  # Approximate tokens for "hello"
        elif test_input == "algorithms are":
            tokens = [15496, 995]  # Same tokens to test
        else:
            tokens = [15496, 995]
            
        print(f"Using tokens: {tokens}")
        
        # Learn from tokens
        for token in tokens:
            print(f"Learning from token: {token}")
            current_state = engine.learn_on_user(current_state, token)
            print(f"New state: 0x{current_state:012X}")
        
        # Try to emit
        print(f"Final state: 0x{current_state:012X}")
        result = engine.emit_next_from_state(current_state)
        print(f"Emit result: {result}")
        
        if result is None:
            print("❌ NO RESPONSE")
        else:
            print(f"✅ Got response: {result[0]}")

if __name__ == "__main__":
    debug_emit()
