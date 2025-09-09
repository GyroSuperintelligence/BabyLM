#!/usr/bin/env python3
from baby.responses_api.inference.gyro import setup_model
from baby.tokenizer import get_tokenizer

def debug_learning():
    print("=== DEBUGGING LEARNING ISSUE ===")
    
    # Get tokenizer
    encoding = get_tokenizer()
    
    # Setup model
    infer_next_token = setup_model(encoding, "baby/config.json")
    
    # Test with "algorithms are"
    tokens = encoding.encode("algorithms are")
    print(f"Input tokens: {tokens}")
    
    # Process tokens step by step
    for i, token in enumerate(tokens):
        print(f"\nProcessing token {i+1}: {token} ({encoding.decode([token])})")
        result = infer_next_token([token], temperature=0.0, new_request=(i==0))
        print(f"Result: {result} ({encoding.decode([result]) if result else 'None'})")
    
    # Now try to get content
    print(f"\n--- Trying to get content ---")
    result = infer_next_token(tokens, temperature=0.0, new_request=False)
    print(f"Content result: {result} ({encoding.decode([result]) if result else 'None'})")

if __name__ == "__main__":
    debug_learning()
