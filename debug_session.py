#!/usr/bin/env python3
from baby.responses_api.inference.gyro import setup_model
from baby.tokenizer import get_tokenizer

def debug_session():
    print("=== DEBUGGING SESSION STATE ===")
    
    # Get tokenizer
    encoding = get_tokenizer()
    
    # Setup model
    infer_next_token = setup_model(encoding, "baby/config.json")
    
    # Test with "algorithms are"
    tokens = encoding.encode("algorithms are")
    print(f"Input tokens: {tokens}")
    
    # Process tokens and check session state
    for i, token in enumerate(tokens):
        print(f"\nProcessing token {i+1}: {token} ({encoding.decode([token])})")
        result = infer_next_token([token], temperature=0.0, new_request=(i==0))
        print(f"Result: {result} ({encoding.decode([result]) if result else 'None'})")
    
    # Check if we can access the session state
    # This is tricky since it's internal to the setup_model function
    print(f"\n--- Session state is internal to setup_model ---")
    print("The issue is that learning is not happening properly.")
    print("The model state is not being updated with the input tokens.")

if __name__ == "__main__":
    debug_session()
