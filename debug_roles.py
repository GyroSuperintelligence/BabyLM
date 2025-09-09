#!/usr/bin/env python3
from baby.tokenizer import get_tokenizer
from openai_harmony import StreamableParser, Role

def debug_roles():
    print("=== DEBUGGING ROLE DETECTION ===")
    
    # Get tokenizer
    encoding = get_tokenizer()
    
    # Create parser like the inference does
    parser = StreamableParser(encoding, role=Role.SYSTEM)
    print(f"Initial parser role: {parser.current_role}")
    print(f"Initial parser channel: {parser.current_channel}")
    
    # Test with "algorithms are"
    tokens = encoding.encode("algorithms are")
    print(f"Input tokens: {tokens}")
    
    # Process tokens and check role
    for i, token in enumerate(tokens):
        print(f"\nProcessing token {i+1}: {token} ({encoding.decode([token])})")
        parser.process(token)
        print(f"Parser role: {parser.current_role}")
        print(f"Parser channel: {parser.current_channel}")
        
        # Check if this would be considered user role
        is_user = (parser.current_role == Role.USER or 
                  parser.current_role == "user" or 
                  getattr(parser.current_role, "value", None) == "user")
        print(f"Is user role: {is_user}")

if __name__ == "__main__":
    debug_roles()
