#!/usr/bin/env python3
"""Check what keys are actually in model_weights after loading."""

import sys
sys.path.append('.')

from pathlib import Path
from kernel.gyro_head import GyroHead

def check_model_keys():
    """Check the actual keys in model_weights."""
    
    try:
        # Create a GyroHead instance to load the weights
        print("[info] Creating GyroHead instance...")
        gyro = GyroHead()
        
        print(f"[info] Model weights keys ({len(gyro.model_weights)}):")
        for i, key in enumerate(gyro.model_weights.keys()):
            print(f"  {i:3d}: {key}")
            if i > 20:  # Limit output
                print(f"  ... and {len(gyro.model_weights) - i - 1} more keys")
                break
        
        # Check specifically for lm_head related keys
        lm_head_keys = [k for k in gyro.model_weights.keys() if 'lm_head' in k]
        print(f"\n[info] LM head related keys: {lm_head_keys}")
        
        # Test the _layer_weight method
        try:
            weight = gyro._layer_weight('lm_head.weight')
            print(f"[SUCCESS] Found lm_head.weight: shape={weight.shape}, dtype={weight.dtype}")
        except KeyError as e:
            print(f"[ERROR] _layer_weight failed: {e}")
            
            # Try alternative searches
            for search_term in ['lm_head', 'weight']:
                matching_keys = [k for k in gyro.model_weights.keys() if search_term in k]
                print(f"[info] Keys containing '{search_term}': {matching_keys[:5]}")
        
    except Exception as e:
        print(f"[error] Failed to create GyroHead: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model_keys()