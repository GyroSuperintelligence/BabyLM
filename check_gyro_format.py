#!/usr/bin/env python3
"""
Check the format consistency of converted gyro files.
"""

import os
from safetensors import safe_open
from pathlib import Path

def check_gyro_format():
    gyro_dir = Path("memories/models/gpt-oss-20b/gyro")
    
    if not gyro_dir.exists():
        print(f"Gyro directory not found: {gyro_dir}")
        return False
    
    gyro_files = list(gyro_dir.glob("*.safetensors"))
    print(f"Found {len(gyro_files)} gyro files")
    
    # Check first 5 files for format consistency
    sample_files = gyro_files[:5]
    
    format_consistent = True
    expected_keys_pattern = None
    
    for i, gyro_file in enumerate(sample_files):
        try:
            with safe_open(str(gyro_file), framework="pt", device="cpu") as f:
                keys = list(f.keys())
                metadata = f.metadata() or {}
                
                print(f"\nFile {i+1}: {gyro_file.name}")
                print(f"  Keys: {keys}")
                print(f"  Gyro flag: {metadata.get('gyro', 'missing')}")
                print(f"  Codec: {metadata.get('codec', 'missing')}")
                
                # Check if keys follow expected pattern: [tensor_name.gyro, tensor_name.meta]
                if len(keys) == 2:
                    gyro_key = [k for k in keys if k.endswith('.gyro')]
                    meta_key = [k for k in keys if k.endswith('.meta')]
                    
                    if len(gyro_key) == 1 and len(meta_key) == 1:
                        tensor_name_gyro = gyro_key[0][:-5]  # remove .gyro
                        tensor_name_meta = meta_key[0][:-5]  # remove .meta
                        
                        if tensor_name_gyro == tensor_name_meta:
                            print(f"  ✓ Format correct: {tensor_name_gyro}")
                        else:
                            print(f"  ✗ Tensor name mismatch: {tensor_name_gyro} vs {tensor_name_meta}")
                            format_consistent = False
                    else:
                        print(f"  ✗ Unexpected key pattern")
                        format_consistent = False
                else:
                    print(f"  ✗ Expected 2 keys, got {len(keys)}")
                    format_consistent = False
                
                # Check gyro metadata
                if metadata.get('gyro') != '1':
                    print(f"  ✗ Missing or incorrect gyro flag")
                    format_consistent = False
                    
        except Exception as e:
            print(f"  ✗ Error reading {gyro_file.name}: {e}")
            format_consistent = False
    
    print(f"\n{'='*50}")
    if format_consistent:
        print("✓ All checked files have consistent gyro format")
        print("✓ Each file contains one tensor with .gyro and .meta keys")
        print("✓ All files have proper gyro metadata")
    else:
        print("✗ Format inconsistencies detected")
    
    return format_consistent

if __name__ == "__main__":
    check_gyro_format()