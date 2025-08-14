#!/usr/bin/env python3
"""
Pack existing gyro files into a single .gyro.safetensors file.
This avoids re-converting from the original safetensors files.
"""

import os
import json
from pathlib import Path
from typing import Dict
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def pack_gyro_files_to_single(gyro_dir: str, output_file: str) -> str:
    """
    Pack existing gyro files into a single .gyro.safetensors file.
    
    Args:
        gyro_dir: Directory containing individual gyro .safetensors files
        output_file: Output path for the single packed file
    
    Returns:
        Path to the created single file
    """
    gyro_path = Path(gyro_dir)
    if not gyro_path.exists():
        raise FileNotFoundError(f"Gyro directory not found: {gyro_dir}")
    
    # Find all gyro safetensors files
    gyro_files = list(gyro_path.glob("*.safetensors"))
    if not gyro_files:
        raise FileNotFoundError(f"No gyro files found in {gyro_dir}")
    
    print(f"Found {len(gyro_files)} gyro files to pack")
    
    # Collect all tensors
    tensors_out: Dict[str, torch.Tensor] = {}
    metadata = {}
    
    for gyro_file in gyro_files:
        print(f"Processing {gyro_file.name}...")
        
        with safe_open(str(gyro_file), framework="pt", device="cpu") as f:
            # Get file metadata
            file_meta = f.metadata() or {}
            if not metadata and file_meta:
                # Use metadata from first file as base
                metadata.update(file_meta)
            
            # Copy all tensors from this file
            for key in f.keys():
                tensor = f.get_tensor(key)
                tensors_out[key] = tensor
    
    # Update metadata for single file format
    metadata.update({
        "gyro_pack": "1",
        "packed_from": gyro_dir,
        "num_source_files": str(len(gyro_files))
    })
    
    # Save the packed file
    print(f"Saving packed file to {output_file}...")
    save_file(tensors_out, output_file, metadata=metadata)
    
    print(f"✓ Successfully packed {len(tensors_out)} tensors into {output_file}")
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pack existing gyro files into a single file")
    parser.add_argument("gyro_dir", help="Directory containing gyro files")
    parser.add_argument("-o", "--output", help="Output file path (default: model.gyro.safetensors in parent dir)")
    
    args = parser.parse_args()
    
    gyro_dir = args.gyro_dir
    if args.output:
        output_file = args.output
    else:
        # Default to model.gyro.safetensors in the parent directory
        parent_dir = Path(gyro_dir).parent
        output_file = str(parent_dir / "model.gyro.safetensors")
    
    try:
        pack_gyro_files_to_single(gyro_dir, output_file)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)